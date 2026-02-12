import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional, Literal
from ema_pytorch import EMA
from torch.nn import Parameter
from einops import rearrange

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.autoencoders import AudioAutoencoder
from stable_audio_tools.models.discriminators import (
    EncodecDiscriminator, OobleckDiscriminator, DACGANLoss,
)
from stable_audio_tools.models.bottleneck import (
    VAEBottleneck, RVQBottleneck, DACRVQBottleneck, DACRVQVAEBottleneck,
    RVQVAEBottleneck, WassersteinBottleneck,
)
from stable_audio_tools.training.losses import (
    MelSpectrogramLoss, MultiLoss, AuralossLoss, ValueLoss, L1Loss,
    LossWithTarget, MSELoss, HubertLoss,
    # PESQMetric, # TODO move PESQ here?
)
from stable_audio_tools.training.losses import auraloss as auraloss
from stable_audio_tools.training.utils import (
    create_optimizer_from_config, create_scheduler_from_config, log_metric,
)

from .ctc_loss import CTCLossModule, PERModule

def trim_to_shortest(a, b):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:,:,:b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:,:,:a.shape[-1]]
    return a, b

class ProjectionHead(nn.Module):
    def __init__(self, latent_dim, proj_head_dim, mid_dim=256):
        super(ProjectionHead, self).__init__()
        self.proj_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(latent_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, proj_head_dim)
        )

    def forward(self, x):
        return self.proj_head(x)

class AutoencoderTrainingWrapper(pl.LightningModule):
    def __init__(self,
        autoencoder: AudioAutoencoder,
        loss_config: dict,
        eval_loss_config: dict,
        optimizer_configs: dict,
        sample_rate: int = 16000,
        lr: float = 1e-4,
        warmup_steps: int = 0,
        warmup_mode: Literal["adv", "full"] = "adv",
        encoder_freeze_on_warmup: bool = False,
        use_ema: bool = True,
        ema_copy = None,
        force_input_mono = False,
        latent_mask_ratio = 0.0,
        teacher_model: Optional[AudioAutoencoder] = None,
        clip_grad_norm = 0.0,
        encoder_mask_ratio = 0.0,
        use_ctc: bool = False,
        proj_head_dim: Optional[int] = None,
        detach_proj_head: bool = False,
    ):
        super().__init__()

        self.automatic_optimization = False
        self.autoencoder = autoencoder

        self.warmed_up = False
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm

        self.force_input_mono = force_input_mono
        self.teacher_model = teacher_model

        self.use_ctc = use_ctc
        self.proj_head_dim = proj_head_dim
        self.detach_proj_head = detach_proj_head
        self.projection_head = (
            ProjectionHead(self.autoencoder.latent_dim, self.proj_head_dim)
            if self.use_ctc and self.proj_head_dim is not None else
            nn.Identity()
        )

        self.optimizer_configs = optimizer_configs
        self.loss_config = loss_config

        # Spectral reconstruction loss
        self.sdstft = auraloss.MultiResolutionSTFTLoss(
            sample_rate=sample_rate, **loss_config['spectral']['config'])

        # Discriminator
        self.use_disc = True if 'discriminator' in loss_config else False
        self.discriminator = None
        if self.use_disc:
            if loss_config['discriminator']['type'] == 'oobleck':
                self.discriminator = OobleckDiscriminator(**loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'encodec':
                self.discriminator = EncodecDiscriminator(
                    in_channels=self.autoencoder.out_channels,
                    **loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'dac':
                self.discriminator = DACGANLoss(
                    channels=self.autoencoder.out_channels,
                    sample_rate=sample_rate,
                    **loss_config['discriminator']['config'])

        gen_loss_modules = []
        if self.use_disc:
            # Discriminator loss.
            self.losses_disc = MultiLoss([
                ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss'),
            ])

            # Adversarial and feature matching losses.
            gen_loss_modules += [
                ValueLoss(
                    key='loss_adv',
                    weight=self.loss_config['discriminator']['weights']['adversarial'],
                    name='loss_adv'),
                ValueLoss(
                    key='feature_matching_distance',
                    weight=self.loss_config['discriminator']['weights']['feature_matching'],
                    name='feature_matching_loss'),
            ]

        # Reconstruction loss
        gen_loss_modules += [AuralossLoss(self.sdstft,
            target_key='reals', input_key='decoded', name='mrstft_loss',
            weight=self.loss_config['spectral']['weights']['mrstft'],
            decay=self.loss_config['spectral'].get('decay', 1.0),
        )]

        if "mrmel" in loss_config:
            mrmel_weight = loss_config["mrmel"]["weights"]["mrmel"]
            if mrmel_weight > 0:
                mrmel_config = loss_config["mrmel"]["config"]
                self.mrmel = MelSpectrogramLoss(sample_rate,
                    n_mels=mrmel_config["n_mels"],
                    window_lengths=mrmel_config["window_lengths"],
                    pow=mrmel_config["pow"],
                    log_weight=mrmel_config["log_weight"],
                    mag_weight=mrmel_config["mag_weight"],
                )
                gen_loss_modules.append(LossWithTarget(
                    self.mrmel, "reals", "decoded",
                    name="mrmel_loss", weight=mrmel_weight,
                ))

        if "hubert" in loss_config:
            hubert_weight = loss_config["hubert"]["weights"]["hubert"]
            if hubert_weight > 0:
                hubert_cfg = (
                    loss_config["hubert"]["config"]
                    if "config" in loss_config["hubert"] else
                    dict()
                )
                self.hubert = HubertLoss(weight=1.0, **hubert_cfg)

                gen_loss_modules.append(LossWithTarget(
                    self.hubert, target_key = "reals", input_key = "decoded",
                    name="hubert_loss", weight=hubert_weight,
                    decay = loss_config["hubert"].get("decay", 1.0)
                ))

        if "l1" in loss_config["time"]["weights"]:
            if self.loss_config['time']['weights']['l1'] > 0.0:
                gen_loss_modules.append(L1Loss(
                    key_a='reals', key_b='decoded',
                    weight=self.loss_config['time']['weights']['l1'],
                    name='l1_time_loss',
                    decay = self.loss_config['time'].get('decay', 1.0),
                ))

        if "l2" in loss_config["time"]["weights"]:
            if self.loss_config['time']['weights']['l2'] > 0.0:
                gen_loss_modules.append(MSELoss(
                    key_a='reals', key_b='decoded',
                    weight=self.loss_config['time']['weights']['l2'],
                    name='l2_time_loss',
                    decay = self.loss_config['time'].get('decay', 1.0),
                ))

        if self.autoencoder.bottleneck is not None:
            gen_loss_modules += create_loss_modules_from_bottleneck(
                self.autoencoder.bottleneck, self.loss_config)

        self.encoder_mask_ratio = encoder_mask_ratio
        if encoder_mask_ratio > 0.0:
            gen_loss_modules.append(L1Loss(
                key_a='detached_latents', key_b='masked_latents',
                weight=1.0,
                name='encoder_mask_loss',
                decay = 1.0,
            ))

        if "ctc" in loss_config:
            ctc_weight = loss_config["ctc"]["weights"]["ctc"]
            if ctc_weight > 0:
                gen_loss_modules.append(CTCLossModule(
                    name = "ctc_loss",
                    target_key = "ctc_tgt",
                    input_key = "log_probs",
                    weight = ctc_weight,
                    decay = loss_config["ctc"].get("decay", 1.0),
                    blank_idx = loss_config["ctc"].get("blank_idx", 80)
                ))

        self.losses_gen = MultiLoss(gen_loss_modules)

        # Set up EMA for model weights
        self.autoencoder_ema = None
        self.use_ema = use_ema
        if self.use_ema:
            self.autoencoder_ema = EMA(
                self.autoencoder,
                ema_model=ema_copy,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1
            )

        self.latent_mask_ratio = latent_mask_ratio

        # evaluation losses & metrics
        self.eval_losses = torch.nn.ModuleDict()
        if eval_loss_config is not None:
            # if "pesq" in eval_loss_config:
            #     self.eval_losses["pesq"] = PESQMetric(sample_rate)
            if "stft"in eval_loss_config:
                self.eval_losses["stft"] = auraloss.STFTLoss(**eval_loss_config["stft"])
            if "sisdr" in eval_loss_config:
                self.eval_losses["sisdr"] = auraloss.SISDRLoss(**eval_loss_config["sisdr"])
            if "mel" in eval_loss_config:
                self.eval_losses["mel"] = auraloss.MelSTFTLoss(
                    sample_rate, **eval_loss_config["mel"])
            if "per" in eval_loss_config:
                self.eval_losses["per"] = PERModule(
                    target_key = "ctc_tgt",
                    input_key = "log_probs",
                    blank_idx = loss_config["ctc"].get("blank_idx", 80))

        self.validation_step_outputs = []

    def configure_optimizers(self):
        gen_params = list(self.autoencoder.parameters())

        if not self.use_disc:
            opt_gen = create_optimizer_from_config(
                self.optimizer_configs['autoencoder']['optimizer'], gen_params)
            if "scheduler" in self.optimizer_configs['autoencoder']:
                sched_gen = create_scheduler_from_config(
                    self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
                return [opt_gen], [sched_gen]
            return [opt_gen]

        # Using discriminator.
        opt_gen = create_optimizer_from_config(
            self.optimizer_configs['autoencoder']['optimizer'], gen_params)
        opt_disc = create_optimizer_from_config(
            self.optimizer_configs['discriminator']['optimizer'],
            self.discriminator.parameters())

        use_scheduler = (
            "scheduler" in self.optimizer_configs['autoencoder'] and
            "scheduler" in self.optimizer_configs['discriminator']
        )
        if use_scheduler:
            sched_gen = create_scheduler_from_config(
                self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
            sched_disc = create_scheduler_from_config(
                self.optimizer_configs['discriminator']['scheduler'], opt_disc)
            return [opt_gen, opt_disc], [sched_gen, sched_disc]
        return [opt_gen, opt_disc]

    def forward(self, reals):
        latents, encoder_info = self.autoencoder.encode(reals, return_info=True)
        decoded = self.autoencoder.decode(latents)
        return decoded

    def validation_step(self, batch, batch_idx):
        reals, _ = batch
        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if len(reals.shape) == 2:
            reals = reals.unsqueeze(1)

        loss_info = {}
        loss_info["reals"] = reals

        encoder_input = reals
        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input

        with torch.no_grad():
            if self.use_ctc:
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
                continuous_latents = encoder_info["pre_bottleneck_latents"]
                proj_features = rearrange(continuous_latents, "b c n -> b n c")
                proj_features = self.projection_head(
                    proj_features.detach()
                    if self.detach_proj_head else
                    proj_features
                )

                loss_info['log_probs'] = proj_features
                loss_info['ctc_tgt'] = batch[1]
            else:
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)

            loss_info["latents"] = latents
            loss_info.update(encoder_info)

            decoded = self.autoencoder.decode(latents)
            #Trim output to remove post-padding.
            decoded, reals = trim_to_shortest(decoded, reals)

            # Run evaluation metrics.
            val_loss_dict = {}
            for eval_key, eval_fn in self.eval_losses.items():
                if  eval_key == 'per':
                    loss_value = eval_fn(loss_info)
                else:
                    loss_value = eval_fn(decoded, reals)
                    if eval_key == "sisdr": loss_value = -loss_value

                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()

                val_loss_dict[eval_key] = loss_value

        self.validation_step_outputs.append(val_loss_dict)
        return val_loss_dict

    def on_validation_epoch_end(self):
        sum_loss_dict = {}
        for loss_dict in self.validation_step_outputs:
            for key, value in loss_dict.items():
                if key not in sum_loss_dict:
                    sum_loss_dict[key] = value
                else:
                    sum_loss_dict[key] += value

        for key, value in sum_loss_dict.items():
            val_loss = value / len(self.validation_step_outputs)
            val_loss = self.all_gather(val_loss).mean().item()
            log_metric(self.logger, f"val/{key}", val_loss)

        self.validation_step_outputs.clear()  # free memory

    def training_step(self, batch, batch_idx):
        reals, _ = batch

        log_dict = {}
        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if len(reals.shape) == 2:
            reals = reals.unsqueeze(1)

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True

        loss_info = {}
        loss_info["reals"] = reals
        encoder_input = reals

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input
        data_std = encoder_input.std()

        if self.warmed_up and self.encoder_freeze_on_warmup:
            with torch.no_grad():
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
        else:
            if self.use_ctc:
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
                continuous_latents = encoder_info["pre_bottleneck_latents"]
                proj_features = rearrange(continuous_latents, "b c n -> b n c")
                proj_features = self.projection_head(
                    proj_features.detach()
                    if self.detach_proj_head else
                    proj_features
                )

                loss_info['log_probs'] = proj_features
                loss_info['ctc_tgt'] = batch[1]
            else:
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)

        if self.encoder_mask_ratio > 0.0:
            masked_latents = self.autoencoder.encode(
                encoder_input, return_info=False, encoder_mask_ratio=self.encoder_mask_ratio)
            detached_latents = latents.detach()
            loss_info["masked_latents"] = masked_latents
            loss_info["detached_latents"] = detached_latents

        loss_info["latents"] = latents
        loss_info.update(encoder_info)

        # Encode with teacher model for distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_latents = self.teacher_model.encode(encoder_input, return_info=False)
                loss_info['teacher_latents'] = teacher_latents

        # Optionally mask out some latents for noise resistance
        if self.latent_mask_ratio > 0.0:
            mask = torch.rand_like(latents) < self.latent_mask_ratio
            latents = torch.where(mask, torch.zeros_like(latents), latents)

        decoded = self.autoencoder.decode(latents)
        #Trim output to remove post-padding
        decoded, reals = trim_to_shortest(decoded, reals)

        loss_info["decoded"] = decoded
        loss_info["reals"] = reals

        if self.autoencoder.out_channels == 2:
            loss_info["decoded_left"] = decoded[:, 0:1, :]
            loss_info["decoded_right"] = decoded[:, 1:2, :]
            loss_info["reals_left"] = reals[:, 0:1, :]
            loss_info["reals_right"] = reals[:, 1:2, :]

        # Distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_decoded = self.teacher_model.decode(teacher_latents)
                own_latents_teacher_decoded = self.teacher_model.decode(latents) #Distilled model's latents decoded by teacher
                teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents) #Teacher's latents decoded by distilled model

                loss_info['teacher_decoded'] = teacher_decoded
                loss_info['own_latents_teacher_decoded'] = own_latents_teacher_decoded
                loss_info['teacher_latents_own_decoded'] = teacher_latents_own_decoded

        if self.use_disc:
            if self.warmed_up:
                loss_dis, loss_adv, feature_matching_distance = self.discriminator.loss(reals=reals, fakes=decoded)
            else:
                loss_adv = torch.tensor(0.).to(reals)
                feature_matching_distance = torch.tensor(0.).to(reals)

                if self.warmup_mode == "adv":
                    loss_dis, _, _ = self.discriminator.loss(reals=reals, fakes=decoded)
                else:
                    loss_dis = torch.tensor(0.0).to(reals)

            loss_info["loss_dis"] = loss_dis
            loss_info["loss_adv"] = loss_adv
            loss_info["feature_matching_distance"] = feature_matching_distance

        opt_gen = None
        opt_disc = None
        if self.use_disc:
            opt_gen, opt_disc = self.optimizers()
        else:
            opt_gen = self.optimizers()

        lr_schedulers = self.lr_schedulers()
        sched_gen = None
        sched_disc = None

        if lr_schedulers is not None:
            if self.use_disc:
                sched_gen, sched_disc = lr_schedulers
            else:
                sched_gen = lr_schedulers

        # Train the discriminator
        use_disc = (
            self.use_disc
            and self.global_step % 2
            # Check warmup mode and if it is time to use discriminator.
            and (
                (self.warmup_mode == "full" and self.warmed_up)
                or self.warmup_mode == "adv")
        )
        if use_disc:
            loss, losses = self.losses_disc(loss_info)
            log_dict['train/disc_lr'] = opt_disc.param_groups[0]['lr']
            opt_disc.zero_grad()
            self.manual_backward(loss)

            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.clip_grad_norm)

            opt_disc.step()
            if sched_disc is not None:
                # sched step every step
                sched_disc.step()

        # Train the generator
        else:
            loss, losses = self.losses_gen(loss_info)
            if self.use_ema:
                self.autoencoder_ema.update()

            opt_gen.zero_grad()
            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.autoencoder.parameters(), self.clip_grad_norm)

            opt_gen.step()
            if sched_gen is not None:
                # scheduler step every step
                sched_gen.step()

            log_dict['train/loss'] =  loss.detach().item()
            log_dict['train/latent_std'] = latents.std().detach().item()
            log_dict['train/data_std'] = data_std.detach().item()
            log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']

        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach().item()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def export_model(self, path, use_safetensors=False):
        if self.autoencoder_ema is not None:
            model = self.autoencoder_ema.ema_model
        else:
            model = self.autoencoder

        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)

def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    losses = []

    if (
        isinstance(bottleneck, VAEBottleneck) or
        isinstance(bottleneck, DACRVQVAEBottleneck) or
        isinstance(bottleneck, RVQVAEBottleneck)
    ):
        try:
            kl_weight = loss_config['bottleneck']['weights']['kl']
        except:
            kl_weight = 1e-6

        kl_loss = ValueLoss(key='kl', weight=kl_weight, name='kl_loss')
        losses.append(kl_loss)

    if (
        isinstance(bottleneck, RVQBottleneck) or
        isinstance(bottleneck, RVQVAEBottleneck)
    ):
        quantizer_loss = ValueLoss(key='quantizer_loss', weight=1.0, name='quantizer_loss')
        losses.append(quantizer_loss)

    if isinstance(bottleneck, DACRVQBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck):
        codebook_loss = ValueLoss(key='vq/codebook_loss', weight=1.0, name='codebook_loss')
        commitment_loss = ValueLoss(key='vq/commitment_loss', weight=0.25, name='commitment_loss')
        losses.append(codebook_loss)
        losses.append(commitment_loss)

    if isinstance(bottleneck, WassersteinBottleneck):
        try:
            mmd_weight = loss_config['bottleneck']['weights']['mmd']
        except:
            mmd_weight = 100

        mmd_loss = ValueLoss(key='mmd', weight=mmd_weight, name='mmd_loss')
        losses.append(mmd_loss)

    return losses

def create_training_wrapper_from_config(model_config, model):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    ema_copy = None
    if training_config.get("use_ema", False):
        ema_copy = create_model_from_config(model_config)
        # Copy each weight to the ema copy
        for name, param in model.state_dict().items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            ema_copy.state_dict()[name].copy_(param)

    use_ema = training_config.get("use_ema", False)
    latent_mask_ratio = training_config.get("latent_mask_ratio", 0.0)

    teacher_model = training_config.get("teacher_model", None)
    if teacher_model is not None:
        teacher_model = create_model_from_config(teacher_model)
        teacher_model = teacher_model.eval().requires_grad_(False)

        teacher_model_ckpt = training_config.get("teacher_model_ckpt", None)
        if teacher_model_ckpt is not None:
            teacher_model.load_state_dict(torch.load(teacher_model_ckpt)["state_dict"])
        else:
            raise ValueError("teacher_model_ckpt must be specified if teacher_model is specified")

    return AutoencoderTrainingWrapper(
        model,
        lr=training_config.get("learning_rate", None),
        warmup_steps=training_config.get("warmup_steps", 0),
        encoder_freeze_on_warmup=training_config.get("encoder_freeze_on_warmup", False),
        sample_rate=model_config["sample_rate"],
        loss_config=training_config.get("loss_configs", None),
        eval_loss_config=training_config.get("eval_loss_configs", None),
        optimizer_configs=training_config.get("optimizer_configs", None),
        use_ema=use_ema,
        ema_copy=ema_copy if use_ema else None,
        force_input_mono=training_config.get("force_input_mono", False),
        latent_mask_ratio=latent_mask_ratio,
        teacher_model=teacher_model,
        encoder_mask_ratio=training_config.get("encoder_mask_ratio", 0.0),
        use_ctc=training_config.get("use_ctc", False),
        proj_head_dim=model_config["model"].get("proj_head_dim", False),
        detach_proj_head=model_config["model"].get("detach_proj_head", None),
    )
