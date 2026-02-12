import os
import torch
import torchaudio
import pytorch_lightning as pl

from einops import rearrange
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from stable_audio_tools.models.autoencoders import (
    fold_channels_into_batch, unfold_channels_from_batch,
)
from stable_audio_tools.training.utils import (
    log_image, log_point_cloud, logger_project_name, log_audio,
)
from stable_audio_tools.interface.aeiou import (
    audio_spectrogram_image, tokens_spectrogram_image,
)

def trim_to_shortest(a, b):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:,:,:b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:,:,:a.shape[-1]]
    return a, b

class AutoencoderDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every = 2000,
        sample_size = 65536,
        sample_rate = 16000,
        max_demos = 8,
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_dl = demo_dl
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.max_demos = max_demos

    @rank_zero_only
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if (
            (trainer.global_step - 1) % self.demo_every != 0 or
            self.last_demo_step == trainer.global_step
        ):
            return

        self.last_demo_step = trainer.global_step
        module.eval()

        try:
            demo_iter = iter(self.demo_dl)
            demo_reals, _ = next(demo_iter)

            # Remove extra dimension added by WebDataset
            if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
                demo_reals = demo_reals[0]

            # Limit the number of demo samples
            if demo_reals.shape[0] > self.max_demos:
                demo_reals = demo_reals[:self.max_demos,...]

            encoder_input = demo_reals
            encoder_input = encoder_input.to(module.device)

            if module.force_input_mono:
                encoder_input = encoder_input.mean(dim=1, keepdim=True)

            demo_reals = demo_reals.to(module.device)

            with torch.no_grad():
                if module.use_ema:
                    latents = module.autoencoder_ema.ema_model.encode(encoder_input)
                    fakes = module.autoencoder_ema.ema_model.decode(latents)
                else:
                    latents = module.autoencoder.encode(encoder_input)
                    fakes = module.autoencoder.decode(latents)

            #Trim output to remove post-padding.
            fakes, demo_reals = trim_to_shortest(fakes.detach(), demo_reals)

            # Visualize discriminator sensitivity.
            if module.discriminator is not None:
                window = torch.kaiser_window(512).to(fakes.device)
                stft_kwargs = {
                    "n_fft": 512,
                    "hop_length": 128,
                    "win_length": 512,
                    "window": window,
                    "center": True,
                }

                fakes_stft = torch.stft(fold_channels_into_batch(fakes),
                    return_complex=True, **stft_kwargs)
                fakes_stft.requires_grad = True
                fakes_signal = unfold_channels_from_batch(
                    torch.istft(fakes_stft, **stft_kwargs), fakes.shape[1])

                real_stft = torch.stft(fold_channels_into_batch(demo_reals),
                    return_complex=True, **stft_kwargs)
                reals_signal = unfold_channels_from_batch(
                    torch.istft(real_stft, **stft_kwargs), demo_reals.shape[1])

                _, loss, _ = module.discriminator.loss(reals_signal, fakes_signal)
                fakes_stft.retain_grad()
                loss.backward()
                grads = unfold_channels_from_batch(fakes_stft.grad.detach().abs(), fakes.shape[1])

                log_image(trainer.logger, 'disciminator_sensitivity',
                    tokens_spectrogram_image(grads.mean(dim=1).log10(),
                    title='Discriminator Sensitivity', symmetric=False))
                opts = module.optimizers()
                opts[0].zero_grad()
                opts[1].zero_grad()

            #Interleave reals and fakes
            reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')
            # Put the demos together
            reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')

            data_dir = os.path.join(
                trainer.logger.save_dir, logger_project_name(trainer.logger),
                trainer.logger.experiment.id, "media")
            os.makedirs(data_dir, exist_ok=True)
            filename = os.path.join(data_dir, f'recon_{trainer.global_step:08}.wav')

            reals_fakes = reals_fakes.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, reals_fakes, self.sample_rate)

            log_audio(trainer.logger, 'recon', filename, self.sample_rate)
            log_point_cloud(trainer.logger, 'embeddings_3dpca', latents)
            log_image(trainer.logger, 'embeddings_spec', tokens_spectrogram_image(latents))
            log_image(trainer.logger, 'recon_melspec_left', audio_spectrogram_image(reals_fakes))
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            module.train()

def create_demo_callback_from_config(model_config, **kwargs):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    demo_config = training_config.get("demo", {})
    return AutoencoderDemoCallback(
        demo_every=demo_config.get("demo_every", 2000),
        sample_size=model_config["sample_size"],
        sample_rate=model_config["sample_rate"],
        **kwargs
    )
