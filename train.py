import copy
import json
import os
from typing import Optional

import pytorch_lightning as pl
from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import (
    copy_state_dict,
    load_ckpt_state_dict,
    remove_weight_norm_from_model,
)

from stable_codec.data.dataset import create_dataloader_from_config
from stable_codec.training_demo import create_demo_callback_from_config
from stable_codec.training_module import create_training_wrapper_from_config


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main():
    args = get_all_args()
    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    print(f"Setting random seed: `{seed}`.")
    pl.seed_everything(seed, workers=True)

    save_dir = args.save_dir
    ckpt_path: Optional[str] = None
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
        print(f"Using user-provided checkpoint: `{ckpt_path}`.")

    with open(args.model_config) as f:
        model_config = json.load(f)
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    val_dl = None
    val_dataset_config = None

    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=False,
        )

    model = create_model_from_config(model_config)
    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)
    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    if args.project is None:
        project_name = args.name
        run_name = None
    else:
        project_name = args.project
        run_name = args.name

    exc_callback = ExceptionCallback()

    logger = None
    ckpt_dir = save_dir
    if args.logger == 'wandb':
        logger = pl.loggers.WandbLogger(
            name=run_name, project=project_name,
            save_dir=save_dir)
        logger.watch(training_wrapper, log_freq=1000)

        ckpt_dir = os.path.join(
            save_dir, logger.experiment.project,
            logger.experiment.id, "checkpoints")
    elif args.logger == 'comet':
        logger = pl.loggers.CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            experiment_name=run_name, project_name=project_name,
            save_dir=save_dir)

        ckpt_dir = os.path.join(
            save_dir, project_name, logger.experiment.id, "checkpoints")

    print(f"Checkpoint dir: `{ckpt_dir}`.")
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every,
        dirpath=ckpt_dir, save_top_k=args.save_top_k)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    demo_dl = copy.deepcopy(val_dl if args.val_dataset_config else train_dl)
    demo_callback = create_demo_callback_from_config(model_config, demo_dl=demo_dl)

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})

    if args.logger == 'wandb':
        push_wandb_config(logger, args_dict)
    elif args.logger == 'comet':
        logger.log_hyperparams(args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
        logger=logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
    )
    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
