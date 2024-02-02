from dataset import DataModule

from pytorch_lightning import LightningModule, Trainer
from transformers import PreTrainedTokenizerFast

def inference_test():
    dm = DataModule(config, num_workers=1)

    if not config.use_slurm:
            trainer = Trainer(
                default_root_dir=model_dir, # main directory for run
                accelerator=config.device,
                devices=config.num_devices,
                max_epochs=config.epochs,
                val_check_interval=config.val_check_interval,
                accumulate_grad_batches=config.accumulate_grad_batches,
                sync_batchnorm=True,
                callbacks=[model_checkpoint]
                )
    else:
        trainer = Trainer(
            default_root_dir=model_dir, # main directory for run
            accelerator=config.device,
            num_nodes=config.num_nodes,
            devices=config.num_devices,
            strategy=config.strategy,
            max_epochs=config.epochs,
            val_check_interval=config.val_check_interval,
            accumulate_grad_batches=config.accumulate_grad_batches,
            sync_batchnorm=True,
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
            callbacks=[model_checkpoint]
            )

    print("\nDone training! Now testing model...")
    trainer.test(datamodule=dm) # Automatically loads best checkpoint, and tests with test dataloader