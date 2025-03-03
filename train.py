
import os
import argparse

import torch
import lightning.pytorch as pl
import tensorboard

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor, EarlyStopping
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from data import MyDataModule
from model import New_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
torch.set_float32_matmul_precision('high')

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--weights', type=str, help='',
                        default=r'D:\information_security\DNGG\code\Train\saved\04\04_stego=7.999099.ckpt')
    parser.add_argument('--is_train', type=bool, default=False)
    parser.add_argument('--pic_size', type=int, default=224)
    parser.add_argument('--time', type=int, default=None)

    return parser.parse_args()


def main(args):
    save_dir = f"saved"
    if args.is_train:
        no = 0
        if os.path.exists(save_dir):
            while os.path.exists(fr'{save_dir}/{str(no).zfill(2)}'):
                no += 1
    else:
        if os.path.exists(str(args.weights)):
            try:
                no = int(os.path.basename(args.weights)[:2])
            except:
                no = 0
        else:
            raise KeyError('Predict Weight doesn\'t exist')
    save_dir = f"{save_dir}/{str(no).zfill(2)}"
    
    dm_paras = {
        'pic_size': 16,
        'batch_size': 16,
        'num_workers': 12,
    }
    dm = MyDataModule(**dm_paras)
    model_paras = {
        'first_channel': 128,
        'sample_num': 5,
        'lr_max': 1e-5,
        'lr_min': 1e-7,
    }

    process_callback = RichProgressBar(
        theme=RichProgressBarTheme(
            description='green_yellow',
            progress_bar='green1',
            progress_bar_finished='blue1',
            progress_bar_pulse='red1',
            batch_progress='green_yellow',
            time = 'grey82',
            processing_speed='grey30',
            metrics='grey82'
        )
    )

    if args.is_train:
        dm.setup("fit")
        if args.weights != "":
            try:
                model = New_model.load_from_checkpoint(args.weights, **model_paras)
            except:
                raise("Weights cannot be loaded")
        else:
            model = New_model(**model_paras)
            
        checkpoint_callback = ModelCheckpoint(monitor='val/stego_entropy',
                                              dirpath=save_dir,
                                              filename=f'{str(no).zfill(2)}_'+'stego={val/stego_entropy:.6f}',
                                              save_top_k=3,
                                              save_last=False,
                                              mode="max",
                                              auto_insert_metric_name=False)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        earlystop_callback = EarlyStopping(monitor='val/stego_entropy', mode="max", patience=3)
        logger = TensorBoardLogger(
            save_dir=save_dir,
            name=f'',
            version=f'log',
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs, 
            check_val_every_n_epoch=5,
            log_every_n_steps=1, 
            logger=logger,
            callbacks=[checkpoint_callback, process_callback, lr_monitor, earlystop_callback],
            reload_dataloaders_every_n_epochs=1,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, datamodule=dm)

    else:
        dm.setup("predict")
        if args.weights != "":
            try:
                model = New_model.load_from_checkpoint(args.weights, **model_paras)
            except:
                raise("Weights cannot be loaded")
            
        trainer = pl.Trainer(
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.predict(model, datamodule=dm)


if __name__ == "__main__":
    os.system("clear")
    args = create_args()
    main(args)