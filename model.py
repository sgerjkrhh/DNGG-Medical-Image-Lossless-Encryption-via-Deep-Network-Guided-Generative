import os
from math import cos, sqrt, pi
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import lightning.pytorch as pl
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

from loss import *
from utils import *


class BaseBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int, 
            kernel_size: int = 3,
            stride: int = 1, 
            padding: int = 1,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.SiLU()
        self.bn = nn.BatchNorm2d(out_channels)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        return x

class UpBlock(nn.Module):
    def __init__(
            self,
            in_channel: int = 16,
            out_channel: int = 2,
        ) -> None:
        super().__init__()
        self.conv0 = BaseBlock(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1)
        self.conv1 = BaseBlock(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1)
        self.pool = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv0(x)
        x = self.conv1(x)

        return x

class New_model(pl.LightningModule):
    def __init__(
            self, 
            first_channel: int = 128,
            sample_num: int = 5,
            lr_max: float = 1e-3,
            lr_min: float = 1e-5,
            patch_size: int = 4,
            *args: torch.Any, 
            **kwargs: torch.Any
        ) -> None:
        super().__init__(*args, **kwargs)
        assert 2**sample_num < first_channel, f"Wrong Input Dim with sample_num: {sample_num} & first_channel: {first_channel}"

        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T = 15
        self.mode = 'xor'
        self.patch_size = [int(16 * 2**sample_num / patch_size) ** 2, patch_size, patch_size]
        self.init_size = [16 * 2**sample_num, 16 * 2**sample_num]

        self.check_dir = r'secret_pics'

        self.up = []
        for i in range(sample_num):
            if i:
                self.up.append(UpBlock(in_channel=in_channel, out_channel=in_channel//2))
                in_channel = in_channel // 2
            else:
                self.up.append(UpBlock(in_channel=1, out_channel=first_channel))
                in_channel = first_channel
        self.up.append(nn.Conv2d(in_channels=in_channel, out_channels=1,
                                 kernel_size=3, stride=1, padding=1))
        self.up = nn.ModuleList(self.up)

        self.loss_func = TotalVariantionLoss()
        self.entropy_1d = EntropyMetircs_1d()
        self.entropy_2d = EntropyMetircs_2d()

        self.metrics_func = EntropyMetircs_1d()

        self.optimizer = optim.AdamW

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = self.optimizer(params, lr=self.lr_max, weight_decay=5e-4)
        def lambda1(epoch): return ((cos((epoch % self.T)/self.T * pi)+1) /
                                    2 * (self.lr_max-self.lr_min)+self.lr_min)/self.lr_max
        scheduler = {
            'scheduler': LambdaLR(optimizer, lambda1),
            'name': 'learming_rate'
        }
        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor):
        for m in self.up:
            x = m(x)
        return x.sigmoid()
    
    def on_train_epoch_start(self) -> None:
        self.train_key = torch.tensor([])
        self.train_record = torch.tensor([])
        return super().on_train_epoch_start()
    
    def training_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        x, keys = batch
        y = self.forward(x).squeeze()
        # self.train_record = torch.cat((self.train_record.to(y.device), y.detach()), dim=0)
        mean_var = self.loss_func(y)
        # entropy_1d = 8 - self.entropy_1d(y)
        # entropy_2d = 16 - self.entropy_2d(y)
        # loss_cover = mean_var + entropy_1d + entropy_2d 
        
        # mean_var = []
        # entropy_1d = []
        # entropy_2d = []
        # for path in os.listdir(self.check_dir):
        #     pic = torch.tensor(cv2.imread(fr'{self.check_dir}\{path}', 0), dtype=torch.float32)
        #     stego = self.implant(y.clone(), pic.to(y.device), keys)

        #     mean_var.append(self.loss_func(stego))
        #     entropy_1d.append(8 - self.entropy_1d(stego))
        #     entropy_2d.append(16 - self.entropy_2d(stego))
        
        # mean_var = torch.tensor(mean_var).mean()
        # entropy_1d = torch.tensor(entropy_1d).mean()
        # entropy_2d = torch.tensor(entropy_2d).mean()
        # loss_stego = mean_var + entropy_1d + entropy_2d 
        
        self.log_dict({
            'train/loss_cover': mean_var,
            # 'train/loss_stego': loss_stego,
            })
        
        return mean_var

    def on_train_epoch_end(self) -> None:
        # cover_metrics = self.metrics_func(self.train_record)
        # stego_metrics = []
        # for path in os.listdir(self.check_dir):
        #     pic = torch.tensor(cv2.imread(fr'{self.check_dir}\{path}', 0), dtype=torch.float32) / 255
        #     stego_metrics.append(self.metrics_func(self.implant(self.train_record, pic.to(self.train_record.device))))

        # log = {
        #     'train/cover_entropy': cover_metrics,
        #     'train/stego_entropy': torch.tensor(stego_metrics).mean(),
        #     'step': torch.tensor(self.current_epoch, dtype=torch.float32)
        # }
        # self.log_dict(log)

        return super().on_train_epoch_end()
    
    def on_validation_epoch_start(self) -> None:
        self.val_key = torch.tensor([])
        self.val_record = torch.tensor([])
        return super().on_validation_epoch_start()
    
    def validation_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        x, keys = batch
        self.val_key = torch.cat((self.val_key.to(x.device), keys.detach()), dim=0)

        y = self.forward(x).squeeze()
        self.val_record = torch.cat((self.val_record.to(y.device), y.detach()), dim=0)

    def on_validation_epoch_end(self) -> None:
        cover_metrics = self.metrics_func(self.val_record)
        stego_metrics = []
        for path in os.listdir(self.check_dir):
            pic = torch.tensor(cv2.imread(fr'{self.check_dir}\{path}', 0), dtype=torch.float32) # / 255
            stego_metrics.append(self.metrics_func(self.implant(self.val_record, pic.to(self.val_record.device), self.val_key)))

        log = {
            'val/cover_entropy': cover_metrics,
            'val/stego_entropy': torch.tensor(stego_metrics).mean(),
            'step': torch.tensor(self.current_epoch, dtype=torch.float32)
        }
        self.log_dict(log)

        pic = self.show_pic(self.val_record, self.val_key)
        self.logger.experiment.add_image('val', pic, self.current_epoch / 5)

        return super().on_validation_epoch_end()
    
    def on_predict_epoch_start(self) -> None:
        self.predict_key = torch.tensor([])
        self.predict_record = torch.tensor([])
        return super().on_predict_epoch_start()
    
    def predict_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        x, keys = batch
        self.predict_key = torch.cat((self.predict_key.to(x.device), keys.detach()), dim=0)

        y = self.forward(x).squeeze().unsqueeze(0)
        self.predict_record = torch.cat((self.predict_record.to(y.device), y.detach()), dim=0)
    
    def on_predict_epoch_end(self) -> None:
        record = []
        for path in os.listdir(self.check_dir):
            pic = torch.tensor(cv2.imread(fr'{self.check_dir}\{path}', 0), dtype=torch.float32) # / 255
            record.append([path, self.metrics_func(self.implant(self.predict_record, pic.to(self.predict_record.device), self.predict_key))])
        record = sorted(record, key=lambda x: x[1], reverse=True)

        pic = self.show_pic(self.predict_record, self.predict_key)

        return super().on_predict_epoch_end()
    
    def transform(self, f: torch.Tensor):
        if self.mode == 'dct':
            if not hasattr(self, 'a'):
                self.a = nn.Parameter(torch.tensor(create_dct_para(self.patch_size[-1])), requires_grad=False)

            if isinstance(f, torch.Tensor):
                a = self.a.clone().to(f.device)
                return a @ f.view(-1, *self.patch_size) @ a.T
            else:
                a = self.a.clone().detach().cpu().numpy()
                return a @ f.reshape(-1, *self.patch_size) @ a.T
        
        elif self.mode == 'fft':
            f = f.clone().to(torch.complex64)
            return torch.fft.fft2(f)
        
        else:
            raise ValueError("Unknown Mode")
        
    def itransform(self, f: torch.Tensor):
        if self.mode == 'dct':

            if isinstance(f, torch.Tensor):
                a = self.a.clone().to(f.device)
                return (a.T @ f @ a).view(-1, *self.init_size)
            else:
                a = self.a.clone().detach().cpu().numpy()
                return (a.T @ f @ a).reshape(-1, *self.init_size)
        
        elif self.mode == 'fft':
            f = f.clone().to(torch.complex64)
            return torch.fft.ifft2(f).real.float()
    
    def implant(self, cover, secret, keys):
        if self.mode in ['dct', 'fft']:
            if isinstance(secret, torch.Tensor):
                secret = secret.to(cover.device)
            # cover_trans, secret_trans = self.transform(cover), self.transform(secret)
            # cover_trans[cover_trans==0.] = 1.
            # stego_trans = torch.abs(cover_trans * secret_trans).sqrt() * torch.sign(secret_trans)
            # return self.itransform(stego_trans)
            return self.itransform((self.transform(cover) + self.transform(secret))/2)

        elif self.mode in ['xor']:
            if cover.max() < 1.:
                cover = (cover * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).float()
            cover_b, secret_b = to_binary(cover), to_binary(secret)

            if secret_b.numel() == 8*self.init_size[0]**2:
                secret_b = torch.cat([secret_b.clone().view(-1)[idx.cpu().int()].view(1, *secret_b.shape) for idx in keys])
            else:
                secret_b = torch.cat([img.view(-1)[idx.cpu().int()].view(1, *img.shape) for idx, img in zip(keys, secret_b.transpose(0, 1).contiguous())])
            secret_b = secret_b.transpose(0, 1)

            stego_b = approx_xor(cover_b, secret_b.to(cover_b.device))
            stego_b = torch.cat([img.view(-1)[idx.cpu().int()].view(1, *img.shape) for idx, img in zip(keys, stego_b.transpose(0, 1).contiguous())]).transpose(0, 1)
            return to_uint8(stego_b)

        
    def recover(self, stego, cover, keys):
        if self.mode in ['dct', 'fft']:
            if isinstance(cover, torch.Tensor):
                cover = cover.to(stego.device)
            # stego_trans, cover_trans = self.transform(stego), self.transform(cover)
            # cover_trans[cover_trans==0.] = 1.
            # recover_trans = torch.abs(stego_trans**2 / cover_trans) * torch.sign(stego_trans)
            # return self.itransform(recover_trans)
            return self.itransform(self.transform(stego)*2 - self.transform(cover))
        
        elif self.mode in ['xor']:
            if cover.max() < 1.:
                cover = (cover * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).float()
            cover_b, stego_b = to_binary(cover), to_binary(stego)

            stego_b = torch.cat([img.view(-1)[torch.argsort(idx.cpu().int())].view(1, *img.shape) for idx, img in zip(keys, stego_b.transpose(0, 1).contiguous())]).transpose(0, 1)
            stego_b = approx_xor(cover_b, stego_b.to(cover_b.device))

            if stego_b.numel() == 8*self.init_size[0]**2:
                stego_b = torch.cat([stego_b.clone().view(-1)[torch.argsort(idx.cpu().int())].view(1, *stego_b.shape) for idx in keys])
            else:
                stego_b = torch.cat([img.view(-1)[torch.argsort(idx.cpu().int())].view(1, *img.shape) for idx, img in zip(keys, stego_b.transpose(0, 1).contiguous())])
            stego_b = stego_b.transpose(0, 1)

            return to_uint8(stego_b)

    
    def show_pic(self, y: torch.Tensor, keys, nrows: int = 8):
        paths = np.array([fr'{self.check_dir}\{path}' for path in os.listdir(self.check_dir)])

        nrows = nrows if len(paths) > nrows else len(paths)
        idx = np.arange(len(paths))
        np.random.shuffle(idx)
        idx = idx[:nrows]
        secrets = torch.cat([torch.tensor(cv2.imread(path, 0)).unsqueeze(0) for path in paths[idx]])

        idx = np.arange(len(y))
        np.random.shuffle(idx)
        idx = idx[:nrows]
        covers = (y[idx].clone()*255).to(torch.uint8)
        keys = keys[idx].clone()

        stegos = self.implant(covers.float(), secrets.float(), keys).clamp(0, 255).to(torch.uint8)
        recoverd = self.recover(stegos.clone().float(), covers.float(), keys).clamp(0, 255).to(torch.uint8)

        img = torch.cat([secrets, recoverd.cpu(), covers.cpu(), stegos.cpu()], dim=0).unsqueeze(1)

        # for i in range(8):
        #     plt.imshow(img[i, 0], cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(f'paper_pic/{paths[i]}_origin.jpg', bbox_inches='tight')
        #     plt.close()

        #     plt.hist(img[i, 0].flatten(), color='skyblue', bins=256, range=[0, 255])
        #     plt.savefig(f'paper_pic/{paths[i]}_origin_hist.jpg', bbox_inches='tight')
        #     plt.close()
            
        #     x, y = np.array([j[:-1] for j in img[i, 0]]).reshape(-1), np.array([j[1:] for j in img[i, 0]]).reshape(-1)
        #     plt.scatter(x[::25], y[::25], marker='.', alpha=.3, s=10)
        #     plt.savefig(f'paper_pic/{paths[i]}_origin_H.jpg', bbox_inches='tight')
        #     plt.close()
            
        #     x, y = np.array([j[:-1] for j in img[i, 0].transpose(0, 1)]).reshape(-1), np.array([j[1:] for j in img[i, 0].transpose(0, 1)]).reshape(-1)
        #     plt.scatter(x[::25], y[::25], marker='.', alpha=.3, s=10)
        #     plt.savefig(f'paper_pic/{paths[i]}_origin_V.jpg', bbox_inches='tight')
        #     plt.close()
            
        #     diag = [[img[i, 0, j-k, k] for k in range(j+1)] for j in range(1, img[i, 0].shape[0])] + [[img[i, 0, img[i, 0].shape[1]-1-k, k+1+j] for k in range(img[i, 0].shape[1]-1-j)] for j in range(img[i, 0].shape[1]-2)]
        #     x, y = np.array([k for j in diag for k in j[:-1]]), np.array([k for j in diag for k in j[1:]])
        #     plt.scatter(x[::25], y[::25], marker='.', alpha=.3, s=10)
        #     plt.savefig(f'paper_pic/{paths[i]}_origin_D.jpg', bbox_inches='tight')
        #     plt.close()

        #     plt.imshow(img[i+8*1, 0], cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(f'paper_pic/{paths[i]}_recover.jpg', bbox_inches='tight')
        #     plt.close()

        #     plt.imshow(img[i+8*2, 0], cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(f'paper_pic/{paths[i]}_cover.jpg', bbox_inches='tight')
        #     plt.close()

        #     plt.imshow(img[i+8*3, 0], cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(f'paper_pic/{paths[i]}_stego.jpg', bbox_inches='tight')
        #     plt.close()

        #     plt.hist(img[i+8*3, 0].flatten(), color='skyblue', bins=256, range=[0, 255])
        #     plt.savefig(f'paper_pic/{paths[i]}_stego_hist.jpg', bbox_inches='tight')
        #     plt.close()

        #     x, y = np.array([j[:-1] for j in img[i+8*3, 0]]).reshape(-1), np.array([j[1:] for j in img[i+8*3, 0]]).reshape(-1)
        #     plt.scatter(x[::25], y[::25], marker='.', alpha=.3, s=10)
        #     plt.savefig(f'paper_pic/{paths[i]}_stego_H.jpg', bbox_inches='tight')
        #     plt.close()
            
        #     x, y = np.array([j[:-1] for j in img[i+8*3, 0].transpose(0, 1)]).reshape(-1), np.array([j[1:] for j in img[i+8*3, 0].transpose(0, 1)]).reshape(-1)
        #     plt.scatter(x[::25], y[::25], marker='.', alpha=.3, s=10)
        #     plt.savefig(f'paper_pic/{paths[i]}_stego_V.jpg', bbox_inches='tight')
        #     plt.close()
            
        #     diag = [[img[i+8*3, 0, j-k, k] for k in range(j+1)] for j in range(1, img[i+8*3, 0].shape[0])] + [[img[i+8*3, 0, img[i+8*3, 0].shape[1]-1-k, k+1+j] for k in range(img[i+8*3, 0].shape[1]-1-j)] for j in range(img[i+8*3, 0].shape[1]-2)]
        #     x, y = np.array([k for j in diag for k in j[:-1]]), np.array([k for j in diag for k in j[1:]])
        #     plt.scatter(x[::25], y[::25], marker='.', alpha=.3, s=10)
        #     plt.savefig(f'paper_pic/{paths[i]}_stego_D.jpg', bbox_inches='tight')
        #     plt.close()

        # stegos_5_b = stegos.clone()
        # stegos_5_b[:, 243:269, 243:269] = 0
        # recoverd_5_b = self.recover(stegos_5_b.clone().float(), covers.float(), keys).clamp(0, 255).to(torch.uint8).cpu()
        # stegos_5_b = stegos_5_b.cpu()
        # for i in range(8):
        #     plt.imshow(stegos_5_b[i], cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(f'paper_pic/{paths[i]}_stego_5_black.jpg', bbox_inches='tight')
        #     plt.close()

        #     plt.imshow(recoverd_5_b[i], cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(f'paper_pic/{paths[i]}_recover_5_balck.jpg', bbox_inches='tight')
        #     plt.close()
            
        #     x = (recoverd_5_b[i]-recoverd[i]).abs()
        #     plt.imshow(x, cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(f'paper_pic/{paths[i]}_recover_5_black_diff.jpg', bbox_inches='tight')
        #     plt.close()

        #     plt.hist(x.flatten(), color='skyblue', bins=256, range=[1, 255])
        #     plt.savefig(f'paper_pic/{paths[i]}_recover_5_black_diff-hist.jpg', bbox_inches='tight')
        #     plt.close()

        # def add_gaussian_noise(image, mean=0, std=0.1):
        #     device = image.device
        #     if image.max() > 1.:
        #         image = image.float().cpu() / 255
        #     noise = torch.randn(image.size()) * std + mean
        #     noisy_image = image + noise
        #     noisy_image = (noisy_image * 255).clamp(0, 255).to(torch.uint8).to(device)
        #     return noisy_image

        # def add_salt_and_pepper_noise(image, prob=0.05):
        #     device = image.device
        #     if image.max() > 1.:
        #         image = image.float().cpu() / 255
        #     noisy_image = TF.to_tensor(image)
        #     img_size = noisy_image.size()
        #     num_salt = int(prob * torch.prod(torch.tensor(img_size)))
        #     coords = [torch.randint(0, i, (num_salt,)) for i in img_size]
        #     noisy_image[coords] = 1
        #     num_pepper = int(prob * torch.prod(torch.tensor(img_size)))
        #     coords = [torch.randint(0, i, (num_pepper,)) for i in img_size]
        #     noisy_image[coords] = 0
        #     noisy_image = (noisy_image * 255).clamp(0, 255).to(torch.uint8).to(device)
        #     return noisy_image

        # stegos_g = add_gaussian_noise(stegos.clone(), 0., .001)
        # recoverd_g = self.recover(stegos_g.clone().float(), covers.float(), keys).clamp(0, 255).to(torch.uint8).cpu()
        # stegos_g = stegos_g.cpu()
        # for i in range(8):
        #     plt.imshow(stegos_g[i], cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(f'paper_pic/{paths[i]}_stego_G_0.001.jpg', bbox_inches='tight')
        #     plt.close()

        #     plt.imshow(recoverd_g[i], cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(f'paper_pic/{paths[i]}_recover_G_0.001.jpg', bbox_inches='tight')
        #     plt.close()
            
        #     x = (recoverd_g[i]-recoverd[i]).abs()
        #     plt.imshow(x, cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(f'paper_pic/{paths[i]}_recover_G_0.001_diff.jpg', bbox_inches='tight')
        #     plt.close()

        #     plt.hist(x.flatten(), color='skyblue', bins=256, range=[1, 255])
        #     plt.savefig(f'paper_pic/{paths[i]}_recover_G_0.001_diff-hist.jpg', bbox_inches='tight')
        #     plt.close()
        
        return make_grid(img, nrow=nrows, padding=2, pad_value=1)



if __name__ == "__main__":
    os.system("clear")
    model = New_model()
    print(model)
