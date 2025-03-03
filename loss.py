import torch
from torch import nn
from torch.nn import functional as f

class TotalVariantionLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor): 
        # x = x.clone().detach()
        if x.max() > 1.:
            x = x / 255
        if len(x.shape) > 3:
            H, W = x.shape[-2:]
            x = x.view(-1, H, W)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0)
        elif len(x.shape) == 1:
            raise ValueError('Wrong Input Dimention')

        tv_h = ((x[:,1:,:] - x[:,:-1,:]).pow(2)).sum()
        tv_w = ((x[:,:,1:] - x[:,:,:-1]).pow(2)).sum()

        tv_loss = 1 - (tv_h + tv_w) / x.numel()
        # mean_loss = (x.mean() - .5).abs() * 2
        # var_loss = (x.var() - 1/12).abs() * 12

        return tv_loss - x.var()

class EntropyMetircs_2d(nn.Module):
    def __init__(
            self, 
            side: int = 1,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.side = side
        
    def forward(self, x: torch.Tensor): 
        H, W = x.shape[-2:]
        if x.max() < 1.:
            x = (x.clone().detach().view(-1, H, W) * 255).to(torch.uint8).float()
        else:
            x = x.clone().detach().view(-1, H, W).to(torch.uint8).float()

        x_pad = f.pad(x.clone().int(), (self.side, self.side, self.side, self.side), "constant", 0)
        mean_nb = torch.zeros(x.shape).to(x.device)

        for h in range(self.side*2+1):
            for w in range(self.side*2+1):
                if h == self.side*2 and w == self.side*2:
                    mean_nb += x_pad[:, h:, w:]
                elif w == self.side*2:
                    mean_nb += x_pad[:, h:(h-self.side*2), w:]
                elif h == self.side*2:
                    mean_nb += x_pad[:, h:, w:(w-self.side*2)]
                elif h == self.side and w == self.side:
                    continue
                else:
                    mean_nb += x_pad[:, h:(h-self.side*2), w:(w-self.side*2)]
        num = self.side*(2+self.side)
        for t in range(2*self.side + 1):
            if t < self.side:
                for i in range(t+1):# 点个数
                    mean_nb[t-i, i] /= num
                    mean_nb[i-1-t, i] /= num
                    mean_nb[t-i, -i-1] /= num
                    mean_nb[i-1-t, -i-1] /= num
                num = num + self.side + 1
            else:
                if t == 2*self.side:
                    mean_nb[self.side:-self.side, self.side:-self.side] /= num
                elif t < 2*self.side-self.side//2:
                    mean_nb[2*self.side-t:t-2*self.side, t-self.side] /= num
                    mean_nb[2*self.side-t:t-2*self.side, self.side-t-1] /= num
                    mean_nb[t-self.side, 2*self.side-t:t-2*self.side] /= num
                    mean_nb[self.side-t-1, 2*self.side-t:t-2*self.side] /= num
                else:
                    mean_nb[t-self.side+1:self.side-t-1, t-self.side] /= num
                    mean_nb[t-self.side+1:self.side-t-1, self.side-t-1] /= num
                    mean_nb[t-self.side, t-self.side:self.side-t] /= num
                    mean_nb[self.side-t-1, t-self.side:self.side-t] /= num
        x = torch.stack([x.int(), mean_nb.int()], dim=-1)

        loss = []
        for slide in x:
            _, counts = torch.unique(slide.view(-1, 2), return_counts=True, dim=0)
            p = counts / (H*W)
            loss.append((p*torch.log2(1/p)).sum())

        return torch.tensor(loss).mean()

class EntropyMetircs_1d(nn.Module):
    def __init__(
            self, 
            side: int = 1,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.side = side
        
    def forward(self, x: torch.Tensor): 
        H, W = x.shape[-2:]
        if x.max() < 1.:
            x = (x.clone().detach().view(-1, H, W) * 255).to(torch.uint8).float()
        else:
            x = x.clone().detach().view(-1, H, W).to(torch.uint8).float()

        loss = []
        for slide in x:
            _, counts = torch.unique(slide.view(-1), return_counts=True)
            p = counts / (H*W)
            loss.append((p*torch.log2(1/p)).sum())

        return torch.tensor(loss).mean()