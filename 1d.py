#  pix2pix‑GAN (U‑Net + PatchGAN) with two scalar conditions Da & Pe
# Folder layout:
#   project/
#    ├─ ppdp.py
#    ├─ mapping.xlsx    # gray | Da | Pe | colour
#    ├─ gray_imgs/ *.png
#    └─ color_imgs/*.png
#
from pathlib import Path
from typing import Union
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils as vutils

# ---------- 1. Dataset ----------
class Pixel2PixelDataset(Dataset):
    """return (gray_tensor, colour_tensor, params[Da,Pe])"""
    def __init__(self, mapping: Union[str, Path] = 'mapping.xlsx',
                 root: Union[str, Path] = '.',
                 size: int = 256,
                 gray_dir: str = 'gray_imgs',
                 color_dir: str = 'color_imgs'):
        self.df = pd.read_excel(mapping)
        if len(self.df.columns) < 4:
            raise ValueError('mapping.xlsx needs ≥4 columns')
        self.root = Path(root)
        self.gray_dir, self.color_dir = gray_dir, color_dir
        self.gray_c, self.param_cols, self.col_c = (
            self.df.columns[0], self.df.columns[1:-1], self.df.columns[-1])
        self.t_gray = transforms.Compose([
            transforms.Resize(size), transforms.CenterCrop(size),
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.t_col = transforms.Compose([
            transforms.Resize(size), transforms.CenterCrop(size),
            transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])

    def __len__(self): return len(self.df)

    def _read(self, fname: str, gray: bool):
        sub = self.gray_dir if gray else self.color_dir
        path = self.root / sub / fname
        if not path.exists():
            raise FileNotFoundError(path)
        img = Image.open(path).convert('L' if gray else 'RGB')
        return self.t_gray(img) if gray else self.t_col(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        g = self._read(row[self.gray_c], True)
        c = self._read(row[self.col_c], False)
        params = pd.to_numeric(row[self.param_cols],
                               errors='coerce').fillna(0).astype('float32')
        return g, c, torch.from_numpy(params.values)
# ---------- 2. Model building blocks ----------
class ParamEncoder(nn.Module):
    def __init__(self, dim: int, hid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hid), nn.ReLU(True),
            nn.Linear(hid, hid), nn.ReLU(True))
    def forward(self, x): return self.net(x)

def down(i, o, norm=True):
    layers = [nn.Conv2d(i, o, 4, 2, 1, bias=False)]
    if norm: layers.append(nn.BatchNorm2d(o))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)

def up(i, o, drop=False):
    layers = [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False),
              nn.BatchNorm2d(o), nn.ReLU(True)]
    if drop: layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, p_dim: int, base: int = 64):
        super().__init__()
        self.pe = ParamEncoder(p_dim, base)
        self.d1 = down(1+base, base, False)
        self.d2 = down(base, base*2)
        self.d3 = down(base*2, base*4)
        self.d4 = down(base*4, base*8)
        self.d5 = down(base*8, base*8)
        self.u1 = up(base*8, base*8, True)
        self.u2 = up(base*16, base*4, True)
        self.u3 = up(base*8, base*2)
        self.u4 = up(base*4, base)
        self.fin = nn.Sequential(
            nn.ConvTranspose2d(base*2, 3, 4, 2, 1), nn.Tanh())

    def forward(self, g, p):
        B, _, H, W = g.shape
        f = self.pe(p)[:, :, None, None].expand(B, -1, H, W)
        x = torch.cat([g, f], 1)
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2)
        d4 = self.d4(d3); d5 = self.d5(d4)
        u1 = self.u1(d5)
        u2 = self.u2(torch.cat([u1, d4], 1))
        u3 = self.u3(torch.cat([u2, d3], 1))
        u4 = self.u4(torch.cat([u3, d2], 1))
        return self.fin(torch.cat([u4, d1], 1))

class Discriminator(nn.Module):
    def __init__(self, p_dim: int, base: int = 64):
        super().__init__()
        self.pe = ParamEncoder(p_dim, base)
        self.net = nn.Sequential(
            down(1+3+base, base, False),
            down(base, base*2),
            down(base*2, base*4),
            nn.Conv2d(base*4, 1, 4, 1, 1))
    def forward(self, g, c, p):
        B, _, H, W = g.shape
        f = self.pe(p)[:, :, None, None].expand(B, -1, H, W)
        return self.net(torch.cat([g, c, f], 1))
# ---------- 3. Utility ----------
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)

def save_grid(img, path, nrow=4):
    grid = vutils.make_grid(img, nrow=nrow, normalize=True, padding=2)
    vutils.save_image(grid, path)

def r2_batch(pred, target):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    ss_res = ((pred - target)**2).sum(1)
    ss_tot = ((target - target.mean(1, keepdim=True))**2).sum(1)
    return 1 - ss_res / (ss_tot + 1e-8)          # (B,)

# ---------- 4. Training ----------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = Pixel2PixelDataset(args.mapping, args.root, args.size,
                            args.gray_dir, args.color_dir)
    if len(ds) == 0:
        raise RuntimeError('Dataset is empty!')
    p_dim = len(ds.param_cols)
    tr_len = max(int(0.8*len(ds)), 1); te_len = len(ds)-tr_len
    tr_ds, te_ds = random_split(ds, [tr_len, te_len],
                                generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr_ds, args.batch, True, num_workers=0)
    te_loader = DataLoader(te_ds if te_len else tr_ds,
                           args.batch, False, num_workers=0)

    G = Generator(p_dim).to(device).apply(init_weights)
    D = Discriminator(p_dim).to(device).apply(init_weights)
    optG = torch.optim.Adam(G.parameters(), args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), args.lr, betas=(0.5, 0.999))
    adv = nn.BCEWithLogitsLoss(); L1 = nn.L1Loss()

    fixed_g, fixed_c, fixed_p = [t.to(device) for t in next(iter(te_loader))]
    for ep in range(1, args.epochs+1):
        G.train(); D.train()
        g_sum = d_sum = r2_sum = 0.0

        pbar = tqdm(tr_loader, desc=f'Epoch {ep}/{args.epochs}')
        for g, c, p in pbar:
            g, c, p = g.to(device), c.to(device), p.to(device)
            # label size from D output
            lbl_shape = D(g, c, p).shape
            real = torch.ones(lbl_shape, device=device)
            fake = torch.zeros_like(real)

            # ----- D -----
            fake_c_det = G(g, p).detach()
            loss_d = 0.5*(adv(D(g, c, p), real) +
                          adv(D(g, fake_c_det, p), fake))
            optD.zero_grad(); loss_d.backward(); optD.step()

            # ----- G -----
            fake_c = G(g, p)
            loss_g = adv(D(g, fake_c, p), real) + \
                     args.lambda_l1 * L1(fake_c, c)
            optG.zero_grad(); loss_g.backward(); optG.step()

            # metrics
            batch_r2 = r2_batch(fake_c.detach(), c).mean().item()
            g_sum += loss_g.item(); d_sum += loss_d.item(); r2_sum += batch_r2
            pbar.set_postfix(G=f'{loss_g.item():.3f}',
                             D=f'{loss_d.item():.3f}',
                             R2=f'{batch_r2:.3f}')

        n = len(tr_loader)
        print(f'[Epoch {ep}]  G={g_sum/n:.4f}  D={d_sum/n:.4f}  R²={r2_sum/n:.4f}')

        if ep % args.save_interval == 0 or ep == args.epochs:
            G.eval()
            with torch.no_grad():
                sample = G(fixed_g, fixed_p).cpu()*0.5+0.5
            out = Path(args.outdir); out.mkdir(exist_ok=True)
            save_grid(sample, out/f'sample_{ep:03d}.png')
            torch.save({'epoch': ep,
                        'G': G.state_dict(),
                        'D': D.state_dict()},
                       out/f'ckpt_{ep:03d}.pth')

# ---------- 5. CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mapping', default='mapping.xlsx')
    p.add_argument('--root', default='.')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lambda_l1', type=float, default=100.0)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--gray_dir', default='gray_imgs')
    p.add_argument('--color_dir', default='color_imgs')
    p.add_argument('--save_interval', type=int, default=50)
    p.add_argument('--outdir', default='results')
    return p.parse_args()

if __name__ == '__main__':
    print('=== script start ===')
    args = parse_args()
    print('ARGS:', args)
    train(args)
