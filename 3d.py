# train_3d_pix2pix.py  (Part 1/4)
# ===============================
# 3-D pix2pix GAN (U-Net + PatchGAN) conditioned on Da & Pe
#
# Supports grouping by rows (group_size) or by prefix
#
# Folder layout:
#   mapping.xlsx : input image | Da | Pe
#   gray_imgs/   : prefix_idx.png ...
#   color_imgs/  : prefix_idx.png ...

from pathlib import Path
from typing import Union
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import utils as vutils
from torchvision.transforms import functional as TF

# train_3d_pix2pix.py  (Part 2/4)
# ---------- Dataset ----------

class VolumeDataset(Dataset):
    def __init__(self,
                 mapping: Union[str, Path] = 'mapping.xlsx',
                 root: Union[str, Path] = '.',
                 size: int = 128,
                 gray_dir: str = 'gray_imgs',
                 color_dir: str = 'color_imgs',
                 group_size: int = None):
        # Read mapping.xlsx, at least 3 columns: filename, Da, Pe
        self.df = pd.read_excel(mapping)
        if len(self.df.columns) < 3:
            raise ValueError('mapping.xlsx must have ≥3 columns: input image | Da | Pe')
        self.root = Path(root)
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.size = size
        self.group_size = group_size

        # Precompute group indices
        if self.group_size is not None:
            n = len(self.df)
            self.groups = [
                list(range(i, min(i + self.group_size, n)))
                for i in range(0, n, self.group_size)
            ]
        else:
            # Group by prefix: remove the last -<slice> with rsplit('-',1)
            self.groups = {}
            col0 = self.df.columns[0]
            for idx, fname in enumerate(self.df[col0].astype(str)):
                stem = Path(fname).stem               # e.g. "0.3-1-10-12"
                prefix = stem.rsplit('-', 1)[0]       # e.g. "0.3-1-10"
                self.groups.setdefault(prefix, []).append(idx)
            self.groups = list(self.groups.values())

    def __len__(self):
        return len(self.groups)

    def _load_slice(self, prefix: str, idx: int, gray: bool):
        fname = f"{prefix}-{idx}.png"  # Note: also changed to hyphen here
        path = self.root / (self.gray_dir if gray else self.color_dir) / fname
        img = Image.open(path).convert('L' if gray else 'RGB')
        img = img.resize((self.size, self.size), Image.BICUBIC)
        t = TF.to_tensor(img) * 2 - 1
        return t

    def __getitem__(self, i):
        idxs = self.groups[i]
        col0 = self.df.columns[0]
        fnames  = self.df.iloc[idxs, 0].astype(str).tolist()
        Da_vals = self.df.iloc[idxs, 1].astype(float).tolist()
        Pe_vals = self.df.iloc[idxs, 2].astype(float).tolist()
        Da, Pe  = Da_vals[0], Pe_vals[0]

        g_slices, c_slices = [], []
        for fname in fnames:
            stem = Path(fname).stem                 # "0.3-1-10-12"
            prefix, slice_str = stem.rsplit('-', 1) # ["0.3-1-10", "12"]
            slice_idx = int(slice_str)
            g_slices.append(self._load_slice(prefix, slice_idx, True))
            c_slices.append(self._load_slice(prefix, slice_idx, False))

        g_vol = torch.stack(g_slices, dim=1)
        c_vol = torch.stack(c_slices, dim=1)
        params = torch.tensor([Da, Pe], dtype=torch.float32)
        return g_vol, c_vol, params

# train_3d_pix2pix.py  (Part 3/4)
# ---------- Model Definitions ----------

class ParamEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, dim), nn.ReLU(True),
            nn.Linear(dim, dim), nn.ReLU(True)
        )
    def forward(self, p):
        return self.net(p)  # (B, dim)

def dblock(i, o, norm=True):
    layers = [nn.Conv3d(i, o, 4, 2, 1, bias=False)]
    if norm: layers.append(nn.BatchNorm3d(o))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)

def ublock(i, o, drop=False):
    layers = [
        nn.ConvTranspose3d(i, o, 4, 2, 1, bias=False),
        nn.BatchNorm3d(o), nn.ReLU(True)
    ]
    if drop: layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class Generator3D(nn.Module):
    def __init__(self, embed=64, base=64):
        super().__init__()
        self.pe = ParamEncoder(embed)
        self.d1 = dblock(1+embed, base, False)
        self.d2 = dblock(base,   base*2)
        self.d3 = dblock(base*2, base*4)
        self.d4 = dblock(base*4, base*8)
        self.d5 = dblock(base*8, base*8)
        self.u1 = ublock(base*8,  base*8, True)
        self.u2 = ublock(base*16, base*4, True)
        self.u3 = ublock(base*8,  base*2)
        self.u4 = ublock(base*4,  base)
        self.out = nn.Sequential(
            nn.ConvTranspose3d(base*2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, p):
        B, _, D, H, W = x.shape
        pe = self.pe(p).view(B, -1, 1, 1, 1).expand(-1, -1, D, H, W)
        x = torch.cat([x, pe], 1)
        d1 = self.d1(x); d2 = self.d2(d1)
        d3 = self.d3(d2); d4 = self.d4(d3); d5 = self.d5(d4)
        u1 = self.u1(d5)
        u2 = self.u2(torch.cat([u1, d4], 1))
        u3 = self.u3(torch.cat([u2, d3], 1))
        u4 = self.u4(torch.cat([u3, d2], 1))
        return self.out(torch.cat([u4, d1], 1))

class Discriminator3D(nn.Module):
    def __init__(self, embed=64, base=64):
        super().__init__()
        self.pe = ParamEncoder(embed)
        self.net = nn.Sequential(
            dblock(1+3+embed, base, False),
            dblock(base,   base*2),
            dblock(base*2, base*4),
            nn.Conv3d(base*4, 1, 4, 1, 1)
        )

    def forward(self, g, c, p):
        B, _, D, H, W = g.shape
        pe = self.pe(p).view(B, -1, 1, 1, 1).expand(-1, -1, D, H, W)
        return self.net(torch.cat([g, c, pe], 1))

# Utils
def init_weights(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)

def save_mid_slice(vol, path):
    with torch.no_grad():
        img = vol[0][:, vol.shape[2]//2]  # (3, H, W)
        vutils.save_image(img*0.5+0.5, path)

# train_3d_pix2pix.py  (Part 4/4)
# ---------- Training & CLI ----------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = VolumeDataset(
        args.mapping, args.root, args.size,
        args.gray_dir, args.color_dir, args.group_size
    )
    tr_len = max(int(0.8 * len(ds)), 1)
    va_len = len(ds) - tr_len
    tr_ds, va_ds = random_split(
        ds, [tr_len, va_len],
        generator=torch.Generator().manual_seed(42)
    )
    tr_loader = DataLoader(tr_ds, args.batch, True)
    va_loader = DataLoader(va_ds, 1, False)

    G = Generator3D(args.embed_dim).to(device).apply(init_weights)
    D = Discriminator3D(args.embed_dim).to(device).apply(init_weights)
    optG = torch.optim.Adam(G.parameters(), args.lr, betas=(0.5,0.999))
    optD = torch.optim.Adam(D.parameters(), args.lr, betas=(0.5,0.999))
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()

    fixed_g, _, fixed_p = next(iter(va_loader))
    fixed_g, fixed_p = fixed_g.to(device), fixed_p.to(device)

    for ep in range(1, args.epochs+1):
        g_tot = d_tot = 0.0
        for g, c, p in tqdm(tr_loader, desc=f'Epoch {ep}/{args.epochs}'):
            g, c, p = g.to(device), c.to(device), p.to(device)
            # Discriminator update
            with torch.no_grad():
                fake = G(g, p).detach()
            loss_D = 0.5 * (
                bce(D(g, c, p), torch.ones_like(D(g, c, p))) +
                bce(D(g, fake, p), torch.zeros_like(D(g, fake, p)))
            )
            optD.zero_grad(); loss_D.backward(); optD.step()
            # Generator update
            fake = G(g, p)
            loss_G = bce(D(g, fake, p), torch.ones_like(D(g, fake, p))) + \
                     args.lambda_l1 * l1(fake, c)
            optG.zero_grad(); loss_G.backward(); optG.step()

            g_tot += loss_G.item()
            d_tot += loss_D.item()

        print(f"Epoch {ep}: G={g_tot/len(tr_loader):.4f} | D={d_tot/len(tr_loader):.4f}")

        if ep % args.save_interval == 0 or ep == args.epochs:
            with torch.no_grad():
                sample = G(fixed_g, fixed_p).cpu()
            Path(args.outdir).mkdir(exist_ok=True)
            save_mid_slice(sample, Path(args.outdir) / f'sample_{ep:03d}.png')


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--mapping',     default='mapping.xlsx')
    ap.add_argument('--root',        default='.')
    ap.add_argument('--group_size',  type=int, default=None,
                    help='Number of rows to group into a 3D sample; None=group by prefix')
    ap.add_argument('--epochs',      type=int, default=100)
    ap.add_argument('--batch',       type=int, default=2)
    ap.add_argument('--size',        type=int, default=128)
    ap.add_argument('--lr',          type=float, default=2e-4)
    ap.add_argument('--lambda_l1',   type=float, default=100.0)
    ap.add_argument('--embed_dim',   type=int, default=64)
    ap.add_argument('--gray_dir',    default='gray_imgs')
    ap.add_argument('--color_dir',   default='color_imgs')
    ap.add_argument('--save_interval', type=int, default=10)
    ap.add_argument('--outdir',      default='results')
    return ap.parse_args()


if __name__ == '__main__':
    import sys
    # VS Code “Run” (no args) → quick demo settings
    if len(sys.argv) == 1:
        from argparse import Namespace
        args = Namespace(
            mapping='mapping.xlsx',
            root='.',
            group_size=5,     # or 8, 4, ... try what you want
            epochs=2,         # demo runs 2 epochs
            batch=1,
            size=128,
            lr=2e-4,
            lambda_l1=100.0,
            embed_dim=64,
            gray_dir='gray_imgs',
            color_dir='color_imgs',
            save_interval=1,
            outdir='results'
        )
        print('No CLI args → using demo settings:', args)
    else:
        args = parse_args()
    print('Running with:', args)
    train(args)
