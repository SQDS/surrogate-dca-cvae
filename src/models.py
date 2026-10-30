"""Four matched configurations, compatible with the 20 archived models."""
import hashlib
import json
import numpy as np
import torch
from torch import nn
from .losses import objective, LOSS_FIELDS

VARIANTS = ('M', 'S', 'S-Y', 'MS')
SEEDS = (42, 43, 44, 45, 46)
PATHS = {'M': ['M'], 'S': ['S'], 'S-Y': ['S-Y'], 'MS': ['MS-main', 'MS-aux']}
SELECT_PATH = {'M': 'M', 'S': 'S', 'S-Y': 'S-Y', 'MS': 'MS-aux'}


def load_weights(variant='MS', seed=42, device='cpu', weights_dir=None):
    from pathlib import Path
    from ._io import ROOT, read_csv, sha256, config
    folder = Path(weights_dir) if weights_dir else ROOT / 'weights'
    rows = read_csv(folder / 'manifest.csv')
    matching = [r for r in rows if r['variant'] == variant and int(r['seed']) == int(seed)]
    if len(matching) != 1:
        raise ValueError('Expected exactly one manifest entry for this configuration and seed')
    row = matching[0]
    path = folder / row['filename']
    if path.resolve().parent != folder.resolve():
        raise ValueError('Weight filename must be inside weights_dir')
    if sha256(path) != row['sha256']:
        raise ValueError('Checkpoint SHA256 mismatch: ' + str(path))
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['variant'] != variant or checkpoint['seed'] != int(seed):
        raise ValueError('Checkpoint identity differs from manifest')
    if int(checkpoint['epoch']) != int(row['epoch']) or checkpoint['scaler'] != config()['scaler']:
        raise ValueError('Checkpoint epoch or signal scaler differs from the release contract')
    net = AblationNet(variant, seed)
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    net.to(device).eval().requires_grad_(False)
    return net, checkpoint

class CVAE(nn.Module):
    def __init__(
        self,
        x_dim: int = 100,       # CBv
        c_dim: int = 100,       # ABP (condition)
        hidden: tuple = (150, 150, 500),
        z_dim: int = 30,
        epsilon_scale: float = 1.0,
    ):
        super().__init__()
        self.x_dim, self.c_dim, self.z_dim = x_dim, c_dim, z_dim
        self.epsilon_scale = float(epsilon_scale)

        # Encoder 主干
        self.enc_trunk = nn.Sequential(
            nn.Linear(x_dim + c_dim, hidden[0]), nn.Tanh(),
            nn.Linear(hidden[0], hidden[1]),     nn.Tanh(),
            nn.Linear(hidden[1], hidden[2]),     nn.Tanh(),
        )
        self.fc_mu     = nn.Linear(hidden[2], z_dim)
        self.fc_logvar = nn.Linear(hidden[2], z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + c_dim, hidden[2]), nn.Tanh(),
            nn.Linear(hidden[2], hidden[1]),     nn.Tanh(),
            nn.Linear(hidden[1], hidden[0]),     nn.Tanh(),
            nn.Linear(hidden[0], x_dim),
        )

    def encode(self, x, c):
        h = self.enc_trunk(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + self.epsilon_scale * eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar


class AuxVAE(nn.Module):
    """
    辅助分支 VAE：输入 12 维 FIR 阶跃响应，潜空间 z_dim 必须与主 CVAE 一致
    Manuscript Methods, Architecture and training objective: hidden widths 50, 50, 100.
    """
    def __init__(self, s_dim: int = 12, hidden: tuple = (50, 50, 100), z_dim: int = 30, epsilon_scale: float = 1.0):
        super().__init__()
        self.s_dim, self.z_dim = s_dim, z_dim
        self.epsilon_scale = float(epsilon_scale)

        self.enc_trunk = nn.Sequential(
            nn.Linear(s_dim, hidden[0]),    nn.Tanh(),
            nn.Linear(hidden[0], hidden[1]), nn.Tanh(),
            nn.Linear(hidden[1], hidden[2]), nn.Tanh(),
        )
        self.fc_mu     = nn.Linear(hidden[2], z_dim)
        self.fc_logvar = nn.Linear(hidden[2], z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim,    hidden[2]), nn.Tanh(),
            nn.Linear(hidden[2], hidden[1]), nn.Tanh(),
            nn.Linear(hidden[1], hidden[0]), nn.Tanh(),
            nn.Linear(hidden[0], s_dim),
        )

    def encode(self, s):
        h = self.enc_trunk(s)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + self.epsilon_scale * eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, s):
        mu, logvar = self.encode(s)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def derive_seed(*parts):
    b=json.dumps(parts,ensure_ascii=False,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8],'little')%(2**63-1)


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(key.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


class Encoder(nn.Module):
    def __init__(self,trunk,mu,logvar):
        super().__init__();self.trunk=trunk;self.mu=mu;self.logvar=logvar
    def forward(self,x):
        h=self.trunk(x);return self.mu(h),self.logvar(h)


class AblationNet(nn.Module):
    def __init__(self,variant,seed):
        super().__init__()
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant;self.seed=int(seed)
        # Templates are untrained random modules. Only retained modules register.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derive_seed(seed,'init_main'));m=CVAE()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derive_seed(seed,'init_aux'));a=AuxVAE()
        self.em=Encoder(m.enc_trunk,m.fc_mu,m.fc_logvar) if variant in ['M','MS'] else None
        self.dy=m.decoder if variant in ['M','S-Y','MS'] else None
        self.es=Encoder(a.enc_trunk,a.fc_mu,a.fc_logvar) if variant in ['S','S-Y','MS'] else None
        self.ds=a.decoder if variant in ['S','S-Y','MS'] else None

    def encode_main(self,y,x):
        if self.em is None:raise ValueError('No main encoder in '+self.variant)
        return self.em(torch.cat([y,x],-1))
    def encode_aux(self,s):
        if self.es is None:raise ValueError('No auxiliary encoder in '+self.variant)
        return self.es(s)
    def decode_y(self,z,x):
        if self.dy is None:raise ValueError('No CBFV decoder in '+self.variant)
        return self.dy(torch.cat([z,x],-1))
    def initial_hashes(self):
        return {k:state_hash(v.state_dict()) for k,v in [('em',self.em),('dy',self.dy),('es',self.es),('ds',self.ds)] if v is not None}

    def objective(self, y, x, s, beta, epsilon_main=None, epsilon_aux=None):
        return objective(self, y, x, s, beta, epsilon_main, epsilon_aux)


def epoch_order(n,seed,epoch):
    return np.random.default_rng(derive_seed(seed,epoch,'shuffle')).permutation(n)


def epoch_noise(n,seed,epoch,branch,device):
    generator=torch.Generator(device=device).manual_seed(derive_seed(seed,epoch,'epsilon_'+branch))
    return torch.randn((n,30),dtype=torch.float32,device=device,generator=generator)


@torch.inference_mode()
def encode_arrays(net,y,x,s,branch,device='cpu',batch=4096):
    parts=[];variances=[]
    for first in range(0,len(x),batch):
        sl=slice(first,first+batch)
        if branch=='main':
            mu,lv=net.encode_main(torch.as_tensor(np.array(y[sl]),device=device),torch.as_tensor(np.array(x[sl]),device=device))
        else:mu,lv=net.encode_aux(torch.as_tensor(np.array(s[sl]),device=device))
        parts.append(mu.cpu().numpy());variances.append(lv.cpu().numpy())
    return np.concatenate(parts),np.concatenate(variances)
