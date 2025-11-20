import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# -----------------------
# RBF + MolConv (with attention dropout + residual)
# -----------------------
class RBFExpansion(nn.Module):
    def __init__(self, num_kernels=16, rbf_gamma=10.0, cutoff=5.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_kernels))
        self.gamma = float(rbf_gamma)
    def forward(self, d):
        diff = d.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff ** 2).clamp(min=1e-10, max=1.0)

class MolConv(nn.Module):
    def __init__(self, in_dim, out_dim, k, remove_xyz=False, rbf_kernels=16, attn_drop=0.1):
        super().__init__()
        self.k = int(k)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.remove_xyz = bool(remove_xyz)
        self.rbf = RBFExpansion(num_kernels=rbf_kernels)
        # self.attn_drop = nn.Dropout(p=attn_drop) if attn_drop and attn_drop > 0 else nn.Identity()
        self.attn_drop = nn.Identity()
        eff_in = in_dim if not remove_xyz else in_dim - 3
        att_in_channels = eff_in * 2 + self.rbf.centers.size(0)

        self.att_mlp = nn.Sequential(
            nn.Conv2d(att_in_channels, 64, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=1, bias=False)
        )
        upd_in = in_dim if not remove_xyz else in_dim - 3
        self.update_ff = nn.Sequential(
            nn.Conv2d(upd_in, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.02)
        )
        # projection for residual if channel dims mismatch
        self.res_proj = nn.Conv2d(upd_in, out_dim, kernel_size=1, bias=False) if upd_in != out_dim else nn.Identity()
        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x, idx_base):
        # x: (B,C,N)
        dist, _, feat_c, feat_n = self._feat(x, idx_base, self.k, self.remove_xyz)  # feat_*: (B,C',N,k)
        rbf_feat = self.rbf(dist.squeeze(1))                 # (B,N,k,R)
        feat_c_p = feat_c.permute(0,2,3,1)                   # (B,N,k,C')
        feat_n_p = feat_n.permute(0,2,3,1)                   # (B,N,k,C')
        att_in = torch.cat([feat_c_p, feat_n_p, rbf_feat], dim=-1).permute(0,3,1,2).contiguous()
        att_logits = self.att_mlp(att_in)                    # (B,1,N,k)
        att_logits = self.attn_drop(att_logits)
        att = F.softmax(att_logits, dim=-1)

        # weighted neighbor features
        neigh = att * feat_n                                  # (B,C',N,k)
        neigh_upd = self.update_ff(neigh)                     # (B,Co,N,k)

        # simple residual across neighbors (project if needed)
        res = self.res_proj(feat_n)                           # (B,Co,N,k)
        feat = neigh_upd + 0.1 * res                          # mild residual blending

        feat = feat.mean(dim=-1)                              # (B,Co,N)
        return feat

    def _feat(self, x, idx_base, k, remove_xyz):
        B, C, N = x.size()
        inner = -2 * torch.matmul(x.transpose(2,1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pair = -xx - inner - xx.transpose(2,1)               # -||xi-xj||^2
        dist, idx = pair.topk(k=k, dim=2)                    # largest -> nearest
        dist = (-dist).clamp(min=1e-12).sqrt()               # Euclidean

        idx = (idx + idx_base).view(-1)
        xt = x.transpose(2,1).contiguous()
        neigh = xt.view(B*N, -1)[idx, :].view(B, N, k, C)
        cent  = xt.view(B, N, 1, C).repeat(1,1,k,1)

        if remove_xyz:
            return (dist.unsqueeze(1),
                    None,
                    cent[:,:,:,3:].permute(0,3,1,2),
                    neigh[:,:,:,3:].permute(0,3,1,2))
        else:
            return (dist.unsqueeze(1), None,
                    cent.permute(0,3,1,2),
                    neigh.permute(0,3,1,2))

# -----------------------
# Encoder (unchanged core, uses MolConv above)
# -----------------------
class Encoder(nn.Module):
    def __init__(self, in_dim, layers, emb_dim, k):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_layers = nn.ModuleList([MolConv(in_dim, layers[0], k, remove_xyz=True,  attn_drop=0.1)])
        for i in range(1, len(layers)):
            self.hidden_layers.append(MolConv(layers[i-1], layers[i], k, remove_xyz=False, attn_drop=0.1))
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, 1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(0.2)
        )
        self.merge = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(0.2)
        )
        self._reset()

    def _reset(self):
        for m in self.merge:
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x, idx_base):
        xs = []
        for i, layer in enumerate(self.hidden_layers):
            xs.append(layer(x if i==0 else xs[-1], idx_base))
        x = torch.cat(xs, dim=1)             # (B,emb_dim,N)
        x = self.conv(x)
        p1 = F.adaptive_max_pool1d(x, 1).squeeze().view(-1, self.emb_dim)
        p2 = F.adaptive_avg_pool1d(x, 1).squeeze().view(-1, self.emb_dim)
        x = self.merge(torch.cat([p1, p2], dim=1))
        return x

# -----------------------
# Heads, env, fusion (you already had these)
# -----------------------
class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim, bias=False),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
    def forward(self, x): return self.net(x)

class EnvTower(nn.Module):
    def __init__(self, species_dim=0, ion_dim=0, cont_dim=0,
                 branch_hidden=128, out_dim=256, dropout=0.1):
        super().__init__()
        self.species_dim = int(species_dim or 0)
        self.ion_dim     = int(ion_dim or 0)
        self.cont_dim    = int(cont_dim or 0)

        self.species = _MLP(self.species_dim, branch_hidden, out_dim//3, p=dropout) if self.species_dim>0 else None
        self.ions    = _MLP(self.ion_dim,     branch_hidden, out_dim//3, p=dropout) if self.ion_dim>0 else None
        self.cont    = _MLP(self.cont_dim,    branch_hidden, out_dim//3, p=dropout) if self.cont_dim>0 else None
        self.proj = None

    def set_output_dim(self, emb_dim: int):
        self.proj = _MLP(self.output_dim_raw, emb_dim, emb_dim, p=0.1)

    @property
    def output_dim_raw(self) -> int:
        parts = 0
        if self.species is not None: parts += self.species.net[-2].normalized_shape[0]
        if self.ions    is not None: parts += self.ions.net[-2].normalized_shape[0]
        if self.cont    is not None: parts += self.cont.net[-2].normalized_shape[0]
        return parts if parts>0 else 0

    def forward(self, env_full: torch.Tensor) -> torch.Tensor:
        B, D = env_full.shape
        offset = 0; reps = []
        if self.species is not None:
            reps.append(self.species(env_full[:, offset:offset+self.species_dim])); offset += self.species_dim
        if self.ions is not None:
            reps.append(self.ions(env_full[:, offset:offset+self.ion_dim]));       offset += self.ion_dim
        if self.cont is not None:
            reps.append(self.cont(env_full[:, offset:offset+self.cont_dim]));      offset += self.cont_dim
        if not reps: reps = [env_full]
        z_env_cat = torch.cat(reps, dim=1) if len(reps)>1 else reps[0]
        if self.proj is None:
            raise RuntimeError("Call env_tower.set_output_dim(emb_dim) during model init.")
        return self.proj(z_env_cat)

class GatedFusion(nn.Module):
    def __init__(self, emb_dim, hidden=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(emb_dim*2, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, emb_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
    def forward(self, z_mol, z_env) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([z_mol, z_env], dim=1)))  # (B,D)
        return g * z_mol + (1.0 - g) * z_env

# -----------------------
# Decoder (unchanged)
# -----------------------
class FCResBlock(nn.Module): 
    def __init__(self, in_dim: int, out_dim: int, dropout: float=0.) -> torch.Tensor: 
        super(FCResBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(in_dim, out_dim, bias=False) 
        self.bn1 = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.bn2 = nn.LayerNorm(out_dim)
        self.linear3 = nn.Linear(out_dim, out_dim, bias=False)
        self.bn3 = nn.LayerNorm(out_dim)
        self.dp = nn.Dropout(dropout)
        self._reset_parameters()
    def _reset_parameters(self): 
        for m in self.modules(): 
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)): 
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    def forward(self, x):
        identity = x
        x = self.bn1(self.linear1(x)); x = F.leaky_relu(x, 0.2)
        x = self.bn2(self.linear2(x)); x = F.leaky_relu(x, 0.2)
        x = self.bn3(self.linear3(x))
        x = x + F.interpolate(identity.unsqueeze(1), size=x.size()[1]).squeeze()
        x = F.leaky_relu(x, 0.2)
        x = self.dp(x)
        return x

class MSDecoder(nn.Module): 
    def __init__(self, in_dim, layers, out_dim, dropout): 
        super(MSDecoder, self).__init__()
        self.blocks = nn.ModuleList([FCResBlock(in_dim=in_dim, out_dim=layers[0])])
        for i in range(len(layers)-1): 
            if len(layers) - i > 3:
                self.blocks.append(FCResBlock(in_dim=layers[i], out_dim=layers[i+1]))
            else:
                self.blocks.append(FCResBlock(in_dim=layers[i], out_dim=layers[i+1], dropout=dropout))
        self.fc = nn.Linear(layers[-1], out_dim)
        self._reset_parameters()
    def _reset_parameters(self): 
        nn.init.kaiming_normal_(self.fc.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.fc(x)

class RegAnchoredOrdinalHead(nn.Module):
    def __init__(self, num_classes: int = 5, init_edges=None, learn_theta=True):
        super().__init__()
        self.num_classes = num_classes
        if init_edges is None:
            init_edges = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        assert len(init_edges) == num_classes - 1
        self.theta = nn.Parameter(init_edges.clone(), requires_grad=learn_theta)
        
        # Simple affine transformation (NO detach, but lightweight)
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.alpha_log = nn.Parameter(torch.tensor(0.0))
        
    def regularization(self, margin: float = 0.1, lambda_theta: float = 5e-3) -> torch.Tensor:
        diffs = self.theta[1:] - self.theta[:-1] - margin
        mono_violation = F.relu(-diffs).sum()
        # Regularize scale to stay near 1
        scale_pen = (self.a - 1.0).pow(2)
        return lambda_theta * mono_violation + 1e-4 * scale_pen
    
    def forward(self, reg_scalar: torch.Tensor) -> torch.Tensor:
        # NO detach - allow gradient flow
        s = reg_scalar.view(-1)
        s = self.a * s + self.b
        alpha = F.softplus(self.alpha_log) + 0.5  # Ensure reasonable scaling
        logits = alpha * (s.unsqueeze(1) - self.theta.view(1, -1))
        return logits
# -----------------------
# Reg-anchored ordinal head (as you use)
# -----------------------
# class RegAnchoredOrdinalHead(nn.Module):
#     def __init__(self, num_classes: int = 5, init_edges=None, learn_theta=True):
#         super().__init__()
#         self.num_classes = num_classes
#         if init_edges is None:
#             init_edges = torch.tensor([-1.0, 0.0, 1.0, 2.0])
#         assert len(init_edges) == num_classes - 1
#         self.theta = nn.Parameter(init_edges.clone(), requires_grad=learn_theta)
#         self.a = nn.Parameter(torch.tensor(1.0))
#         self.b = nn.Parameter(torch.tensor(0.0))
#         self.alpha_log = nn.Parameter(torch.zeros(1))
#     def regularization(self, margin: float = 1e-3, lambda_theta: float = 1e-3, lambda_scale: float = 1e-6) -> torch.Tensor:
#         diffs = self.theta[1:] - self.theta[:-1] - margin
#         mono_violation = F.relu(-diffs).sum()
#         scale_pen = (self.a - 1.0).pow(2)
#         return lambda_theta * mono_violation + lambda_scale * scale_pen
#     def forward(self, reg_scalar: torch.Tensor) -> torch.Tensor:
#         s = reg_scalar.view(-1)
#         s = self.a * s + self.b
#         alpha = F.softplus(self.alpha_log) + 1e-4
#         logits = alpha * (s.unsqueeze(1) - self.theta.view(1, -1))
#         return logits

# -----------------------
# MolNet_MS (single conformer, env fusion, coord norm, stronger heads)
# -----------------------
class MolNet_MS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.add_num   = int(config['add_num'])
        self.emb_dim   = int(config['emb_dim'])
        self.num_classes = int(config.get('num_classes', 5))
        self.single_conf_strategy = config.get('single_conf_strategy', 'first')  # 'first'|'largest_mask'

        # ===== encoder =====
        self.encoder = Encoder(
            in_dim=int(config['in_dim']),
            layers=config['encode_layers'],
            emb_dim=self.emb_dim,
            k=int(config['k'])
        )

        # ===== env handling =====
        # simple projection of raw env to env_emb_dim (keeps it cheap)
        env_emb_dim = int(config.get('env_emb_dim', 256)) if self.add_num > 0 else 0
        self.env_proj = None
        if env_emb_dim > 0:
            self.env_proj = nn.Sequential(
                nn.Linear(self.add_num if self.add_num > 1 else 1, env_emb_dim, bias=False),
                nn.LayerNorm(env_emb_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.get('dropout', 0.1)),
                nn.Linear(env_emb_dim, env_emb_dim, bias=False),
                nn.LayerNorm(env_emb_dim),
                nn.LeakyReLU(0.2),
            )

        # ===== shared trunk on [z_mol | z_env] =====
        trunk_in = self.emb_dim + (env_emb_dim if self.env_proj is not None else (self.add_num if self.add_num>1 else (1 if self.add_num==1 else 0)))
        trunk_hidden = int(config.get('trunk_hidden', 512))
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, trunk_hidden, bias=False),
            nn.LayerNorm(trunk_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.get('dropout', 0.3)),
            nn.Linear(trunk_hidden, self.emb_dim, bias=False),
            nn.LayerNorm(self.emb_dim),
            nn.LeakyReLU(0.2),
        )

        # ===== heads =====
        # classification head (5-way)
        self.cls_head = nn.Sequential(
            nn.Linear(self.emb_dim, 128, bias=False),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.get('dropout', 0.3)),
            nn.Linear(128, self.num_classes)
        )

        # regression decoder (make shallower via config)
        self.decoder = MSDecoder(
            in_dim=self.emb_dim,
            layers=config.get('decode_layers', [1024, 256, 64]),   # <-- shallower default
            out_dim=1,
            dropout=config.get('dropout', 0.3)
        )

        # ordinal head (CORAL anchored on reg)
        self.ordinal_head = RegAnchoredOrdinalHead(
            num_classes=self.num_classes,
            init_edges=torch.tensor([-1.0, 0.0, 1.0, 2.0]),
            learn_theta=True
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    # ---------- helpers ----------
    def _prep_env_vec(self, env: torch.Tensor) -> torch.Tensor:
        if self.add_num <= 0:
            return None
        if self.add_num == 1:
            return env.unsqueeze(1)   # (B,) -> (B,1)
        return env                    # (B,D)

    def _normalize_coords(self, x_bn_f: torch.Tensor, mask_bn: torch.Tensor) -> torch.Tensor:
        # center/scale xyz by masked RMS; keep other channels
        xyz = x_bn_f[..., :3]
        m = mask_bn.unsqueeze(-1).float()
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_xyz = (xyz * m).sum(dim=1, keepdim=True) / denom
        xyz = (xyz - mean_xyz) * m
        rms = torch.sqrt(((xyz**2) * m).sum(dim=(1,2), keepdim=True) / denom).clamp_min(1e-6)
        xyz = xyz / rms
        return torch.cat([xyz, x_bn_f[..., 3:]], dim=-1)

    def _encode_one_conformer(self, x_bn_f: torch.Tensor, mask_bn: torch.Tensor) -> torch.Tensor:
        x_bn_f = self._normalize_coords(x_bn_f, mask_bn)
        B, N, F = x_bn_f.shape
        x_bf_n = x_bn_f.permute(0, 2, 1).contiguous()         # [B, F, N]
        idx_base = torch.arange(0, B, device=x_bn_f.device).view(-1, 1, 1) * N
        z = self.encoder(x_bf_n, idx_base)                    # [B, emb_dim]
        return z

    # ---------- forward ----------
    def forward(self, x_bm_nf: torch.Tensor, mask_bm_n: torch.Tensor, env_bd: torch.Tensor):
        """
        x_bm_nf: [B,M,N,F]  (we pick one conformer)
        mask_bm_n: [B,M,N]
        env_bd: [B, D_env]
        """
        B, M, N, F = x_bm_nf.shape

        # pick conformer
        if M == 1:
            x_sel   = x_bm_nf[:, 0, :, :]
            mask_sel= mask_bm_n[:, 0, :]
        else:
            if self.single_conf_strategy == 'largest_mask':
                lengths = mask_bm_n.float().sum(dim=2)              # [B,M]
                m_idx = lengths.argmax(dim=1)                       # [B]
            else:
                m_idx = torch.zeros(B, dtype=torch.long, device=x_bm_nf.device)
            ar = torch.arange(B, device=x_bm_nf.device)
            x_sel    = x_bm_nf[ar, m_idx, :, :]
            mask_sel = mask_bm_n[ar, m_idx, :]

        # zero padded atoms
        x_sel = x_sel * mask_sel.unsqueeze(-1).float()

        # mol embedding
        z_mol = self._encode_one_conformer(x_sel, mask_sel)         # [B, emb_dim]

        # env embedding (project then concat)
        env_vec = self._prep_env_vec(env_bd)                        # [B,D_env] or None
        if env_vec is not None and self.env_proj is not None:
            z_env = self.env_proj(env_vec)                          # [B, env_emb_dim]
        elif env_vec is not None:
            z_env = env_vec                                         # raw concat if projector disabled
        else:
            z_env = None

        z_cat = torch.cat([z_mol, z_env], dim=1) if z_env is not None else z_mol  # [B, emb_dim + env_emb]
        z = self.trunk(z_cat)                                       # [B, emb_dim]

        # heads
        cls_logits  = self.cls_head(z)                              # (B,K)
        reg_output  = self.decoder(z)                               # (B,1)
        coral_logits = self.ordinal_head(reg_output)                # (B,K-1)  (no detach!)

        return reg_output, cls_logits, coral_logits
