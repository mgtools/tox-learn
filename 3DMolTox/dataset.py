import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import Literal, List, Tuple

ConformerMode = Literal['flatten', 'pool', 'rand', 'single']

class MolMS_Dataset(Dataset):
    """
    NEW format item:
        { 'title': str, 'mol_confs': float32 [M,N,F], 'atom_mask': uint8 [M,N],
          'features': float32 [D], 'Effect value': float }
    LEGACY format item:
        { 'title': str, 'mol': float32 [N,F], 'features': float32 [D], 'Effect value': float }

    conformer_mode:
      - 'flatten' : each conformer is its own dataset sample (M -> 1 for each)
      - 'single'  : one deterministic conformer per compound (eval/test)
      - 'pool'    : all conformers together (original)
      - 'rand'    : one random conformer per __getitem__
    """

    def __init__(
        self,
        path: str,
        data_augmentation: bool = True,
        flip_axis: int = 0,
        conformer_mode: ConformerMode = 'flatten',
        single_strategy: str = 'first',   # 'first' or 'largest_mask'
    ):
        super().__init__()
        assert conformer_mode in ('flatten', 'pool', 'rand', 'single')
        assert single_strategy in ('first', 'largest_mask')
        self.conformer_mode: ConformerMode = conformer_mode
        self.single_strategy = single_strategy
        self.flip_axis = int(flip_axis)  # 0=x, 1=y, 2=z
        self.data_augmentation = bool(data_augmentation)

        with open(path, "rb") as f:
            raw = pickle.load(f)

        if len(raw) == 0:
            self.compounds = []
            self.items = []
            print(f"Load 0 data from {path}")
            return

        def to_new_fmt(d):
            if ('mol_confs' in d) and ('atom_mask' in d):
                x = np.asarray(d['mol_confs'], dtype=np.float32)  # [M,N,F]
                m = np.asarray(d['atom_mask'], dtype=np.uint8)    # [M,N]
            else:
                mol = np.asarray(d['mol'], dtype=np.float32)      # [N,F]
                m1d = (np.abs(mol[:, :3]).sum(axis=1) > 0).astype(np.uint8)  # [N]
                x = mol[None, ...]                                 # [1,N,F]
                m = m1d[None, ...]                                 # [1,N]
            e = np.asarray(d['features'], dtype=np.float32)        # [D]
            y = float(d['Effect value'])
            return {
                'title': str(d['title']),
                'mol_confs': x,
                'atom_mask': m,
                'features': e,
                'Effect value': y,
            }

        self.compounds: List[dict] = [to_new_fmt(d) for d in raw]

        # Optional flip augmentation (makes extra compounds; safe for train, keep off for valid/test)
        if self.data_augmentation:
            aug = []
            ax = self.flip_axis
            for d in self.compounds:
                x = np.copy(d['mol_confs'])
                if x.shape[-1] >= 3:
                    x[..., ax] *= -1.0
                aug.append({
                    'title': d['title'] + '_f',
                    'mol_confs': x,
                    'atom_mask': np.copy(d['atom_mask']),
                    'features': np.copy(d['features']),
                    'Effect value': d['Effect value'],
                })
            self.compounds = self.compounds + aug
            print(f"Load {len(self.compounds)} compounds from {path} (flip augmentation on axis {ax})")
        else:
            print(f"Load {len(self.compounds)} compounds from {path}")

        # Build indices
        if self.conformer_mode == 'pool':
            self.items: List[Tuple[int, int]] = [(i, -1) for i in range(len(self.compounds))]
        elif self.conformer_mode == 'flatten':
            idx = []
            for i, d in enumerate(self.compounds):
                M = int(d['mol_confs'].shape[0])
                for m in range(M):
                    idx.append((i, m))
            self.items = idx
        elif self.conformer_mode == 'rand':
            self.items = [(i, -2) for i in range(len(self.compounds))]  # -2: draw random in __getitem__
        else:  # 'single'
            # Pre-compute a deterministic conformer index per compound
            picks = []
            for i, d in enumerate(self.compounds):
                M = int(d['mol_confs'].shape[0])
                if M == 0:
                    picks.append((i, 0))
                    continue
                if self.single_strategy == 'first':
                    m = 0
                else:  # 'largest_mask'
                    m = int(np.argmax(d['atom_mask'].sum(axis=1)))
                picks.append((i, m))
            self.items = picks

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        comp_idx, conf_idx = self.items[idx]
        d = self.compounds[comp_idx]
        title_base = d['title']
        X = d['mol_confs']          # [M,N,F]
        mask_all = d['atom_mask']   # [M,N]
        env = torch.from_numpy(d['features']).float()
        y = torch.tensor(d['Effect value'], dtype=torch.float32)

        M, N, F = X.shape

        if self.conformer_mode == 'pool':
            x_confs = torch.from_numpy(X).float()                # (M,N,F)
            mask    = torch.from_numpy(mask_all).bool()          # (M,N)
            title   = title_base
        elif self.conformer_mode == 'flatten':
            m = int(conf_idx)
            x_confs = torch.from_numpy(X[m:m+1]).float()         # (1,N,F)
            mask    = torch.from_numpy(mask_all[m:m+1]).bool()   # (1,N)
            title   = f"{title_base}::conf{m}"
        elif self.conformer_mode == 'rand':
            m = np.random.randint(M) if M > 0 else 0
            x_confs = torch.from_numpy(X[m:m+1]).float()
            mask    = torch.from_numpy(mask_all[m:m+1]).bool()
            title   = f"{title_base}::conf{m}"
        else:  # 'single' (deterministic)
            m = int(conf_idx)
            x_confs = torch.from_numpy(X[m:m+1]).float()         # (1,N,F)
            mask    = torch.from_numpy(mask_all[m:m+1]).bool()   # (1,N)
            title   = f"{title_base}::single_conf{m}"

        return title, x_confs, mask, env, y

# class MolMS_Dataset(Dataset):
#     def __init__(self, path, data_augmentation=True): 
#         with open(path, 'rb') as file: 
#             data = pickle.load(file)

#         # data augmentation by flipping the x,y,z-coordinates
#         if data_augmentation: 
#             flipping_data = []
#             for d in data:
#                 flipping_mol_arr = np.copy(d['mol'])
#                 flipping_mol_arr[:, 0] *= -1
#                 flipping_data.append({'title': d['title']+'_f', 'mol': flipping_mol_arr, 'features': d['features'], 'Effect value': d['Effect value']})
            
#             self.data = data + flipping_data
#             print('Load {} data from {} (with data augmentation by flipping coordinates)'.format(len(self.data), path))
#         else:
#             self.data = data
#             print('Load {} data from {}'.format(len(self.data), path))

#     def __len__(self): 
#         return len(self.data)

#     def __getitem__(self, idx): 
#         title = self.data[idx]['title']
#         mol = torch.from_numpy(self.data[idx]['mol']).float()  # convert to tensor
#         features = np.asarray(self.data[idx]['features'], dtype=np.float32)
#         features = torch.from_numpy(features)

#         effect_value = torch.tensor(self.data[idx]['Effect value']).float()  # convert to tensor
#         return title, mol, features, effect_value

class Mol_Dataset(Dataset):
    def __init__(self, data): 
        self.data = data
        print('Load {} data'.format(len(self.data)))

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['features'], self.data[idx]['Effect value']

class MolPRE_Dataset(Dataset): 
    def __init__(self, path): 
        with open(path, 'rb') as file: 
            self.data = pickle.load(file)
        print('Load {} data from {}'.format(len(self.data), path))

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['features'], self.data[idx]['Effect value']

class MolRT_Dataset(Dataset): 
    def __init__(self, path, data_augmentation=True): 
        with open(path, 'rb') as file: 
            data = pickle.load(file)

        # data augmentation by flipping the x,y,z-coordinates
        if data_augmentation: 
            flipping_data = []
            for d in data:
                flipping_mol_arr = np.copy(d['mol'])
                flipping_mol_arr[:, 0] *= -1
                flipping_data.append({'title': d['title']+'_f', 'mol': flipping_mol_arr, 'features': d['features'], 'Effect value': d['Effect value']})
            
            self.data = data + flipping_data
            print('Load {} data from {} (with data augmentation by flipping coordinates)'.format(len(self.data), path))
        else:
            self.data = data
            print('Load {} data from {}'.format(len(self.data), path))

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['features'], self.data[idx]['Effect value']
