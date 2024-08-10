from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyPairDataset(Dataset):
    def __init__(
        self,
        path: str,
        targetDtype: Optional[np.dtype] = None,
        useMmap: bool = True,
        transform:  Callable[[torch.Tensor], torch.Tensor] = lambda x:x,
        transform1: Callable[[torch.Tensor], torch.Tensor] = lambda x:x,
        transform2: Callable[[torch.Tensor], torch.Tensor] = lambda x:x,
    ):
        self._transform1 = lambda x: transform1(transform(x))
        self._transform2 = lambda x: transform2(transform(x))

        self.data = np.load(path, mmap_mode=("r" if useMmap else None))

        self._targetDtype = targetDtype or self.data.dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self._transform1(torch.from_numpy(self.data[index][0].astype(self._targetDtype, copy=True))),
            self._transform2(torch.from_numpy(self.data[index][1].astype(self._targetDtype, copy=True)))
        )
