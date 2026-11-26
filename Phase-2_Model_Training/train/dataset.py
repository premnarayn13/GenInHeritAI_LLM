"""
GenInherit Shard Dataset & Streaming Batch Loader (Pillar 2 Specification)
Streams packed binary PyTorch shards (.pt) from Google Drive / local disk.
"""

import os
import glob
from typing import List, Dict, Generator
import torch
from torch.utils.data import Dataset, DataLoader


class ShardDataset(Dataset):
    def __init__(self, shard_paths: List[str]):
        self.shard_paths = shard_paths
        self.current_shard_idx = -1
        self.cached_data: Dict[str, torch.Tensor] = {}
        self.total_samples = 0
        self._index_dataset()

    def _index_dataset(self):
        # Quick index of total sample count across shards
        total = 0
        for p in self.shard_paths:
            d = torch.load(p, map_location="cpu", weights_only=True)
            total += d["tok"].shape[0]
        self.total_samples = total

    def __len__(self):
        return self.total_samples


class StreamingShardLoader:
    def __init__(self, shards_dir: str, micro_batch: int = 4, shuffle: bool = True):
        self.shards_dir = shards_dir
        self.micro_batch = micro_batch
        self.shuffle = shuffle
        pattern = os.path.join(shards_dir, "shard_*.pt")
        self.shard_paths = sorted(glob.glob(pattern))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files found in {shards_dir}")

    def stream_batches(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        paths = list(self.shard_paths)
        while True:
            if self.shuffle:
                import random
                random.shuffle(paths)

            for p in paths:
                shard = torch.load(p, map_location="cpu", weights_only=True)
                n = shard["tok"].shape[0]
                indices = torch.randperm(n) if self.shuffle else torch.arange(n)

                for start in range(0, n, self.micro_batch):
                    end = min(start + self.micro_batch, n)
                    batch_idx = indices[start:end]
                    batch = {
                        "tok": shard["tok"][batch_idx],
                        "pos": shard["pos"][batch_idx],
                        "slot": shard["slot"][batch_idx],
                        "gen": shard["gen"][batch_idx],
                        "fld": shard["fld"][batch_idx],
                        "num": shard["num"][batch_idx],
                        "y": shard["y"][batch_idx],
                        "w": shard["w"][batch_idx],
                        "m": shard["m"][batch_idx]
                    }
                    yield batch

