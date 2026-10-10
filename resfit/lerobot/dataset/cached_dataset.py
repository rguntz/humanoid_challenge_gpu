"""A LeRobotDataset that reads its videos and action chunks from memory.

Per sample, LeRobotDataset decodes the requested video frames (~1.1 ms) and queries
the Arrow table for the action chunk (~1.2 ms), which made data loading the
bottleneck of training (~0.5 s per batch of 256 with 8 workers). The pick-and-place
datasets are small (6.6k frames, two 128x128 cameras: ~650 MB as uint8), so every
frame is decoded once and every column read once, and kept in shared memory (read by
every DataLoader worker without a copy). Only these two lookups change: which frames
and actions are fetched, the padding and the image augmentations are LeRobotDataset's
own code.
"""
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import decode_video_frames


class CachedVideoLeRobotDataset(LeRobotDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        starts = self.episode_data_index["from"].tolist()
        ends = self.episode_data_index["to"].tolist()
        self._frames = {}
        for key in self.meta.video_keys:
            per_episode = []
            for ep_idx, (i0, i1) in enumerate(zip(starts, ends)):
                path = self.root / self.meta.get_video_file_path(ep_idx, key)
                frames = decode_video_frames(path, timestamps[i0:i1].tolist(), self.tolerance_s, self.video_backend)
                per_episode.append((frames * 255).round().to(torch.uint8))  # decoded as uint8 / 255
            self._frames[key] = torch.cat(per_episode).share_memory_()
        self._episode_start = starts
        self._columns = {key: torch.stack(self.hf_dataset[key]).share_memory_() for key in self.delta_indices or {}
                         if key not in self.meta.video_keys}

    def _query_hf_dataset(self, query_indices):
        return {key: self._columns[key][q_idx] for key, q_idx in query_indices.items() if key not in self.meta.video_keys}

    def _query_videos(self, query_timestamps, ep_idx):
        start = self._episode_start[ep_idx]
        item = {}
        for key, query_ts in query_timestamps.items():
            index = [start + round(ts * self.fps) for ts in query_ts]
            item[key] = (self._frames[key][index].float() / 255).squeeze(0)
        return item
