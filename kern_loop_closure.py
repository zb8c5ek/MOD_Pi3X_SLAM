"""
kern_loop_closure - SALAD-based visual place recognition for loop closure detection.

Uses DINOv2 backbone + SALAD aggregator to produce 8448-dim global descriptors.
Finds loop closure candidates via L2 distance brute-force search.

Ported from vggt_slam/loop_closure.py.
Requires: pip install -e 3rdParty/salad/ (see 3rdParty/MANIFEST.md)
"""

import os
import torch
import numpy as np
import heapq
from typing import NamedTuple
import torchvision.transforms as T

from salad.eval import load_model

tensor_transform = T.ToPILImage()
denormalize = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])


def input_transform(image_size=None):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    transform_list = [T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
    if image_size:
        transform_list.insert(0, T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))
    return T.Compose(transform_list)


class LoopMatch(NamedTuple):
    similarity_score: float
    query_submap_id: int
    query_submap_frame: int
    detected_submap_id: int
    detected_submap_frame: int


class LoopMatchQueue:
    """Bounded max-heap for top-k loop closure matches (lowest L2 = best)."""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.heap = []

    def add(self, match: LoopMatch):
        item = (-match.similarity_score, match)
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, item)
        else:
            heapq.heappushpop(self.heap, item)

    def get_matches(self):
        return [match for _, match in sorted(self.heap, reverse=True)]


class ImageRetrieval:
    def __init__(self, input_size=224, device='cuda'):
        self.device = device
        ckpt_pth = os.path.join(torch.hub.get_dir(), "checkpoints/dino_salad.ckpt")
        self.model = load_model(ckpt_pth)
        self.model.eval()
        self.transform = input_transform((input_size, input_size))

    def get_single_embedding(self, cv_img):
        with torch.no_grad():
            pil_img = self.transform(tensor_transform(cv_img))
            return self.model(pil_img.to(self.device))

    _SALAD_BATCH = 16

    def get_batch_descriptors(self, imgs):
        with torch.no_grad():
            pil_imgs = [self.transform(tensor_transform(img)) for img in imgs]
            n = len(pil_imgs)
            if n <= self._SALAD_BATCH:
                batch = torch.stack(pil_imgs).to(self.device)
                return self.model(batch)
            all_descs = []
            for i in range(0, n, self._SALAD_BATCH):
                batch = torch.stack(pil_imgs[i:i + self._SALAD_BATCH]).to(self.device)
                descs = self.model(batch)
                all_descs.append(descs.cpu())
                print(f"    [SALAD] {min(i + self._SALAD_BATCH, n)}/{n} embeddings",
                      flush=True)
            return torch.cat(all_descs, dim=0).to(self.device)

    def get_all_submap_embeddings(self, submap):
        frames = submap.get_all_frames()
        print(f"    [SALAD] Computing embeddings for {len(frames)} frames...",
              flush=True)
        return self.get_batch_descriptors(frames)

    def find_loop_closures(self, map, submap, max_similarity_thres=0.80, max_loop_closures=0):
        """
        Search all past submaps for loop closure candidates.
        Returns list of LoopMatch sorted by score (best first).
        """
        matches_queue = LoopMatchQueue(max_size=max_loop_closures)
        query_id = 0
        for query_vector in submap.get_all_retrieval_vectors():
            best_score, best_submap_id, best_frame_id = map.retrieve_best_score_frame(
                query_vector, submap.get_id(), ignore_last_submap=True
            )
            if best_score < max_similarity_thres:
                new_match_data = LoopMatch(best_score, submap.get_id(), query_id, best_submap_id, best_frame_id)
                matches_queue.add(new_match_data)
            query_id += 1

        return matches_queue.get_matches()
