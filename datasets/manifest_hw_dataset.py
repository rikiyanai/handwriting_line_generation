import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import string_utils

PADDING_CONSTANT = -1


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    assert len(set([b["image"].shape[0] for b in batch])) == 1
    assert len(set([b["image"].shape[2] for b in batch])) == 1

    dim0 = batch[0]["image"].shape[0]
    dim1 = max([b["image"].shape[1] for b in batch])
    dim2 = batch[0]["image"].shape[2]

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i, entry in enumerate(batch):
        b_img = entry["image"]
        to_pad = dim1 - b_img.shape[1]
        if entry.get("center"):
            to_pad //= 2
        else:
            to_pad = 0
        input_batch[i, :, to_pad : to_pad + b_img.shape[1], :] = b_img

        label = entry["gt_label"]
        all_labels.append(label)
        label_lengths.append(len(label))

    label_lengths = torch.IntTensor(label_lengths)
    max_len = label_lengths.max()
    all_labels = [np.pad(l, ((0, max_len - l.shape[0]),), "constant") for l in all_labels]
    all_labels = np.stack(all_labels, axis=1)

    images = input_batch.transpose([0, 3, 1, 2])
    images = torch.from_numpy(images)
    labels = torch.from_numpy(all_labels.astype(np.int32))

    return {
        "image": images,
        "label": labels,
        "label_lengths": label_lengths,
        "gt": [b["gt"] for b in batch],
        "name": [b["name"] for b in batch],
        "author": [b["author"] for b in batch],
        "spaced_label": None,
        "style": None,
        "fg_mask": None,
    }


class ManifestHWDataset(Dataset):
    def __init__(self, dirPath, split, config):
        self.img_height = config["img_height"]
        self.center = config.get("center_pad", False)
        self.add_spaces = config.get("add_spaces", False)
        self.no_spaces = config.get("no_spaces", False)
        self.max_width = config.get("max_width")

        char_set_path = Path(config["char_file"])
        if not char_set_path.is_absolute():
            char_set_path = Path(dirPath) / char_set_path
        with char_set_path.open() as handle:
            char_set = json.load(handle)
        self.char_to_idx = char_set["char_to_idx"]

        manifest_path = Path(config.get("manifest_csv", "manifests/lines_v0.csv"))
        if not manifest_path.is_absolute():
            manifest_path = Path(dirPath) / manifest_path

        if isinstance(split, list):
            splits = set(split)
        else:
            splits = {split}

        self.samples = []
        self.authors = []
        self.max_char_len = 0
        author_set = set()
        with manifest_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("split") not in splits:
                    continue
                text = row.get("text", "")
                if self.no_spaces:
                    text = text.replace(" ", "")
                if self.add_spaces:
                    text = f" {text} "
                sample_path = Path(row["path"])
                if not sample_path.is_absolute():
                    sample_path = Path(dirPath) / sample_path
                author = row.get("source_doc") or row.get("author") or "unknown"
                author_set.add(author)
                self.max_char_len = max(self.max_char_len, len(text))
                self.samples.append(
                    {
                        "path": sample_path,
                        "text": text,
                        "id": row.get("id", sample_path.stem),
                        "author": author,
                    }
                )
        self.authors = sorted(author_set)

    def __len__(self):
        return len(self.samples)

    def max_len(self):
        return self.max_char_len

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img = cv2.imread(str(entry["path"]), 0)
        if img is None:
            return None

        if img.shape[0] != self.img_height:
            scale = float(self.img_height) / img.shape[0]
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if self.max_width is not None and img.shape[1] > self.max_width:
            return None

        if len(img.shape) == 2:
            img = img[..., None]

        img = img.astype(np.float32)
        img = 1.0 - img / 128.0

        gt = entry["text"]
        if len(gt) == 0:
            return None

        gt_label = string_utils.str2label_single(gt, self.char_to_idx)

        return {
            "image": img,
            "gt": gt,
            "gt_label": gt_label,
            "name": entry["id"],
            "center": self.center,
            "author": entry["author"],
        }
