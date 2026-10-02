"""``split`` stage: index the raw dataset and write train/val/test manifests.

Splits are written once as CSV files (``path,label,group``) so that every
experiment is trained and validated on exactly the same images and DVC only
re-splits when the data or ``split`` params change.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from asl.lab.config import ExperimentConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Record:
    path: str
    label: str
    group: str


def discover_classes(root) -> list[str]:
    classes = sorted(d.name for d in Path(root).iterdir() if d.is_dir() and not d.name.startswith("."))
    if not classes:
        raise ValueError(f"no class folders found in {root}")
    return classes


def list_records(root, classes: list[str], group_pattern: str | None = None) -> list[Record]:
    pattern = re.compile(group_pattern) if group_pattern else None
    records = []
    for label in classes:
        folder = Path(root) / label
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            match = pattern.search(path.name) if pattern else None
            group = f"{label}/{path.stem}" if match is None else match.group(1)
            records.append(Record(str(path), label, group))
    return records


def _cap_per_class(records: list[Record], cap: int | None, rng: random.Random) -> list[Record]:
    if cap is None:
        return records
    by_class: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_class[record.label].append(record)
    capped = []
    for label in sorted(by_class):
        items = by_class[label]
        capped += sorted(rng.sample(items, cap), key=lambda r: r.path) if len(items) > cap else items
    return capped


def _stratified(records: list[Record], fraction: float, rng: random.Random) -> tuple[list[Record], list[Record]]:
    by_class: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_class[record.label].append(record)
    keep, held_out = [], []
    for label in sorted(by_class):
        items = list(by_class[label])
        rng.shuffle(items)
        n = round(len(items) * fraction)
        if len(items) > 1:
            n = min(max(n, 1), len(items) - 1)
        held_out += items[:n]
        keep += items[n:]
    return keep, held_out


def _grouped(records: list[Record], fraction: float, rng: random.Random) -> tuple[list[Record], list[Record]]:
    by_group: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_group[record.group].append(record)
    groups = sorted(by_group)
    rng.shuffle(groups)
    target = len(records) * fraction
    held_groups, count = set(), 0
    for group in groups:
        if count >= target:
            break
        held_groups.add(group)
        count += len(by_group[group])
    keep = [r for r in records if r.group not in held_groups]
    held_out = [r for r in records if r.group in held_groups]
    return keep, held_out


def split_records(records: list[Record], fraction: float, strategy: str, rng: random.Random):
    return (_grouped if strategy == "group" else _stratified)(records, fraction, rng)


def write_split(path, records: list[Record]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "group"])
        writer.writerows((r.path, r.label, r.group) for r in records)


def read_split(splits_dir, name: str) -> list[Record]:
    with open(Path(splits_dir) / f"{name}.csv", newline="") as f:
        return [Record(row["path"], row["label"], row["group"]) for row in csv.DictReader(f)]


def read_classes(splits_dir) -> list[str]:
    return json.loads((Path(splits_dir) / "classes.json").read_text())


def run_split(config: ExperimentConfig) -> dict[str, list[Record]]:
    cfg = config.split
    rng = random.Random(cfg.seed)
    classes = config.data.classes or discover_classes(config.data.train_dir)

    records = _cap_per_class(list_records(config.data.train_dir, classes, cfg.group_pattern), cfg.max_per_class, rng)
    if not records:
        raise ValueError(f"no images found in {config.data.train_dir}")
    if config.data.test_dir:
        test = _cap_per_class(list_records(config.data.test_dir, classes, cfg.group_pattern), cfg.max_per_class, rng)
    else:
        records, test = split_records(records, cfg.test_fraction, cfg.strategy, rng)
    train, val = split_records(records, cfg.val_fraction, cfg.strategy, rng)
    splits = {"train": train, "val": val, "test": test}

    out = Path(config.paths.splits_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        write_split(out / f"{name}.csv", items)
    (out / "classes.json").write_text(json.dumps(classes, indent=2))
    summary = {name: {"images": len(items), "per_class": dict(sorted(Counter(r.label for r in items).items()))}
               for name, items in splits.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(", ".join(f"{name}: {len(items)} images" for name, items in splits.items()))
    return splits
