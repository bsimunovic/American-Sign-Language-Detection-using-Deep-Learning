import json
from pathlib import Path

from asl.lab.data import read_classes, read_split, run_split


def test_stratified_split(make_config):
    config = make_config()
    splits = run_split(config)
    assert {name: len(items) for name, items in splits.items()} == {"train": 30, "val": 6, "test": 12}
    assert read_classes(config.paths.splits_dir) == ["a", "b", "c"]
    for label in "abc":
        assert sum(r.label == label for r in splits["val"]) == 2
    assert not {r.path for r in splits["train"]} & {r.path for r in splits["val"]}
    assert read_split(config.paths.splits_dir, "val") == splits["val"]


def test_split_is_deterministic(make_config):
    config = make_config()
    first = run_split(config)
    assert run_split(config) == first
    assert run_split(make_config("split.seed=7"))["val"] != first["val"]


def test_group_split_keeps_groups_together(make_config):
    config = make_config("split.strategy=group", r"split.group_pattern=^(signer\d+)_")
    splits = run_split(config)
    train_groups = {r.group for r in splits["train"]}
    val_groups = {r.group for r in splits["val"]}
    assert val_groups and not train_groups & val_groups
    assert train_groups | val_groups == {"signer0", "signer1", "signer2", "signer3"}


def test_test_split_carved_from_train(make_config):
    splits = run_split(make_config("data.test_dir=null", "split.test_fraction=0.25"))
    assert len(splits["test"]) == 9 and len(splits["train"]) + len(splits["val"]) == 27


def test_max_per_class_and_summary(make_config):
    config = make_config("split.max_per_class=5")
    splits = run_split(config)
    assert len(splits["train"]) + len(splits["val"]) == 15
    summary = json.loads((Path(config.paths.splits_dir) / "summary.json").read_text())
    assert summary["test"]["images"] == 12
