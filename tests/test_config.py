import pytest
from pydantic import ValidationError

from asl.lab.config import deep_merge, load_config, parse_override
from tests.conftest import ROOT


def test_params_yaml_is_valid():
    config = load_config([ROOT / "params.yaml"])
    assert config.model.name == "cnn"
    assert config.model.params["filters"] == [32, 64, 128]


@pytest.mark.parametrize("preset", sorted((ROOT / "asl/lab/configs").glob("*.yaml")))
def test_presets_are_valid(preset):
    load_config([ROOT / "params.yaml", preset])


def test_parse_override():
    assert parse_override("train.optimizer.learning_rate=3e-4") == {"train": {"optimizer": {"learning_rate": 3e-4}}}
    assert parse_override("model.cnn.filters=[8, 16]") == {"model": {"cnn": {"filters": [8, 16]}}}
    assert parse_override("data.test_dir=null") == {"data": {"test_dir": None}}
    assert parse_override("name=1e5x") == {"name": "1e5x"}
    with pytest.raises(ValueError):
        parse_override("no-equals-sign")


def test_deep_merge_keeps_siblings():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_overrides_switch_model_without_mixing_params():
    config = load_config([ROOT / "params.yaml"], ["model.name=mlp", "features.preprocess.representation=landmarks"])
    assert config.model.params == {"hidden": [256, 128], "batch_norm": True, "dropout": 0.3}


@pytest.mark.parametrize(
    "overrides",
    [
        ["model.name=mlp"],  # landmark model on image features
        ["model.name=resnet"],  # unknown model
        ["model.cnn.unknown=1"],  # typo in model params
        ["train.optimizer.name=rmsprop"],
        ["train.epochs=0"],
        ["split.strategy=group"],  # needs a group pattern
        ["features.preprocess.img_size=8"],
        ["train.typo=1"],
    ],
)
def test_invalid_configs_fail_early(overrides):
    with pytest.raises((ValidationError, ValueError)):
        load_config([ROOT / "params.yaml"], overrides)
