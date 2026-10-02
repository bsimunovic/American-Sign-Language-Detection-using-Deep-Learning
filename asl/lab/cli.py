"""Command line entry point for the experiment pipeline.

Each pipeline stage is a subcommand reading the same config::

    asl-lab split
    asl-lab featurize
    asl-lab train  --set train.epochs=5 --set model.name=legacy_cnn
    asl-lab evaluate
    asl-lab run    -c params.yaml -c asl/lab/configs/landmarks_mlp.yaml   # all stages

``-c/--config`` files are deep-merged in order (default ``params.yaml``) and
``-s/--set key.path=value`` overrides are applied last. Under DVC
(``dvc repro`` / ``dvc exp run -S ...``) the stages are invoked with the
default ``params.yaml`` and DVC handles caching and experiment tracking.
"""

from __future__ import annotations

import argparse

from asl.lab.config import dump_config, load_config

STAGES = ("split", "featurize", "train", "evaluate")


def run_stage(stage: str, config) -> None:
    if stage == "split":
        from asl.lab.data import run_split

        run_split(config)
    elif stage == "featurize":
        from asl.lab.features import run_featurize

        run_featurize(config)
    elif stage == "train":
        from asl.lab.train import run_train

        run_train(config)
    elif stage == "evaluate":
        from asl.lab.evaluate import run_evaluate

        run_evaluate(config)
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(stage)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asl-lab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", action="append", help="YAML config file(s), merged in order")
    common.add_argument("-s", "--set", action="append", default=[], metavar="KEY=VALUE", help="override a value")

    sub = parser.add_subparsers(dest="command", required=True)
    for stage in STAGES:
        sub.add_parser(stage, parents=[common], help=f"run the {stage} stage")
    run = sub.add_parser("run", parents=[common], help="run stages in order (without DVC caching)")
    run.add_argument("--from", dest="start", choices=STAGES, default=STAGES[0], help="first stage to run")
    sub.add_parser("show-config", parents=[common], help="print the resolved config")

    legacy = sub.add_parser("export-legacy", help="convert original .h5 weights into a model bundle")
    legacy.add_argument("weights")
    legacy.add_argument("--output-dir", default="artifacts/model")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "export-legacy":
        from asl.lab.export_legacy import export_legacy

        print(f"wrote {export_legacy(args.weights, args.output_dir)}")
        return

    config = load_config(args.config or ["params.yaml"], args.set)
    if args.command == "show-config":
        print(dump_config(config), end="")
    elif args.command == "run":
        for stage in STAGES[STAGES.index(args.start):]:
            print(f"=== {stage}")
            run_stage(stage, config)
    else:
        run_stage(args.command, config)


if __name__ == "__main__":
    main()
