"""Experiment lab: a configurable split -> featurize -> train -> evaluate pipeline.

Every stage is a function of an :class:`~asl.lab.config.ExperimentConfig`; the
same stages are wired into ``dvc.yaml`` so DVC can cache them and track
experiments. Preprocessing comes from :mod:`asl.prod.preprocessing`, so
features are built exactly the way the API builds them.
"""
