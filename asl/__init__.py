"""American Sign Language (ASL) alphabet and digit recognition.

``asl.prod`` holds everything needed to serve a trained model (preprocessing,
model bundles, inference and the FastAPI service). ``asl.lab`` holds the
experiment pipeline and depends on ``asl.prod`` for preprocessing, so training
and serving always transform images identically. ``asl.prod`` never imports
``asl.lab``.
"""

__version__ = "0.2.0"
