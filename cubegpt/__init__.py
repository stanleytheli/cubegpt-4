"""CubeGPT-3: transformer-based Rubik's cube distance estimators.

Submodules:
    config    - run/eval configuration + validation (no torch; safe to import locally)
    cube      - cube representation, symmetries, GPU data generation
    model     - model architectures
    train     - resumable, fail-fast training loop
    evaluate  - metrics (MSE / within-k accuracy) and beam-search solving

This package deliberately does not import torch at package level, so the local
Modal client can use `cubegpt.config` without torch installed.
"""
