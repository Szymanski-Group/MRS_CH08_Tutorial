"""Notebook-friendly runner functions with customizable arguments.

These wrappers keep section flow explicit in notebooks while delegating
low-level implementation details to reusable module files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .sections import challenge_baseline as s04a
from .sections import conventional_profile_correlation as s02b
from .sections import conventional_rietveld as s02c
from .sections import conventional_search_match as s02a
from .sections import dl_cnn_multiphase as s03e
from .sections import dl_cnn_no_augmentation as s03f
from .sections import dl_cnn_random_shifts as s03g
from .sections import dl_moe as s03h
from .sections import dl_nn_1phase as s03c
from .sections import dl_nn_multiphase as s03d
from .sections import ml_conv_1phase as s03a
from .sections import ml_multiphase as s03b


def _maybe_path(value: Optional[str]):
    return None if value is None else Path(value)


def _set_if_not_none(module, **overrides):
    for key, value in overrides.items():
        if value is not None:
            setattr(module, key, value)


def _print_steps(title: str, steps: Sequence[str], show_steps: bool):
    if not show_steps:
        return
    print(f"\n{title}")
    for i, step in enumerate(steps, start=1):
        print(f"  Step {i}: {step}")


def run_search_match(
    top_k_to_print: int = 3,
    max_experiment_patterns: Optional[int] = 4,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s02a,
        TOP_K_TO_PRINT=top_k_to_print,
        MAX_EXPERIMENT_PATTERNS=max_experiment_patterns,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "Search-Match Pipeline",
        (
            "Load experimental patterns and reference sticks",
            "Detect major peaks in each pattern",
            "Rank phases by de Wolff and Smith-Snyder FoM",
            "Save rankings and summary plots",
        ),
        show_steps,
    )
    s02a.main()
    return s02a.OUTPUT_DIR


def run_profile_correlation(
    top_k_to_print: int = 3,
    max_experiment_patterns: Optional[int] = 4,
    fwhm: Optional[float] = None,
    gauss_frac: Optional[float] = None,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s02b,
        TOP_K_TO_PRINT=top_k_to_print,
        MAX_EXPERIMENT_PATTERNS=max_experiment_patterns,
        FWHM=fwhm,
        GAUSS_FRAC=gauss_frac,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "Full-Profile Correlation Pipeline",
        (
            "Load experimental and reference patterns",
            "Simulate broadened reference profiles",
            "Compute Pearson and cosine similarity",
            "Save ranking table and comparison plots",
        ),
        show_steps,
    )
    s02b.main()
    return s02b.OUTPUT_DIR


def run_rietveld_sequential(
    top_k_to_print: int = 3,
    patterns_to_run: Optional[Sequence[str]] = ("TiO2", "ZrO2"),
    background_degree: Optional[int] = None,
    fwhm_init: Optional[float] = None,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s02c,
        TOP_K_TO_PRINT=top_k_to_print,
        PATTERNS_TO_RUN=list(patterns_to_run) if patterns_to_run is not None else None,
        BACKGROUND_DEGREE=background_degree,
        FWHM_INIT=fwhm_init,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "Sequential Rietveld-Style Pipeline",
        (
            "Load experimental pattern and candidate structures",
            "Coarse-fit scale and background",
            "Refine lattice scales and peak width sequentially",
            "Rank by Rwp/Pearson and save plots",
        ),
        show_steps,
    )
    s02c.main()
    return s02c.OUTPUT_DIR


def run_ml_conv_1phase(
    synth_samples_per_formula: int = 12,
    random_seed: int = 42,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s03a,
        SYNTH_SAMPLES_PER_FORMULA=synth_samples_per_formula,
        RANDOM_SEED=random_seed,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "Conventional ML (1-Phase)",
        (
            "Build synthetic artifact-rich training profiles",
            "Split into train/validation",
            "Tune and compare k-NN / Random Forest / SVM",
            "Evaluate on experimental patterns and save metrics",
        ),
        show_steps,
    )
    s03a.main()
    return s03a.OUTPUT_DIR


def run_ml_multiphase(
    synth_samples_per_formula: int = 10,
    random_seed: int = 42,
    single_phase_fraction: Optional[float] = None,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s03b,
        SYNTH_SAMPLES_PER_FORMULA=synth_samples_per_formula,
        RANDOM_SEED=random_seed,
        SINGLE_PHASE_FRACTION=single_phase_fraction,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "Conventional ML (Multiphase)",
        (
            "Generate synthetic multiphase mixtures",
            "Train one-vs-rest classifiers",
            "Pick threshold on validation data",
            "Score micro precision/recall/F1 on test patterns",
        ),
        show_steps,
    )
    s03b.main()
    return s03b.OUTPUT_DIR


def run_nn_1phase(
    synth_samples_per_formula: int = 12,
    nn_max_iter: int = 80,
    hidden_layer_sizes: Optional[tuple[int, ...]] = (128, 64),
    alpha: Optional[float] = 1e-4,
    learning_rate_init: Optional[float] = 1e-3,
    random_seed: int = 42,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s03c,
        SYNTH_SAMPLES_PER_FORMULA=synth_samples_per_formula,
        NN_MAX_ITER=nn_max_iter,
        NN_HIDDEN_LAYER_SIZES=hidden_layer_sizes,
        NN_ALPHA=alpha,
        NN_LEARNING_RATE_INIT=learning_rate_init,
        RANDOM_SEED=random_seed,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "Neural Network (1-Phase)",
        (
            "Generate synthetic training set",
            "Train feed-forward neural network",
            "Track loss curve",
            "Evaluate on experimental patterns",
        ),
        show_steps,
    )
    s03c.main()
    return s03c.OUTPUT_DIR


def run_nn_multiphase(
    synth_samples_per_formula: int = 8,
    nn_max_iter: int = 80,
    hidden_layer_sizes: Optional[tuple[int, ...]] = (128, 64),
    alpha: Optional[float] = 1e-4,
    learning_rate_init: Optional[float] = 1e-3,
    random_seed: int = 42,
    prediction_threshold: Optional[float] = None,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s03d,
        SYNTH_SAMPLES_PER_FORMULA=synth_samples_per_formula,
        NN_MAX_ITER=nn_max_iter,
        NN_HIDDEN_LAYER_SIZES=hidden_layer_sizes,
        NN_ALPHA=alpha,
        NN_LEARNING_RATE_INIT=learning_rate_init,
        RANDOM_SEED=random_seed,
        PREDICTION_THRESHOLD=prediction_threshold,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "Neural Network (Multiphase)",
        (
            "Build synthetic multiphase dataset",
            "Train multi-label neural network",
            "Apply threshold to label scores",
            "Report micro precision/recall/F1",
        ),
        show_steps,
    )
    s03d.main()
    return s03d.OUTPUT_DIR


def run_cnn_multiphase(
    synth_samples_per_formula: int = 8,
    nn_max_iter: int = 20,
    conv_channels: Optional[tuple[int, ...]] = (16, 32),
    kernel_sizes: Optional[tuple[int, ...]] = (7, 5),
    pool_kernel_size: Optional[int] = 2,
    hidden_layer_sizes: Optional[tuple[int, ...]] = (128, 64),
    alpha: Optional[float] = 1e-4,
    learning_rate_init: Optional[float] = 1e-3,
    batch_size: Optional[int] = 32,
    prediction_threshold: Optional[float] = 0.50,
    random_seed: int = 42,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s03e,
        SYNTH_SAMPLES_PER_FORMULA=synth_samples_per_formula,
        NN_MAX_ITER=nn_max_iter,
        CNN_CONV_CHANNELS=conv_channels,
        CNN_KERNEL_SIZES=kernel_sizes,
        CNN_POOL_KERNEL_SIZE=pool_kernel_size,
        NN_HIDDEN_LAYER_SIZES=hidden_layer_sizes,
        NN_ALPHA=alpha,
        NN_LEARNING_RATE_INIT=learning_rate_init,
        NN_BATCH_SIZE=batch_size,
        PREDICTION_THRESHOLD=prediction_threshold,
        RANDOM_SEED=random_seed,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "CNN (Multiphase)",
        (
            "Build augmented multiphase dataset",
            "Train 1D CNN model",
            "Track loss curve",
            "Evaluate and save test metrics",
        ),
        show_steps,
    )
    s03e.main()
    return s03e.OUTPUT_DIR


def run_cnn_no_augmentation(
    synth_samples_per_formula: int = 8,
    nn_max_iter: int = 20,
    conv_channels: Optional[tuple[int, ...]] = (16, 32),
    kernel_sizes: Optional[tuple[int, ...]] = (7, 5),
    pool_kernel_size: Optional[int] = 2,
    hidden_layer_sizes: Optional[tuple[int, ...]] = (128, 64),
    alpha: Optional[float] = 1e-4,
    learning_rate_init: Optional[float] = 1e-3,
    batch_size: Optional[int] = 32,
    prediction_threshold: Optional[float] = 0.50,
    random_seed: int = 42,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s03f,
        SYNTH_SAMPLES_PER_FORMULA=synth_samples_per_formula,
        NN_MAX_ITER=nn_max_iter,
        CNN_CONV_CHANNELS=conv_channels,
        CNN_KERNEL_SIZES=kernel_sizes,
        CNN_POOL_KERNEL_SIZE=pool_kernel_size,
        NN_HIDDEN_LAYER_SIZES=hidden_layer_sizes,
        NN_ALPHA=alpha,
        NN_LEARNING_RATE_INIT=learning_rate_init,
        NN_BATCH_SIZE=batch_size,
        PREDICTION_THRESHOLD=prediction_threshold,
        RANDOM_SEED=random_seed,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "CNN Ablation (No Augmentation)",
        (
            "Build near-ideal synthetic dataset",
            "Train 1D CNN model",
            "Track loss curve",
            "Evaluate and save test metrics",
        ),
        show_steps,
    )
    s03f.main()
    return s03f.OUTPUT_DIR


def run_cnn_random_shifts(
    synth_samples_per_formula: int = 8,
    nn_max_iter: int = 20,
    conv_channels: Optional[tuple[int, ...]] = (16, 32),
    kernel_sizes: Optional[tuple[int, ...]] = (7, 5),
    pool_kernel_size: Optional[int] = 2,
    hidden_layer_sizes: Optional[tuple[int, ...]] = (128, 64),
    alpha: Optional[float] = 1e-4,
    learning_rate_init: Optional[float] = 1e-3,
    batch_size: Optional[int] = 32,
    prediction_threshold: Optional[float] = 0.50,
    random_seed: int = 42,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s03g,
        SYNTH_SAMPLES_PER_FORMULA=synth_samples_per_formula,
        NN_MAX_ITER=nn_max_iter,
        CNN_CONV_CHANNELS=conv_channels,
        CNN_KERNEL_SIZES=kernel_sizes,
        CNN_POOL_KERNEL_SIZE=pool_kernel_size,
        NN_HIDDEN_LAYER_SIZES=hidden_layer_sizes,
        NN_ALPHA=alpha,
        NN_LEARNING_RATE_INIT=learning_rate_init,
        NN_BATCH_SIZE=batch_size,
        PREDICTION_THRESHOLD=prediction_threshold,
        RANDOM_SEED=random_seed,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "CNN Ablation (Random Shifts)",
        (
            "Build shift-only synthetic dataset",
            "Train 1D CNN model",
            "Track loss curve",
            "Evaluate and save test metrics",
        ),
        show_steps,
    )
    s03g.main()
    return s03g.OUTPUT_DIR


def run_mixture_of_experts(
    synth_samples_per_formula: int = 6,
    nn_max_iter: int = 20,
    conv_channels: Optional[tuple[int, ...]] = (6, 12),
    kernel_sizes: Optional[tuple[int, ...]] = (5, 3),
    pool_kernel_size: Optional[int] = 2,
    adaptive_pool_points: Optional[int] = 32,
    hidden_layer_sizes: Optional[tuple[int, ...]] = (16, 8),
    alpha: Optional[float] = 1e-4,
    learning_rate_init: Optional[float] = 1e-3,
    batch_size: Optional[int] = 64,
    prediction_threshold: Optional[float] = 0.50,
    min_epochs: int = 5,
    patience: int = 4,
    random_seed: int = 42,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s03h,
        SYNTH_SAMPLES_PER_FORMULA=synth_samples_per_formula,
        NN_MAX_ITER=nn_max_iter,
        CNN_CONV_CHANNELS=conv_channels,
        CNN_KERNEL_SIZES=kernel_sizes,
        CNN_POOL_KERNEL_SIZE=pool_kernel_size,
        CNN_ADAPTIVE_POOL_POINTS=adaptive_pool_points,
        NN_HIDDEN_LAYER_SIZES=hidden_layer_sizes,
        NN_ALPHA=alpha,
        NN_LEARNING_RATE_INIT=learning_rate_init,
        NN_BATCH_SIZE=batch_size,
        PREDICTION_THRESHOLD=prediction_threshold,
        NN_EARLY_STOPPING_MIN_EPOCHS=min_epochs,
        NN_EARLY_STOPPING_PATIENCE=patience,
        RANDOM_SEED=random_seed,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "Mixture of Experts (Multiphase)",
        (
            "Build synthetic multiphase dataset",
            "Train one binary expert per phase",
            "Apply early stopping and aggregate outputs",
            "Evaluate micro precision/recall/F1",
        ),
        show_steps,
    )
    s03h.main()
    return s03h.OUTPUT_DIR


def run_challenge_baseline(
    top_k_to_print: int = 5,
    max_experiment_patterns: Optional[int] = None,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s04a,
        TOP_K_TO_PRINT=top_k_to_print,
        MAX_EXPERIMENT_PATTERNS=max_experiment_patterns,
        OUTPUT_DIR=_maybe_path(output_dir),
    )
    _print_steps(
        "Challenge Baseline (Profile Correlation)",
        (
            "Load mystery patterns and reference library",
            "Rank phases by profile correlation",
            "Save summary plots",
            "Use predictions for ground-truth comparison",
        ),
        show_steps,
    )
    s04a.main()
    return s04a.OUTPUT_DIR


def get_challenge_predictions():
    """Return top-1 predicted phase for each challenge pattern."""
    ref_lib = s04a.load_reference_stick_library(sorted(Path("data/reference_structures").glob("*.cif")))
    mystery_files = sorted(Path("data/challenge/mystery_patterns").glob("*.xy"))
    predictions = []
    for fpath in mystery_files:
        tt, y = s04a.load_experimental_profile(fpath)
        by_pearson, _, _ = s04a.rank_phases(y, tt, ref_lib)
        predictions.append((fpath.name, by_pearson[0]["phase"]))
    return predictions
