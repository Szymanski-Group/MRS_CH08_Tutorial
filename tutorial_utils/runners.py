"""Notebook-friendly runner functions with customizable arguments.

These wrappers keep section flow explicit in notebooks while delegating
low-level implementation details to reusable module files.
"""

from __future__ import annotations

import contextlib
import csv
import io
import warnings
from pathlib import Path
from typing import Optional, Sequence

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


def _run_section_main(module, compact_output: bool):
    if not compact_output:
        module.main()
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module.main()


def _read_csv_rows(csv_path: Path):
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value, digits: int = 3):
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _print_compact_ml_1phase_summary(output_dir: Path):
    rows = _read_csv_rows(output_dir / "model_metrics.csv")
    print("  Accuracy summary (validation/test):")
    for row in rows:
        print(
            f"  - {row.get('model', 'model')}: "
            f"{_fmt(row.get('validation_accuracy'))}/{_fmt(row.get('test_accuracy'))}"
        )
    print(f"  Saved outputs: {output_dir}")


def _print_compact_ml_multiphase_summary(output_dir: Path):
    rows = _read_csv_rows(output_dir / "model_test_metrics.csv")
    print("  Multiphase test metrics (P/R/F1):")
    for row in rows:
        print(
            f"  - {row.get('model', 'model')}: "
            f"{_fmt(row.get('test_precision_micro'))}/"
            f"{_fmt(row.get('test_recall_micro'))}/"
            f"{_fmt(row.get('test_f1_micro'))} "
            f"(thr={_fmt(row.get('threshold'), 2)})"
        )
    print(f"  Saved outputs: {output_dir}")


def _print_compact_nn_1phase_summary(output_dir: Path):
    rows = _read_csv_rows(output_dir / "nn_metrics.csv")
    row = rows[0] if rows else {}
    print(
        "  Neural Net accuracy (validation/test): "
        f"{_fmt(row.get('validation_accuracy'))}/{_fmt(row.get('test_accuracy'))}"
    )
    print(f"  Saved outputs: {output_dir}")


def _print_compact_multilabel_nn_summary(output_dir: Path):
    rows = _read_csv_rows(output_dir / "nn_metrics.csv")
    row = rows[0] if rows else {}
    print(
        "  Validation P/R/F1: "
        f"{_fmt(row.get('validation_precision_micro'))}/"
        f"{_fmt(row.get('validation_recall_micro'))}/"
        f"{_fmt(row.get('validation_f1_micro'))}"
    )
    print(
        "  Test P/R/F1: "
        f"{_fmt(row.get('test_precision_micro'))}/"
        f"{_fmt(row.get('test_recall_micro'))}/"
        f"{_fmt(row.get('test_f1_micro'))} "
        f"(thr={_fmt(row.get('threshold'), 2)})"
    )
    print(f"  Saved outputs: {output_dir}")


def run_search_match(
    top_k_to_print: int = 3,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s02a,
        TOP_K_TO_PRINT=top_k_to_print,
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
    fwhm: Optional[float] = None,
    gauss_frac: Optional[float] = None,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s02b,
        TOP_K_TO_PRINT=top_k_to_print,
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
    prefilter_top_k: Optional[int] = 5,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
):
    _set_if_not_none(
        s02c,
        TOP_K_TO_PRINT=top_k_to_print,
        PATTERNS_TO_RUN=list(patterns_to_run) if patterns_to_run is not None else None,
        BACKGROUND_DEGREE=background_degree,
        FWHM_INIT=fwhm_init,
        PREFILTER_TOP_K=prefilter_top_k,
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
    synth_samples_per_formula: int = 80,
    random_seed: int = 42,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
    compact_output: bool = True,
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
    _run_section_main(s03a, compact_output=compact_output)
    if compact_output:
        _print_compact_ml_1phase_summary(s03a.OUTPUT_DIR)
    return s03a.OUTPUT_DIR


def run_ml_multiphase(
    synth_samples_per_formula: int = 60,
    random_seed: int = 42,
    single_phase_fraction: Optional[float] = None,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
    compact_output: bool = True,
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
    _run_section_main(s03b, compact_output=compact_output)
    if compact_output:
        _print_compact_ml_multiphase_summary(s03b.OUTPUT_DIR)
    return s03b.OUTPUT_DIR


def run_nn_1phase(
    synth_samples_per_formula: int = 80,
    nn_max_iter: int = 220,
    hidden_layer_sizes: Optional[tuple[int, ...]] = (128, 64),
    alpha: Optional[float] = 1e-4,
    learning_rate_init: Optional[float] = 1e-3,
    random_seed: int = 42,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
    compact_output: bool = True,
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
    _run_section_main(s03c, compact_output=compact_output)
    if compact_output:
        _print_compact_nn_1phase_summary(s03c.OUTPUT_DIR)
    return s03c.OUTPUT_DIR


def run_nn_multiphase(
    synth_samples_per_formula: int = 60,
    nn_max_iter: int = 220,
    hidden_layer_sizes: Optional[tuple[int, ...]] = (128, 64),
    alpha: Optional[float] = 1e-4,
    learning_rate_init: Optional[float] = 1e-3,
    random_seed: int = 42,
    prediction_threshold: Optional[float] = None,
    output_dir: Optional[str] = None,
    show_steps: bool = True,
    compact_output: bool = True,
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
    _run_section_main(s03d, compact_output=compact_output)
    if compact_output:
        _print_compact_multilabel_nn_summary(s03d.OUTPUT_DIR)
    return s03d.OUTPUT_DIR

def run_cnn_multiphase(
    synth_samples_per_formula: int = 60,
    nn_max_iter: int = 50,
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
    compact_output: bool = True,
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
    _run_section_main(s03e, compact_output=compact_output)
    if compact_output:
        _print_compact_multilabel_nn_summary(s03e.OUTPUT_DIR)
    return s03e.OUTPUT_DIR


def run_cnn_no_augmentation(
    synth_samples_per_formula: int = 60,
    nn_max_iter: int = 220,
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
    compact_output: bool = True,
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
    _run_section_main(s03f, compact_output=compact_output)
    if compact_output:
        _print_compact_multilabel_nn_summary(s03f.OUTPUT_DIR)
    return s03f.OUTPUT_DIR


def run_cnn_random_shifts(
    synth_samples_per_formula: int = 60,
    nn_max_iter: int = 220,
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
    compact_output: bool = True,
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
    _run_section_main(s03g, compact_output=compact_output)
    if compact_output:
        _print_compact_multilabel_nn_summary(s03g.OUTPUT_DIR)
    return s03g.OUTPUT_DIR

def run_mixture_of_experts(
    synth_samples_per_formula: int = 200,
    nn_max_iter: int = 50,
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
    compact_output: bool = True,
):
    _set_if_not_none(
        s03h,
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
        "Mixture of Experts (Multiphase)",
        (
            "Build synthetic multiphase dataset",
            "Train one binary expert per phase",
            "Aggregate expert outputs",
            "Evaluate micro precision/recall/F1",
        ),
        show_steps,
    )
    _run_section_main(s03h, compact_output=compact_output)
    if compact_output:
        _print_compact_multilabel_nn_summary(s03h.OUTPUT_DIR)
    return s03h.OUTPUT_DIR
