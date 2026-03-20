# MRS-XRD-Tutorial

Google Colab notebooks for the Spring MRS tutorial on machine learning for powder XRD.

## Open the Tutorial

Start here: [00_START_HERE.ipynb](https://colab.research.google.com/github/njszym/MRS-XRD-Tutorial/blob/main/00_START_HERE.ipynb)

## Repository Layout

- `00_START_HERE.ipynb`: entry-point and Colab table of contents
- `01_Pattern-Generation/`: physics-based synthetic pattern artifacts
- `02_Conventional-Methods/`: search-match, correlation, and Rietveld-style workflows
- `03_Machine-Learning/`: conventional ML (single-phase and multi-phase)
- `04_Deep-Learning/`: NN/CNN modules and augmentation ablations
- `05_Generative-AI/`: diffusion demo and structure sanity checks
- `06_Challenge/`: open challenge notebook
- `data/`: CIFs, reference structures, and experimental/mystery patterns
- `xrd_tutorial/`: shared Python modules used by notebooks (to avoid repeated code)

## Notes

- Training notebooks use reduced defaults for live Colab runtime.
- You can increase dataset size/epochs in each notebook for offline runs.
- Optional pretrained checkpoints can be placed under `data/pretrained/`.
