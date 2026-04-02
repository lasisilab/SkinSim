# SkinSim

SkinSim is a Quarto-based research website for statistical modeling of human skin reflectance spectra.

The project integrates real measurements (ISSA) and simulated spectra to support:

- reproducible spectral analysis,
- biologically informed modeling (melanin and erythema effects),
- synthetic spectrum generation and validation.

## Project structure

- `analysis/`: Quarto analysis reports (`.qmd`)
- `notebooks/`: Jupyter notebooks (`.ipynb`)
- `code/`: reusable Python and R modules/scripts
- `data/`: source and derived datasets
- `output/`: generated figures, tables, and exports
- `docs/`: rendered static website output

## Environment setup

### R environment

```r
source("renv-setup.R")
```

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local development

Run the live preview server:

```bash
quarto preview
```

## Render website

Build the full site to `docs/`:

```bash
quarto render
```

## Recommended workflow

1. Add or update analysis content in `analysis/` or `notebooks/`.
2. Keep reusable model code in `code/`.
3. Store intermediate and final artifacts in `output/`.
4. Preview frequently with `quarto preview`.
5. Commit both source files and rendered updates when publishing.

## Notes

- The website uses Quarto navigation and supports both R and Python execution.
- Notebook pages are rendered as part of the same site output.
