# EMTscore Python port (`Python_Conv`)

A publishable, self-contained Python package and notebook that mirrors the R
[EMTscore]((https://github.com/wenmm/EMTscore)) vignette (`EMTscore.html`)
section-by-section. The package ships with:

- **Reusable scoring modules** in `emtscore/` — the four building blocks used
  throughout the vignette (`nsprcomp`, `nnpca`, `aucell`, `ssGSEA`).
- **Data loader** in `utility/load_cook2020.py` for the three Cook *et al.*
  2020 single-cell datasets used in Section 3.
- **Gene signatures and reference GMTs** in `data/`.
- **Two notebooks** in `notebooks/`:
  - `EMTscore_full_analysis.ipynb` — the validated, cell-by-cell working copy
    of the original analysis (49 cells covering §2.1 – §3.6).
  - `EMTscore_automated.ipynb` — a slimmed-down, publication-style notebook
    that calls the `emtscore/` API and reproduces every figure from the R
    vignette in five clean cells.

## Installation

```bash
pip install -e .
```

or, from a fresh environment:

```bash
pip install -r requirements.txt
```

Python ≥ 3.10 is required.

## Publish as a standalone repo (or zip and ship it)

`Python_Conv/` is fully self-contained — you can copy or zip the folder
and publish it as its own GitHub repository without bringing the rest
of `PythonLab` along. After cloning/unzipping a fresh copy:

```bash
cd Python_Conv
pip install -e .
jupyter notebook notebooks/EMTscore_automated.ipynb
```

§2.1–§2.11, §3.1.1, §3.2, §3.2.1, and §3.3 will run end-to-end on the
bundled data alone. §3.1, §3.4, §3.5, and §3.6 require the heavy
`*_counts.csv` files — see *Sections that need extra data* below for
the one-time Zenodo download. The `.gitignore` already excludes those
files so they will never accidentally be committed if you do place
them in `data/cook2020/` locally.

## What ships with the package (and what doesn't)

The repo bundles every file needed to reproduce most of the EMTscore
vignette out of the box. Heavy raw-count matrices are excluded — they
exceed GitHub's 100 MB per-file limit — but you can fetch them on demand.

**Bundled (works after `git clone`):**

| File | Size | Used by |
|---|---|---|
| `data/Panchy_et_al_{E,M}_signature.csv` | <2 KB | §2.1–2.11, §3.1.1, §3.2 |
| `data/EM_signature.gmt` | 3 KB | §2.5 (multi-method scoring) |
| `data/cell_annotation_file.csv` | 2 KB | §2.6–2.10 |
| `data/geneExp.csv` | 19 MB | §2.1–2.11 (bulk RNA-seq input) |
| `data/stemsig.tsv` | 0.5 KB | §3.5 (stemness signature) |
| `data/cellular_senescence_sig.tsv` | 1 KB | §3.5 (senescence signature) |
| `data/filtered.c2.gmt` | 10 KB | §3.3 (small fallback pathway set) |
| `data/c2.all.v2025.1.Hs.symbols.gmt` | 4.4 MB | §3.3 (full MSigDB C2) |
| `data/TianLab_collected_EMT_signatures.gmt` | 70 KB | §3.3 (reference EMT sets) |
| `data/cook2020/A549_*_em_expr.csv` | ~19 MB total | §3.1.1, §3.2, §3.2.1 |

**Sections that run with the bundled data only:**
§2.1 – §2.11 (bulk RNA-seq), §3.1.1 (EMT vs pseudotime), §3.2 (GMM),
§3.2.1 (Sankey), §3.3 (pathway correlation).

**Sections that need extra data:**
§3.1 (`load_cook_adatas` builds a full AnnData), §3.4 (E-vs-M & PC1-vs-PC2
panels), §3.5 (stemness/senescence), §3.6 (integration). These call
`load_cook2020(...)` which reads the per-cell `*_counts.csv`,
`*_genes.csv`, `*_cells.csv`, `*_metadata.csv` files. The counts CSVs are
150–576 MB each.

**To enable §3.1 / §3.4 / §3.5 / §3.6:**

1. Download the Cook *et al.* 2020 CSV exports from the project's Zenodo
   archive ([10.5281/zenodo.18489669](https://doi.org/10.5281/zenodo.18489669)).
2. Place the 12 files (`A549_{TNF,EGF,TGFB1}_{counts,cells,genes,metadata}.csv`)
   inside `data/cook2020/`.
3. Re-run the notebook — `resolve_cook_dir` will pick them up automatically.

## Quick-start

```python
import pandas as pd
from emtscore import (
    run_nnPCA, execute_aucell, execute_ssgsva, compute_M1_M2_scores,
)

# geneExp is a (samples × genes) DataFrame of log-normalised expression
geneExp = pd.read_csv("data/geneExp.csv", index_col=0)

nnPCA  = run_nnPCA(geneExp,      gmt_file="data/EM_signature.gmt", dimension=1)
aucell = execute_aucell(geneExp, gmt_file="data/EM_signature.gmt")
ssgsea = execute_ssgsva(geneExp, gmt_file="data/EM_signature.gmt")
```

## Package layout

```
Python_Conv/
├── emtscore/                 ← reusable scoring library (public API)
│   ├── __init__.py           ← curated re-exports (import what you need)
│   ├── nnpca.py              ← non-negative sparse-PCA scoring
│   ├── nsprcomp.py           ← the nsprcomp solver + compute_M1_M2_scores
│   ├── aucell.py             ← AUCell-style top-fraction enrichment
│   └── ssGSEA.py             ← single-sample GSEA
├── utility/
│   ├── __init__.py
│   └── load_cook2020.py    
