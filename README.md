# Penalized Portfolio Optimization
GitHub research portfolio, including a PDF file of the paper. To see the final project, open paper.pdf.

This research reviews practical considerations of long-short portfolio optimization problems. It studies a few penalty forms to
address the limitations of pure constraints.

The Alpha Gate penalizes excess turnover above a soft threshold, with a higher hard bound acting as a strict
safeguard. Regularization penalties widely used in regression are adapted to prune noisy trades. Elastic Net and the Berhu
function both use L1 and L_2 penalties on turnovers, but differently: Elastic Net keeps both active at every trade
size, whereas the Berhu Penalty applies L_1 below a threshold and L_2 above it.

We run various experiments with these penalties on ten years of daily long-short equity signals, evaluating turnover control,
target-tracking loss, and trade sparsity across parameter ranges.

```text
├── paper.pdf                     # Final paper
├── bibliography.pdf              # Full bibliography
├── code/                         # Code used to iterate and optimize portfolios, create data files, process data, etc.
│   ├── archived_code/            # Archived code that is not used anymore
│   └── *.py                      # Active code
├── tex/                          # .txt, .tex, .bib files used to generate final .pdf
│   ├── DING_VICTOR_06840.tex     # LaTeX source code
|   ├── references.bib            # bibliography
│   └── *.txt                     # .txt files containing processed data (plotted in graphs)
├── README.md                     # This file
└── .gitignore
```
