# research-asset-ding
GitHub research portfolio, including a PDF file of the paper. To see the final project, open paper.pdf.

This project utilizes Python solvers to optimize portfolios subjected to constraints and penalties.
It establishes a background in portfolio optimization, then later implements three alternative
approaches: the Alpha Gate, Elastic Net, and Berhu function. These are penalties that soften the
aforementioned hard constraints. For example, the Alpha Gate allows the solver to exceed a strict
total turnover (the sum of all individual turnovers), but simultaneously discourages it by
incurring a penalty on the objective function. To assess the impacts and validity of these models,
we iterated through ten of portfolios several times, using different models. Throughout the paper,
we consider several strategies to make these programs more efficient.

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
