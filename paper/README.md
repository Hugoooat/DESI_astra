# paper/

| Folder | Content |
|---|---|
| `article/` | English paper: `main.tex` assembles `sections/` (00 abstract ... 10 conclusions), `appendices/` (reproduction, maths, methodological path) and `bibliography.tex` |
| `article_fr/` | French translation, same structure |
| `vulgarisation/` | French plain-language companion: concepts, choices, questions and answers |
| `poster/` | A0 poster (tikzposter) |
| `figures/` | all figures, shared by the documents (produced by `scripts/*.py`, see Appendix A) |
| `archive/` | earlier single-file versions of the paper |

Build everything: `make` (requires `module load texlive/2024` on NERSC).
On Overleaf: upload the whole `paper/` folder (zip), set the main document to
`article/main.tex` (or `article_fr/main.tex`, `vulgarisation/vulgarisation.tex`, `poster/poster.tex`).
