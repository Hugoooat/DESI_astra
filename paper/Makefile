LATEX = pdflatex -interaction=nonstopmode
DOCS = article/main.pdf article_fr/main.pdf vulgarisation/vulgarisation.pdf poster/poster.pdf

all: $(DOCS)

%.pdf: %.tex
	cd $(dir $<) && $(LATEX) $(notdir $<) >/dev/null && $(LATEX) $(notdir $<) | grep -E '^!|Output written'

article/main.pdf: article/main.tex article/sections/*.tex article/appendices/*.tex article/bibliography.tex
article_fr/main.pdf: article_fr/main.tex article_fr/sections/*.tex article_fr/appendices/*.tex

clean:
	find . -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' | xargs rm -f
