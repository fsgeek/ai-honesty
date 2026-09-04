# PACMI '26 talk. Figures come from ../../papers/pacmi26/figures.
MAIN = epistemic_observability_pacmi26
LATEX = lualatex -interaction=nonstopmode -halt-on-error

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

# Speaker notes on the right half of each page (for a presenter view / recording script)
notes: $(MAIN).tex
	$(LATEX) -jobname=$(MAIN)-notes "\def\shownotes{1}\input{$(MAIN).tex}"
	$(LATEX) -jobname=$(MAIN)-notes "\def\shownotes{1}\input{$(MAIN).tex}"

# One PNG per slide, for looking at them
png: $(MAIN).pdf
	mkdir -p png && pdftoppm -r 60 -png $(MAIN).pdf png/slide

clean:
	rm -f $(MAIN)*.aux $(MAIN)*.log $(MAIN)*.nav $(MAIN)*.out $(MAIN)*.snm $(MAIN)*.toc $(MAIN)*.vrb
	rm -rf png

distclean: clean
	rm -f $(MAIN).pdf $(MAIN)-notes.pdf

.PHONY: all notes png clean distclean
