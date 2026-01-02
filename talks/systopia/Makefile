# Makefile for Epistemic Honesty Talk
# Systopia Research Group, January 2025

MAIN = epistemic_honesty_talk
LATEX = pdflatex
BIBTEX = bibtex

# Default target
all: $(MAIN).pdf

# Build PDF
$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Build with notes (for presenter view)
notes: $(MAIN).tex
	@echo "Enabling speaker notes..."
	@sed -i 's/% \\setbeameroption/\\setbeameroption/' $(MAIN).tex
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	@sed -i 's/\\setbeameroption/% \\setbeameroption/' $(MAIN).tex
	@echo "Notes version built. Remember: notes are now disabled in source."

# Quick build (single pass)
quick: $(MAIN).tex
	$(LATEX) $(MAIN)

# Watch for changes (requires inotifywait)
watch:
	@echo "Watching for changes to $(MAIN).tex..."
	@while true; do \
		inotifywait -q -e modify $(MAIN).tex; \
		$(LATEX) $(MAIN); \
	done

# Clean auxiliary files
clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).nav $(MAIN).out \
	      $(MAIN).snm $(MAIN).toc $(MAIN).vrb

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf

# View PDF (Linux)
view: $(MAIN).pdf
	xdg-open $(MAIN).pdf &

.PHONY: all notes quick watch clean distclean view
