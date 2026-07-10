# Makefile for Epistemic Honesty SOSP paper

MAIN = epistemic_honest
LATEXMK = latexmk -lualatex

all: 
	$(LATEXMK) $(MAIN)

$(MAIN).pdf:
	$(LATEXMK) $(MAIN)

clean:
	$(LATEXMK) -c $(MAIN)

distclean:
	$(LATEXMK) -C $(MAIN)

watch:
	$(LATEXMK) -pvc $(MAIN)

.PHONY: all clean distclean watch
