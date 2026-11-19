MAIN   = manuscript_oup
LATEX  = pdflatex
BIBTEX = bibtex

.PHONY: all marked clean_pdf clean help

## Default: build marked (tracked-changes) version
all: marked

## Build manuscript_oup.pdf with tracked changes visible (green additions, red strikethrough deletions)
marked: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

## Build manuscript_oup_clean.pdf — no markup, clean accepted text
clean_pdf: $(MAIN).tex
	$(LATEX) -jobname=$(MAIN)_clean "\def\FINAL{}\input{$(MAIN)}"
	$(BIBTEX) $(MAIN)_clean
	$(LATEX) -jobname=$(MAIN)_clean "\def\FINAL{}\input{$(MAIN)}"
	$(LATEX) -jobname=$(MAIN)_clean "\def\FINAL{}\input{$(MAIN)}"

## Remove build artefacts (keeps .tex, .bib, .pdf)
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk *.synctex.gz *.xwm *.loc *.soc

help:
	@echo "make marked    — manuscript_oup.pdf        (tracked changes visible)"
	@echo "make clean_pdf — manuscript_oup_clean.pdf  (clean, no markup)"
	@echo "make clean     — remove build artefacts"
