LATEX       = pdflatex -interaction nonstopmode -halt-on-error -file-line-error
CHECK_RERUN = grep "Rerun to get" $*.log
CRUFT_SUFFS = pdf aux bbl blg log dvi ps eps out brf fls fdb_latexmk synctex.gz
NAME        = products

all: products.pdf

%.pdf: %.tex
	${LATEX} $*
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."

clean:
	${RM} $(foreach suff, ${CRUFT_SUFFS}, ${NAME}.${suff})
	${RM} *Notes.bib