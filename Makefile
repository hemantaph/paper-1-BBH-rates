paper.pdf: paper.tex
	latexmk -pdf $<

clean:
	latexmk -C
