all:
	pdflatex elsarticle-template-1-num.tex
	bibtex elsarticle-template-1-num
	pdflatex elsarticle-template-1-num.tex
	pdflatex elsarticle-template-1-num.tex
#	mv main.pdf Tesis_Multiobjetivo.pdf
#	latex boockchapter.tex
#	bibtex boockchapter
#	latex boockchapter.tex
#	latex boockchapter.tex
#	dvi2ps -o boockchapter.ps boockchapter.dvi
#	ps2pdf -sPAPERSIZE=letter -DMaxSubsetPct=100 -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress -dSubsetFonts=true -dEmbedAllFonts=true boockchapter.ps CameraReady.pdf
	##pdflatex boockchapter.ps

clean:
	rm -f *.log *.dvi *.aux *.bbl *.blg *~
	
cleanall:
	rm -f *.log *.dvi *.aux *.bbl *.blg *~ main.pdf *.ps *.out
