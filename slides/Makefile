# All markdown files are considered sources
SOURCES := $(wildcard *.md)
SOURCES := 1-intro-ml.md
SOURCES += 1-explore-data.md
SOURCES += 2-linear.md
SOURCES += 3-bias-variance-tradeoff.md
SOURCES += 3-model-order.md
SOURCES += 3-regularization.md
SOURCES += 4-logistic-regression.md

# Convert the list of source files (Markdown files )
# into a list of output files
PDFS := $(patsubst %.md,%.pdf,$(SOURCES))
SLIDES := $(patsubst %.md,%.html,$(SOURCES))

PANDOCFLAGS=--pdf-engine=xelatex\
         -V mainfont='Fira Sans' \
         -V geometry:margin=1in \
         --highlight-style pygments \
	 --listings --variable urlcolor=Maroon --toc \
	 -H ../style/listings-setup.tex -H ../style/keystroke-setup.tex -H ../style/includes.tex

%.pdf: %.md
	cat $^ | sed 's/mathbf/symbf/g' > $^.tmp
	pandoc $^.tmp $(PANDOCFLAGS) -o $@ 
	rm $^.tmp

%.html: %.md
	cat $^ | sed '/newpage/d' > $^.tmp
	pandoc -t revealjs -V revealjs-url=reveal.js-master -s --mathjax -o $@ $^.tmp -V theme=white --slide-level 3
	rm $^.tmp

all: $(PDFS) $(SLIDES)

pdfs: $(PDFS)

slides: $(SLIDES)

clean: 
	rm -f *.pdf
	rm -f *.html

