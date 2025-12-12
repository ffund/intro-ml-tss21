# All markdown files are considered sources
SOURCES := $(wildcard *.md)
SOURCES := 1-intro-ml.md
SOURCES += 1-explore-data.md
SOURCES += 1-working-with-data.md
SOURCES += 2-linear.md
SOURCES += 3-gradient-descent.md
SOURCES += 3-bias-variance-tradeoff.md
SOURCES += 4-model-order.md
SOURCES += 4-regularization.md
SOURCES += 5-logistic-regression.md
SOURCES += 6-knn.md
SOURCES += 6-feature.md
SOURCES += 6-tree.md
SOURCES += 7-ensemble.md
SOURCES += 7-svm.md
SOURCES += 8-svm-kernel.md
SOURCES += 8-neural.md
SOURCES += 8-hyperparameter.md
SOURCES += 9-deep-neural-nets.md
SOURCES += 9-convolutional.md
SOURCES += 10-unsupervised.md
SOURCES += 10-rl.md
SOURCES += 10-deployment.md
SOURCES += 13-review.md

# Convert the list of source files (Markdown files )
# into a list of output files
PDFS := $(patsubst %.md,%.pdf,$(SOURCES))
SLIDES := $(patsubst %.md,%.html,$(SOURCES))

PANDOCFLAGS=--pdf-engine=xelatex\
         -V mainfont='Fira Sans' \
         -V geometry:margin=1in \
         --highlight-style pygments \
	 --listings --variable urlcolor=Maroon --toc \
	 -H ../style/listings-setup.tex -H ../style/keystroke-setup.tex -H ../style/emoji-setup.tex -H ../style/includes.tex

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

