SOURCES := 0-online-faq.md 0-tss-faq.md 0-prerequisite-review.md 0-books.md
SOURCES += 1-handwritten-digits.md
SOURCES += 2-gd-background-videos.md
SOURCES += 2-beauty-classroom-case-study.md
SOURCES += 2-linear-extended-derivation.md
SOURCES += 3-cubic-model.md
SOURCES += 3-linear-regression-bias-variance.md
SOURCES += 4-classifier-metrics.md

#SOURCES += 3-when-learning-fails.md

PDFS := $(patsubst %.md,%.pdf,$(SOURCES))
PAGES:=  $(patsubst %.md, %.html,$(SOURCES))

PANDOCFLAGS= --pdf-engine=xelatex\
         -V mainfont='Fira Sans' \
         -V geometry:margin=1in \
	 -V urlcolor=cyan \
         --listings -H ../style/listings-setup.tex -H ../style/keystroke-setup.tex -H ../style/includes.tex

%.pdf: %.md
	cat $^ | sed 's/mathbf/symbf/g' > $^.tmp
	pandoc $^.tmp $(PANDOCFLAGS) -o $@ 
	rm $^.tmp

%.html: %.md
	cat $^ | sed '/newpage/d' > $^.tmp
	pandoc -s -t html5 --mathjax -o $@ $^.tmp -c ../style/pandoc.css
	rm $^.tmp

all: $(PDFS) $(PAGES)

clean: 
	rm -f *.pdf *.html

