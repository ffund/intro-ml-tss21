FILES+=3-cubic-model.pdf
FILES+=3-when-learning-fails.md

SOURCES := 0-tss-faq.md 0-prerequisite-review.md 0-books.md
SOURCES += 1-handwritten-digits.md
SOURCES += 2-gd-background-videos.md
SOURCES += 2-beauty-classroom-case-study.md
SOURCES += 3-cubic-model.md
SOURCES += 3-when-learning-fails.md

PDFS := $(patsubst %.md,%.pdf,$(SOURCES))
PAGES:=  $(patsubst %.md, %.html,$(SOURCES))

PANDOCFLAGS= --pdf-engine=xelatex\
         -V mainfont='Fira Sans' \
         -V geometry:margin=1in \
	 -V urlcolor=cyan \
         --listings -H ../style/listings-setup.tex -H ../style/keystroke-setup.tex -H ../style/includes.tex

%.pdf: %.md
	pandoc $^ $(PANDOCFLAGS) -o $@ 

%.html: %.md
	cat $^ | sed '/newpage/d' > $^.tmp
	pandoc -s -t html5 -o $@ $^.tmp -c ../style/pandoc.css
	rm $^.tmp

all: $(PDFS) $(PAGES)

clean: 
	rm -f *.pdf *.html

