FILES=0-tss-faq.pdf 0-prerequisite-review.pdf 0-books.pdf 
FILES+=1-handwritten-digits.pdf
FILES+=3-cubic-model.pdf

SOURCES := 0-tss-faq.md 0-prerequisite-review.md 0-books.md
SOURCES += 1-handwritten-digits.md

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

