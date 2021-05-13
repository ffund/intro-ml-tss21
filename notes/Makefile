FILES=tss-faq.pdf

PANDOCFLAGS= --pdf-engine=xelatex\
         -V mainfont='Fira Sans' \
         -V geometry:margin=1in \
	 -V urlcolor=cyan \
         --listings -H ../style/listings-setup.tex -H ../style/keystroke-setup.tex -H ../style/includes.tex

%.pdf: %.md
	pandoc $^ $(PANDOCFLAGS) -o $@ 

all: $(FILES)

clean: 
	rm -f *.pdf

