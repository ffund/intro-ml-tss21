
all: in-person.html online.html summer.html

SOURCESIP := title.md schedule-in-person.md materials.md prerequisites.md grading.md integrity.md other-policies.md 
in-person.html: $(SOURCESIP)
	cat $(SOURCESIP) > in-person.tmp
	pandoc -s -t html5 --mathjax -o $@ in-person.tmp -c ../style/pandoc.css
	rm in-person.tmp


SOURCESOL := title.md schedule-online.md materials.md prerequisites.md grading.md integrity.md other-policies.md  

online.html: $(SOURCESOL)
	cat $(SOURCESOL) > online.tmp
	pandoc -s -t html5 --mathjax -o $@ online.tmp -c ../style/pandoc.css
	rm online.tmp

SOURCESAS := title.md schedule-summer.md materials.md prerequisites.md grading-summer.md integrity.md other-policies.md

summer.html: $(SOURCESAS)
	cat $(SOURCESAS) > summer.tmp
	pandoc -s -t html5 --mathjax -o $@ summer.tmp -c ../style/pandoc.css
	rm summer.tmp

clean: 
	rm in-person.html 
	rm online.html
	rm summer.html
