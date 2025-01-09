
all: in-person.html online.html

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

clean: 
	rm in-person.html 
	rm online.html
