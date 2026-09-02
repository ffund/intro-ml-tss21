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
SOURCES += 12-recommender-systems.md
SOURCES += 13-review.md

# Convert the list of source files (Markdown files )
# into a list of output files

PANDOCFLAGS=--pdf-engine=xelatex\
         -V mainfont='Fira Sans' \
         -V geometry:margin=1in \
         --highlight-style pygments \
	 --listings --variable urlcolor=Maroon --toc \
	 -H ../style/listings-setup.tex -H ../style/keystroke-setup.tex -H ../style/emoji-setup.tex -H ../style/includes.tex

AUDIENCE_FILTER := --lua-filter=audience-filter.lua
AUDIENCE ?= combined

ifeq ($(AUDIENCE),ugrad)
OUTPUT_DIR := ../slides-ugrad
OUTPUT_PREFIX := ../slides-ugrad/
REVEALJS_URL := ../slides/reveal.js-master
else
OUTPUT_DIR := .
OUTPUT_PREFIX :=
REVEALJS_URL := reveal.js-master
endif

PDFS := $(patsubst %.md,$(OUTPUT_PREFIX)%.pdf,$(SOURCES))
SLIDES := $(patsubst %.md,$(OUTPUT_PREFIX)%.html,$(SOURCES))

.DEFAULT_GOAL := all

SLIDEBASE = -t revealjs -V revealjs-url=$(REVEALJS_URL) -s --mathjax \
            -V theme=myplanet --slide-level 3 -V 'slideNumber="c/t"' \
            --lua-filter=slide-colors.lua \
            --include-in-header=slide-setup.html \
            --include-in-header=chalkboard-setup.html

define SLIDE_POST
	sed -i '/dist\/reveal\.js"/a <script src="$(REVEALJS_URL)/plugin/chalkboard/plugin.js"></script>' $(1) && \
	sed -i '/RevealMath,/{h;s/.*/          RevealChalkboard,/;p;x}' $(1) && \
	python3 -c "path='$(1)'; html=open(path).read(); html=html.replace('href=\"reveal.js-master/plugin/chalkboard/style.css\"', 'href=\"$(REVEALJS_URL)/plugin/chalkboard/style.css\"'); html=html.replace('  // reveal.js plugins', '        chalkboard: {\\n          grid: false,\\n          theme: \"whiteboard\",\\n        },\\n\\n  // reveal.js plugins'); open(path, 'w').write(html)" && \
	python3 add-title-poll.py $(1)
endef


$(OUTPUT_PREFIX)%.pdf: %.md audience-filter.lua
	mkdir -p $(dir $@)
	cat $< | sed 's/mathbf/symbf/g' > $<.tmp
	pandoc $<.tmp $(PANDOCFLAGS) $(AUDIENCE_FILTER) -M audience=$(AUDIENCE) -o $@
	rm $<.tmp

$(OUTPUT_PREFIX)%.html: %.md audience-filter.lua slide-colors.lua slide-setup.html chalkboard-setup.html add-title-poll.py ../images/polleverywhere-ffund.png reveal.js-master/dist/theme/myplanet.css reveal.js-master/plugin/chalkboard/plugin.js reveal.js-master/plugin/chalkboard/style.css
	mkdir -p $(dir $@)
	cat $< | sed '/newpage/d' > $<.tmp
	pandoc $(SLIDEBASE) -o $@ $<.tmp $(AUDIENCE_FILTER) -M audience=$(AUDIENCE) && \
	$(call SLIDE_POST,$@)
	rm $<.tmp

ifeq ($(AUDIENCE),ugrad)
.PHONY: $(SOURCES:.md=.pdf) $(SOURCES:.md=.html)
$(SOURCES:.md=.pdf): %.pdf: ../slides-ugrad/%.pdf
$(SOURCES:.md=.html): %.html: ../slides-ugrad/%.html
endif

all: $(PDFS) $(SLIDES)

pdfs: $(PDFS)

slides: $(SLIDES)

clean: 
	rm -f $(OUTPUT_DIR)/*.pdf
	rm -f $(OUTPUT_DIR)/*.html
