---
title: This Date in History - 1964 World's Fair
author: Fraida Fund
---


> Inserting information into a computer from handwritten documents - such as inventory lists, sales slips and scientific laboratory data - has always been one of the slowest steps in automatic information processing. The usual method has been to convert the handwritten data into computer "language" by typing it on a coding machine or punching it on cards. Eventually, machines that can interpret handwriting directly will shorten the time it takes to process information, and will help man take fuller advantage of the electronic speed of computing systems.
>
> _~ Excerpt from ["The IBM Pavilion" booklet](https://www.worldsfairphotos.com/nywf64/booklets/ibm-nywf64-booklet.pdf), available at the 1964 World's fair in New York._

A classic example of a problem that has traditionally been easy for humans, but difficult for computers, is handwritten digit recognition. Because of the many variations in human handwriting, early attempts to "read" handwritten digits into a computer were very limited. 

![These sample digits in MNIST dataset show the variations in human handwriting.](../images/digits.png){ width=70% }

It therefore seemed almost "magical" when, at the 1964 World's Fair in New York, IBM unveiled an amazing computer system that *read handwritten dates* off a small card. Visitors to the "This Date in History" exhibit would write down a date on a card and feed it into the machine. 

![Visitors entering date on a card. Source: [Computer History Museum](https://computerhistory.org/blog/ibm-and-1964-worlds-fair/).](../images/1-ibm-1964-worlds-fair-enter-date-on-card.png){ width=30% }

Then, the computer converted the handwritten date into digital form, look up the date in a database of New York Times headlines, and show a headline from that day on an overhead display. Watch the first minute of [this video](https://www.youtube.com/watch?v=6EGllGYiFa8&t=1s) to see a demonstration!

<div style="text-align:center;"><iframe width="640" height="385" src="https://www.youtube.com/embed/6EGllGYiFa8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>


The results were also printed on a card that you could keep as a souvenir.

![Keepsake card from This Date in History. Source: [Computer History Museum](https://computerhistory.org/blog/ibm-and-1964-worlds-fair/).](../images/1-ibm-1964-worlds-fair-ibm-punch-card-memento-1024x466.jpg){ width=80% }


![Another card from This Date in History. Source: [Queens Public Library](http://digitalarchives.queenslibrary.org/).](../images/1-ibm-sample-card.jpeg){ width=80% }



\newpage


How was this achieved? Was this an early example of a machine learning-based handwritten character recognition system?


The back of the keepsake card explained to visitors how the character recognition system works:

![Back of the keepsake card. Source: [Computer History Museum](https://computerhistory.org/blog/ibm-and-1964-worlds-fair/).](../images/1-ibm-back.png){ width=80% }

While this description has some detail about how the computer acquires the "shape" of the handwritten digit from the card using a cathode ray tube scanner (which generates voltages in proportion to the level of light reflected off the card), it's not very clear how the computer then "recognizes" the shape.

To learn more, we will refer to the work of E. C. Greanias, a researcher at IBM Thomas J. Watson Research Center in New York who specialized in handwritten character recognition, whose work was likely the foundation for this exhibit. In a January 1963 article in the IBM Journal of Research and Development titled ["The Recognition of Handwritten Numerals by Contour Analysis"](https://ieeexplore-ieee-org.proxy.library.nyu.edu/document/5392332) there is a more detailed explanation. Open that paper, and follow along as we dig deeper!

First, there is some discussion of the challenges associated with character recognition, and the desired specifications of a character recognition system:

![Greanias et al desired specifications for character recognition system.](../images/1-handwritten-0.png){ width=60% }

Greanias et al want the system to recognize handwritten digits located anywhere within a larger area, and of various sizes. This seems to rule out a naive pixel-based approach involving pre-defined rules that say (for example): "A 1 should always have writing in the horizontal center of the writing area, and should have no writing on the left and right sides of the writing area," since this would fail to recognize a 1 written on the side of the writing area instead of the center. Greanias et al also wanted the system to tolerate variation in digit shape, minor rotation, or slant. 

The next section of the paper describes in more detail how the image is acquired using the cathode ray tube scanner. This section clarifies that in a first "pass" over the image, the scanner acquires the size and position of the handwritten character, and then in the "second" pass it would acquire a more detailed contour of the handwritten character that is normalized to its position within the writing area and its size.

![This image shows how in the first pass over the writing area, the scanner "finds" the handwritten character and determines its position and size. In the second pass (shown on the bottom), the scanner gets the detailed contour of the handwritten character.](../images/1-handwritten-1.png){ width=30% }

The next section of the paper gets into some detail regarding the electrical circuits used to process the scanner output. We are mainly concerned with the logic, rather than the electronics, so all we need to know is that the area with the handwritten character is divided into 12 "zones" in a 3x4 array, and the circuit determines the direction of the line within each zone (N, NE, E, SE, S, SW, W, or NW).

![Each handwritten character is divided into 12 "zones", and the circuit determines the direction of the line within each zone.](../images/1-handwritten-2.png){ width=30% }

\newpage 

Next, we turn to the character *recognition* part of the system. Greanias et al explain that they had 3000 examples of handwritten digits, and used their _subjective_ judgment to describe _features_ that could be used to identify each type of digit - strokes (straight lines), line ends, lakes (round interior areas such as in 0, 6, or 9), bays (concave areas), etc. in one or more of the 12 "zones".

![Shape "features" for each digit.](../images/1-ibm-shape-features.png){ width=60% }

\newpage


Finally, the paper gives a detailed example with reference to a handwritten 5 example, and shows how an electronic circuit identifies what features are present in an image as the scanner moves along the countours of the handwriting.  In this example, as the scanner moves along the edge of the written digit,

![Detailed example showing recognition of a handwritten 5.](../images/1-recognition-example-5.png){ width=80% }

* First, it looks for either a line moving toward the east in Row 2, or a line moving east or southeast in Row 3. (Condition I, in red.)
* As soon as Condition I is satisfied, the next part of the circuit begins to "look" for a line moving west in Row 4.  (Condition II, in orange.)
* Once Condition II is satisfied, the next part of the circuit looks for a line moving east in Row 4. (Condition III, in green.)
* After this, we look for the west or northwest line in cells B2, C2, B3, C3. (Condition IV, in blue.)
* The next part of the circuit looks for a long east or northeast line in Row 1 or cell B2 (Condition V, in purple.)
* The final part of the circuit looks for a long west or southwest line in Row 1 or cells A2 or B2. (Condition VI, in brown.)
* At this point, we have "found" all of the features that identify a 5. However, these five conditions may also be satisfied by a 3! So, we also check the output of part of the 3 identification circuit, which describes a condition that is present in a 3 but not a 5. If that NOT condition from the 3 recognition circuit is satisfied, then we can conclude that the character is a 5.

\newpage

If this sounds complicated and fragile... it definitely was! For 1964, though, it was impressive. In a press release, IBM brags that the system is extremely flexible, with only minor constraints on the types of handwriting it can read:

> The only constraints imposed by the scanner are that writers must avoid large embellishments such as curlicues and "tails," excessive slanting, large breaks in the lines that compose the number, writing too small or too large beyond a 4-to-1 range in height and running numerals together.
> 
> _~ Excerpt from an [IBM Press Release](https://www.ibm.com/ibm/history/exhibits/specialprod2/specialprod2_pr.html)_


In an experimental evaluation on new samples (see Crook and Kellogg in the reference list), 92% of digits were recognized correctly. However, the subjects could be "trained" to write digits that were more easily recognizable by the machine, bringing the accuracy up to 99.3% for digits written by "trained" writers. This was considered exceptionally good.


## Questions

1. Do you think this was a "machine learning" system? Why or why not?
2. Do you think the performance of this method - 92% accuracy for "untrained" writers, writers must beware of the constraints described in the IBM press release - would be acceptable today? 
3. Can you think of better solutions to any of the challenges described by Greanias et al - for example, that the writing can be located anywhere within a larger area, and that the size of the writing can vary?
4. Do you think the "features" used in this system are adequate? Why or why not? Can you think of counterexamples?


### References

* Dag Spicer, "IBM and the 1964 World's Fair, Computer History Museum, April 2014. [URL](https://computerhistory.org/blog/ibm-and-1964-worlds-fair/)
* E. C. Greanias, P. F. Meagher, R. J. Norman and P. Essinger, "The Recognition of Handwritten Numerals by Contour Analysis," in IBM Journal of Research and Development, vol. 7, no. 1, pp. 14-21, Jan. 1963. [URL for access via NYU Library account](https://ieeexplore-ieee-org.proxy.library.nyu.edu/document/5392332)
* M. N. Crook and D. S. Kellogg, "Experimental Study of Human Factors for a Handwritten Numeral Reader [Letter to the Editor]," in IBM Journal of Research and Development, vol. 7, no. 1, pp. 76-78, Jan. 1963. [URL for access via NYU Library account](https://ieeexplore-ieee-org.proxy.library.nyu.edu/document/5392345)

---

![In 1964, handwritten digit recognition was a cutting edge research problem - so much so that IBM ran an ad featuring the character problem, to try and recruit engineers to their team! This IBM ad is from a 1964 issue of Scientific American. Similar ads were featured in other science and math publications around that time.](../images/1-ibm-i-am-a-5.svg){ width=50% }

