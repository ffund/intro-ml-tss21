---
title:  'Exploring your data'
author: 'Fraida Fund'
---

## Garbage in, garbage out

::: notes

If you use "garbage" to train a machine learning model, you will only get "garbage" out. (+ Since you are testing on the same data, you might not even realize it is "garbage" until the model is in production!)

:::


## Exploratory data analysis: what are we looking for?

* Make and check assumptions
* Check for missing data
* Identify potentially predictive features
* Look for patterns you *don't* want model to learn
* Ethics concerns



\newpage

## Make and check assumptions

### Example: author citation data (1)

Data analysis: use PubMed, and identify the year of first publication for the 100,000 most cited authors.

::: notes

What are our expectations about what this should look like?

:::

### Example: author citation data (2)

![Does this look reasonable?](../images/1-pubmed-authors.png){ width=50% }

::: notes

We can think of many potential explanations for this pattern, even though it is actually a data artifact.

The true explanation: in 2002, PubMed started using full first names in authors instead of just initials. The same author is represented in the dataset as a "new" author with a first date of publication in 2002.

:::

### Example: author citation data (3)

![The real distribution, after name unification. Example via [Steven Skiena @ SBU](https://www3.cs.stonybrook.edu/~skiena/519/).](../images/1-pubmed-authors2.png){ width=50% }

\newpage


### Example: anomalous voting data (1)

![Data like this was widely (wrongly) used as evidence of anomaly in the 2020 U.S. Presidential election.](../images/1-election2020.png){ width=30% }

::: notes

What are our assumptions about election night data, and how are they violated here? 

We expect that per-candidate vote totals (computed by multiplying total votes and vote share) should increase as more votes are counted, but never decrease.

What are possible explanations?

:::

### Example: anomalous voting data (2)

![Process by which data is collected by Edison and AP.](../images/1-election2020-process.png){ width=75% }

::: notes

This anomaly makes a lot of sense as a correction of a data entry or duplicate entry error. 

How Edison/AP collects the data for their Election Night feed:

* There are "stringers" (temporary reporters) at various elections offices who call results into their phone center
* They have people who look at official government websites for new results that they manually enter into the system
* They have people who monitor results sent by fax from counties and cities

all working as fast as they can! Data entry and duplicate entry errors are not only likely, they are almost guaranteed. When they are corrected, vote totals may decrease.

Source: [AP](https://web.archive.org/web/20210410214207/https://www.ap.org/en-us/topics/politics/elections/counting-the-vote), [Edison](http://www.edisonresearch.com/wp-content/uploads/2020/10/Web-Entry-Team-Handout-2020.pdf)

:::


::: {.cell .markdown}

### Handling unreasonable data

:::notes


How should you handle unreasonable values or outliers?

* e.g. suppose in a dataset of voter information, some have impossible year of birth - would make the voter a child, or some indicate the voter is 120 years old. (Voters with no known DOB, who registered before DOB was required, are often encoded with a January 1900 DOB.)
* **not** a good idea to just remove outliers unless you are sure they are a data entry error or otherwise not a "true" value.
* Even if an outlier is due to some sort of error, if you remove them, you may skew the dataset (as in the 1/1/1900 voters example).

Consider the possibility of: 

* Different units, time zones, etc. in different rows
* Same value represented several different ways (e.g. names, dates)
* Missing data encoded as zero

:::

:::



## Missing data



::: {.cell .markdown}

### Examples of missing data

:::notes

* Twitter API terms of use don't allow researchers to share tweets directly, only message IDs (except for limited distribution, e.g. by email). To reproduce the dataset, you use the Twitter API to download messages using their IDs. But, tweets that have been removed are not available - the distribution of removed tweets is not flat! (For example: you might end up with a dataset that has offensive posts but few "obvious" offensive posts.)
* A dataset of Tweets following Hurricane Sandy makes it looks like Manhattan was the hub of the disaster, because of power blackouts and limited cell service in the most affected areas. [Source](https://hbr.org/2013/04/the-hidden-biases-in-big-data)
* The City of Boston released a smartphone app that uses accelerometer and GPS data to detect potholes and report them automatically. But, low income and older residents are less likely to have smartphones, so this dataset presents a skewed view of where potholes are. [Source](https://hbr.org/2013/04/the-hidden-biases-in-big-data)


:::

:::


::: {.cell .markdown}

### Indications of missing data

* Rows that have `NaN` values
* Rows that are *not there*

:::

::: {.cell .markdown}

### Types of "missingness"

* Completely random
* Correlated with something that is in data
* Correlated with something not in data

:::notes

These are often referred to using this standard terminology (which can be confusing):

* Missing _completely_ at random: missingness not correlated with any feature or to the target variable.
* Missing at random: missingness correlated with something that is in data.
* Missing not at random: missingness correlated with something that is not in data.

:::
:::

\newpage 

::: {.cell .markdown}

### Handling missing data

How should you handle little bits of missing data? It always depends on the data and the circumstances. Some possibilities include:

* omit the row
* fill with mean, median, max, mode...
* fill back/forward (ordered rows)
* train a model on the rest of the data to "predict" the missing value


:::notes

For example: suppose we are training a model on Intro ML students' score and duration on HW questions, to predict their score on related exam questions.

:::

:::

## Predictive features



::: {.cell .markdown}

### How do we look for predictive features?

* Numeric (continuous) features
* Categorical features
* Graphical features
* Text features

:::


## "Bad patterns" (and data leakage)

### COVID-19 chest radiography (1)

* **Problem**: diagnose COVID-19 from chest radiography images
* **Input**: image of chest X-ray (or other radiography)
* **Target variable**: COVID or no COVID

### COVID-19 chest radiography (2)

![Neural networks can classify the source dataset of these chest X-ray images, even *without lungs*! [Source](https://arxiv.org/abs/2004.12823)](../images/1-covid-xrays.png){ width=60% }


::: notes

Between January and October 2020, more than 2000 papers were published that claimed to use machine learning to diagnose COVID-19 patients based on chest X-rays or other radiography. But a later [review](https://www.nature.com/articles/s42256-021-00307-0) found that "none of the models identified are of potential clinical use due to methodological flaws and/or underlying biases".

To train these models, people used an emerging COVID-19 chest X-ray dataset, along with one or more existing chest X-ray dataset, for example a pre-existing dataset used to try and classify viral vs. bacterial pneumonia.

The problem is that the chest X-rays for each dataset were so "distinctive" to that dataset, that a neural network could be trained with high accuracy to classify an image into its source dataset, even without the lungs showing!

:::

### COVID-19 chest radiography (2)

Findings:

* some non-COVID datasets were pediatric images, COVID images were adult
* there were dataset-level differences in patient positioning
* many COVID images came from screenshots of published papers, which often had text, arrows, or other annotations over the images. (Some non-COVID images did, too.)

### COVID-19 chest radiography (3)

![Saliency map showing the "important" pixels for classification. [Source](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7523163/)](../images/1-covid-xrays-saliency.png){ width=90% }

::: notes

These findings are based on techniques like 

* saliency maps, where the model is made to highlight the part of the image (the pixels) that it considered most relevant to its decision.
* using generative models and asking it to take a COVID-negative X-ray and make it positive (or v.v.)

Many of the findings are not easy to understand without domain knowledge (e.g. knowing what part of the X-ray *should* be important and what part should not be.) For example: should the diaphragm area be helpful?

:::



### Data leakage

::: notes

In machine learning, we train models on a training set of data, then evaluate their performance on a set of data that was not used in training.

Sometimes, information from the training set can "leak" into the evaluation - this is called data leakage.

Or, information from the target variable (which should not be available during inference) leaks into the feature data.

:::

### Some types of data leakage

* Learning from a feature that is a proxy for target variable, but that doesn't generalize 
* Learning from adjacent temporal data
* Learning from duplicate data
* Learning from features that are not available at prediction time (e.g. data from the future)


\newpage

### Signs of potential data leakage (after training)

* Performance is "too good to be true"
* Unexpected behavior of model (e.g. learns from a feature that shouldn't help)

### Detecting data leakage

* Exploratory data analysis
* Study the data before, during, and after you use it!
* Explainable ML methods
* Early testing in production




## Ethics concerns


::: {.cell .markdown}
### Some types of data ethics fails

* Bias
* Privacy
* Consent


:::notes

* Many social media datasets used for "offensive post" classification have biased labels (especially if they were produced without adequate training procedures in place). For example, they may label posts containing African-American dialects of English as "offensive" much more often. [Source](https://www.aclweb.org/anthology/P19-1163.pdf), [User-friendly article](https://www.vox.com/recode/2019/8/15/20806384/social-media-hate-speech-bias-black-african-american-facebook-twitter)
* [On the anonymity of the Facebook dataset](http://www.michaelzimmer.org/2008/09/30/on-the-anonymity-of-the-facebook-dataset/)
* [70,000 OkCupid Users Just Had Their Data Published](https://www.vice.com/en_us/art*cle/8q88nx/70000-okcupid-users-just-had-their-data-published); [OkCupid Study Reveals the Perils of Big-Data Science](https://www.wired.com/2016/05/*kcupid-study-reveals-perils-big-data-science/); [Ethics, scientific consent and OKCupid](https://ironholds.org/scientific-consent/)
* [IBM didn’t inform people when it used their Flickr photos for facial recognition training](https://www.theverge.com/2019/3/12/18262646/ibm-didnt-inform-people-when-it-used-their-flickr-photos-for-facial-recognition-training)

:::

:::



## Many more data problems...


::: notes

* Data or labels reflect human bias
* Data is not representative of your target situation
* Data or situation changes over time

Examples:


Change over time: Imagine you train a machine learning model to classify loan applications. However, if the economy changes, applicants that were previously considered credit-worthy might not be anymore despite having the same income, as the lender becomes more risk-averse. Similarly, if wages increase across the board, the income standard for a loan would increase.

:::

\newpage


