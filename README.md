# Hidden Markov Models

## Coding Notes

* The states on Pomegranate, represented as strings _s<sub>i</sub>_ are mapped to the input states in an alphabetical order, e.g. `['bbb', 'aaa', 'ccc']` means: _s<sub>0</sub>='aaa', s<sub>1</sub>='bbb', s<sub>2</sub>='ccc', s<sub>3</sub>='None-start', s<sub>4</sub>='None-end'_

```python
  ...
  normal_states = list(sorted(normal_states, key=attrgetter('name')))
  silent_states = list(sorted(silent_states, key=attrgetter('name')))
  ...
```

<br><br/>

## Method A

Even if it looks like it, it is not really a classification method.
![Method A](Documentation%20Images/General%20HMM%20Method%201.png?raw=true)

## Method B

Pure classification.
![Method B](Documentation%20Images/General%20HMM%20Method%202.png?raw=true)

<br><br/>

## List of Models with emphasis on Text Classification

(number) *Name* [on what type of sequential data it works on] [difficulty] [method A/B] [is it more or less ready] [references relevant to NLP]

(1) **State-emission HMM** [sentence-based] [★] [yes] [A] [[Manning et al.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.2604&rep=rep1&type=pdf)] [[Mathew](https://www.slideshare.net/thomas_a_mathew/text-categorization-using-ngrams-and-hiddenmarkovmodels)] : An extremely basic form of HMM, used when we have no notion of class/category sequence. Definition: If the symbol emitted at time _t_ depends on both the state at time _t_ and at time _t+1_ we have what is sometimes called an arc-emission HMM. However, an alternative formulation is a state-emission HMM, where the symbol emitted at time _t_ depends just on the state at time _t_. For example, imagine a crazy soft drink machine that prefers to output a certain drink and after each output changes state randomly.

* General Mixture Model which is not a HMM - states: constant document labels, observations: sentence polarity labels
* State-emission HMM - states: sentence polarity labels, observations: sentence polarity labels

(2) **Classic HMM** [any-based] [★] [yes] [A] [[Rabiner](https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)] : The well-known sequential model used for Part-of-Speech tagging, Biology-related tasks, Image recognition (pixels as a sequence) etc.

(3) **Multivariate HMM** [mostly sentence-based] [TO ADD] [ ] [ ] [[Liu et al.](https://www.hindawi.com/journals/mpe/2015/987189/)] [[Quan et al.](https://www.sciencedirect.com/science/article/pii/S0020025515007057)] [[Tune et al.](https://arxiv.org/pdf/1305.0321.pdf)] [[Li et al.](http://vision.gel.ulaval.ca/~parizeau/Publications/P971225.pdf)] : Lifts the restriction of a single observation per time state. As one would imagine this is very useful for Natural Language Processing tasks, since we have a bunch of text per observation and not a time series or a DNA sequence.

Can be either Continuous (Method A, new formula would need to be invented for Method B) or Discrete (Method A/B) and we are also restricted to mostly sentence-based tasks.

* Example of Continuous: pass multiple tfidf values for each sentence, where we have a sequence of sentences. (too many variables)
* Example of Discrete: pass multiple words on each sentence (too many variables); a very good idea would be to pass both the word and its Part-of-Speech tag on a Spyros HMM.  

* Liu et al. - states: crazy Particle Swarm Opt. to find states; 4 hidden states, observations: 4 features of a document such as tfidf etc. (where is the sequence?)
* Other ideas - states: polarity, observations: multiple tfidf values of document (where is the sequence?)

(4) **High-order HMM** [any-based] [★★★★★] [yes] [B] [[Quan et al.](https://www.sciencedirect.com/science/article/pii/S0020025515007057)] [[Preez](https://www.sciencedirect.com/science/article/pii/S0885230897900371)] [[Kochanski](http://kochanski.org/gpk/teaching/0401Oxford/HMM.pdf)] [[Ching et al.](https://link.springer.com/chapter/10.1007/978-3-540-45080-1_73)] [[Lee et al.](http://link-springer-com-s.vpn.whu.edu.cn:9440/content/pdf/10.1007%2F11779568_74.pdf)] : Lifts a major restriction of HMMs and allows the states to also depend on the observation/state preceding the directly previous one. The implementation might work through Kochanski's transformation, Preez's transformation and [miniHMM's](https://github.com/joshuagryphon/minihmm/blob/master/minihmm/represent.py) dummy states and initial probabilities.  

* Quan et al. (+ Multivariate) - states: emotions, observations: custom encoding with 150 possible observations per state

(5) **Clustering then HMM** [any-based] [★★★★★] [yes] [B] [[Kang et al.](https://www.sciencedirect.com/science/article/pii/S0957417417304979)]  

* Kang et al. - states: clusters, obervations: words - We can either (1) form some sort of clustering on the SVD term-to-term dot matrix like Kang did, (2) form a clustering of the documents and then predict on each word, (3) form a clustering on word2vec (better not https://datascience.stackexchange.com/questions/30917/k-means-clustering-of-word-embedding-gives-strange-results).
* Other ideas - Combine (5) and (6) or something, states: labels, observations: clusters

(6) **Spyros HMM** [any-based] [★★★] [yes] [B] [My Mind] : We want to perform classification but in text-related tasks we have a single label instead of a sequence of labels. In order to tackle this, take any Machine Learning classifier or tool, train it on the data and then have it predict the class label of every single word - we create artificial labels. Even though we are performing predictions on the data it was trained on, the resulting labels are VERY noisy since we are doing it on a single-word basis (e.g. get me the sentiment label of "movie" and get me the label of "review"). Then, the proposed HMM is trained on the artificial data and actually performs better than the original classifiers. For even higher performance, have 10 state-of-the-art classifiers and tools (LSTM, lexicon etc.)  predict the (artificial) label of each word and perform a majority vote.

* Kang et al. - states: clusters, obervations: words

(7) **Bidirectional HMM** [any-based] [ ] [ ] [ ] [[Zacher et al.](http://msb.embopress.org/content/msb/10/12/768.full.pdf)] [[Arani et al.](https://digital-library.theiet.org/content/journals/10.1049/iet-cvi.2017.0645)]  

(8) **Hierarchical HMM** [TO ADD] [ ] [ ] [ ] [[Fine et al.](https://link.springer.com/content/pdf/10.1023/A:1007469218079.pdf)]  

(9) The remaining HMM models that alter assumptions about time etc. (e.g. Semi-Markov)

<br><br/>

# Experiments

## Conclusions

(all) Utilizing n-grams on the observations heavily increases accuracy, see console logs and graphs.  
(all) Using a higher n-gram means more new unseen transitions, see graphs.  
(all) Adding the smoothing factor doesn't affect performance that much, at least on Pomegranate.  
(1) [Experimental Results of State-emission HMM.txt](Console%20Logs/Experimental%20Results%20of%20State-emission%20HMM.txt)  
(2) Can be run at any time.  
(3) Didn't work  
(4) [Experimental Results of State-emission HMM.txt](Console%20Logs/Experimental%20Ressults%20of%20State-emission%20HMM.txt)  
(5) Idea 2 didn't work, idea 3 didn't work [Experimental Results of Clustered HMM.txt](Console%20Logs/Experimental%20Results%20of%20Clustered%20HMM.txt).  
(6) A HMM can increase the performance of any bag-of-words-based Machine Learning classifier or tool by utilizing the sequential information of text. This is done by producing artificial labels. [Experimental Results of State-emission HMM.txt](Console%20Logs/Experimental%20Results%20of%20State-emission%20HMM.txt) and [Experimental Results on Big Dataset.txt](Console%20Logs/Experimental%20Results%20on%20Big%20Dataset.txt)  

<br><br/>

# To Do

* Fix general HMM code to be mapped on the states _s<sub>i</sub>_ depending on alphabetical order. However, I already have a smart implementation on `Finegrained.py` by using state names as labels instead of "pos"/"neg"/"neu".
* On the IMDb artificial labels HMM, try to train it only on emotional words instead of all the words while using a state-of-the-art lexicon/tool/whatever.
* Switch code to object-oriented, where the class has a function to plot results (e.g. functions such as 'compare_length', 'print_console_to_text', 'plot_results').
* Look for ideal smoothing factor.
* Look for ideal ensemble of HMMs such as dor product instead of average. 
* Bidrectional HMM.
* See HMM (3) discrete.  
* Dataframe sample function should take random_state as a parameter.
* Take a look at the weighting function from Quan and Ren

## Counting

* Showcase how as the order increases we have transitions that are too low probability; the problematic count will help with this.
* Showcase how as the order increases we have more empty sequences; `['neg', 'neg-pos', 'neg-pos-pos']` is an empty sequence since HOHMM doesn't use dummy states.
* Note the new unseen observation count which stays constant; the unseen count will help with this.

## Known Issues

* Training on the "neg" subset of the IMDb dataset on Pomegranate completely bugs when using emission_pseudocount or higher-order represented as first-order; possibly semi-supervised learning gets enabled; sequence are slightly longer than "pos" ones. Temporary fix is to shorten the "neg" sequences by 1.

<br><br/>

## List of other Ideas

* Use Naive Bayes since they only have sentence sentiment to classify each word and that is the state - https://github.com/saumyakb/CS4740-NLP-Sentiment-Analysis-using-HMM - states: naive bayes artificial labels, observations: words

* Each word is one-hot encoded and used as observations; since it is built like a classifier, clueless people run it with (n_sequences, seq_length=1) by throwing a tfidf matrix as the sequence of length 1 and it kind of works - https://github.com/larsmans/seqlearn/blob/master/seqlearn/hmm.py - states: Part-of-Speech tags, observations: one-hot encoded words
