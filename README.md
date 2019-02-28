# Hidden Markov Models

## Coding Notes

* The states on pomegranate, represented as strings _s<sub>i</sub>_ are mapped to the input states in an alphabetical order, e.g. `['bbb', 'aaa', 'ccc']` means: _s<sub>0</sub>='aaa', s<sub>1</sub>='bbb', s<sub>2</sub>='ccc'_

```python
  ...
  normal_states = list(sorted(normal_states, key=attrgetter('name')))
  silent_states = list(sorted(silent_states, key=attrgetter('name')))
  ...
```

![alt text](http://url/to/img.png)



## List of Models with emphasis on Text Classification

(number) [on what type of sequential data it works on] [difficulty] [is it more or less ready] [references relevant to NLP]

(1) [any-based] [★] [yes] [[Manning et al.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.2604&rep=rep1&type=pdf)] [[Mathew](https://www.slideshare.net/thomas_a_mathew/text-categorization-using-ngrams-and-hiddenmarkovmodels)] **State-emission HMM**: An extremely basic form of HMM, used when we have no notion of class/category sequence. Definition: If the symbol emitted at time _t_ depends on both the state at time _t_ and at time _t+1_ we have what is sometimes called an arc-emission HMM. However, an alternative formulation is a state-emission HMM, where the symbol emitted at time _t_ depends just on the state at time _t_. For example, imagine a crazy soft drink machine that prefers to output a certain drink and after each output changes state randomly.

(2) [any-based] [★] [yes]  **Classic HMM**: The well-known sequential model used for Part-of-Speech tagging, Biology-related tasks, Image recognition (pixels as a sequence) etc.

(3) [[Liu et al.](https://www.hindawi.com/journals/mpe/2015/987189/)] [[Quan et al.](https://www.sciencedirect.com/science/article/pii/S0020025515007057)] [[Tune et al.](https://arxiv.org/pdf/1305.0321.pdf)] [[Li et al.](http://vision.gel.ulaval.ca/~parizeau/Publications/P971225.pdf)] **Multivariate HMM**: Lifts the restriction of a single observation per time state. As one would imagine this is very useful for Natural Language Processing tasks, since we have a bunch of text per observation and not a time series or a DNA sequence.

* Liu et al. - states: crazy Particle Swarm Opt. to find states; 4 hidden states, observations: 4 features such as tfidf etc.

(4) [any-based] [★★★★★] [yes] [[Quan et al.](https://www.sciencedirect.com/science/article/pii/S0020025515007057)] [[Preez](https://www.sciencedirect.com/science/article/pii/S0885230897900371)] **High-order HMM**: Lifts a major restriction of HMMs and allows the states to also depend on the observation/state preceding the directly previous one.

* Quan et al. (+ Multivariate) - states: emotions, observations: custom encoding with 150 possible observations per state

(5) [[Kang et al.](https://www.sciencedirect.com/science/article/pii/S0957417417304979)]    Clustering then HMM

* Kang et al. - states: clusters, obervations: words

(6) [any-based] [[Zacher et al.](http://msb.embopress.org/content/msb/10/12/768.full.pdf)] [[Arani et al.](https://digital-library.theiet.org/content/journals/10.1049/iet-cvi.2017.0645)] **Bidirectional HMM**

(7) [ ] [ ] [ ] [[Fine et al.](https://link.springer.com/content/pdf/10.1023/A:1007469218079.pdf)] **Hierarchical HMM** 

(8) The remaining HMM models that alter assumptions about time etc. (e.g. Semi-Markov)

<br><br/>

## To Do

* Fix general HMM code to be mapped on the states _s<sub>i</sub>_ depending on alphabetical order

### Counting

* Showcase how as the order increases we have transitions that are too low probability; the problematic count will help with this.
* Showcase how as the order increases we have more empty sequences; `['neg', 'neg-pos', 'neg-pos-pos']` is an empty sequence since HOHMM doesn't use dummy states.
* Note the new unseen observation count which stays constant; the unseen count will help with this.
<br><br/>

## Conclusions

* A HMM can increase the performance of any ... by utilizing the sequential information of text
<br><br/>

## List of other people's Implementations

* Uses Naive Bayes since they only have sentence sentiment to classify each word and that is the state - https://github.com/saumyakb/CS4740-NLP-Sentiment-Analysis-using-HMM - states: naive bayes artificial labels, observations: words

* Each word is one-hot encoded and used as observations; since it is built like a classifier, clueless people run it with (n_sequences, seq_length=1) by throwing a tfidf matrix as the sequence of length 1 and it kind of works - https://github.com/larsmans/seqlearn/blob/master/seqlearn/hmm.py - states: Part-of-Speech tags, observations: one-hot encoded words