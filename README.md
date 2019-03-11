# Hidden Markov Models

## Coding Notes - Overall

* On architecture B, if we require exact probabilities per class we have to normalize the result of the math formula. [Kang et al.](https://www.sciencedirect.com/science/article/pii/S0957417417304979) divide the multiplied probability score by the length of the sequence.

## Coding Notes - Pomegranate

* The states in Pomegranate, represented as strings _s<sub>i</sub>_ are mapped to the input state labels in an alphabetical order, e.g. `['bbb', 'aaa', 'ccc']` means: _s<sub>0</sub>='aaa', s<sub>1</sub>='bbb', s<sub>2</sub>='ccc', s<sub>3</sub>='None-start', s<sub>4</sub>='None-end'_.

```python
  ...
  normal_states = list(sorted(normal_states, key=attrgetter('name')))
  silent_states = list(sorted(silent_states, key=attrgetter('name')))
  ...
```

For anything that isn't algorithm='labeled': just use the state_names parameter to avoid some bugs, e.g.1. state_labels:`["bbb", "aaa", "bbb"]` then state_names=`["aaa", "bbb"]`. EVEN better, convert all state labels to s<sub>0</sub>,s<sub>1</sub>...s<sub>n</sub>, e.g.2. state_labels:`["s2", "s1", "s0"]` then state_names=`["s0", "s1", "s2"]`.

For algorithm='labeled': extremely bad implementation riddled with bugs, see [.py](/Other%20Concepts/Weird%20Labeled%20training%20Bug.py). The '_labeled_summarize' function is a mess.  
Observation Probabilities Bug: Instead of taking into consideration the current observation, it takes the previous observation for each time step. In detail: (1) if the label names are "s0", "s1" etc. even if 'state_names' parameter is not used, the bug happens, (2) if the label names are anything else and 'state_names' is used to assist the function, the bug happens, (3) however if the label names are anything alse and 'state_names' is not used, the bug doesn't happen. The bug also doesn't happen in the opposite scenario, where "s0" etc. are used, but irelevant 'state_names' are also given.  
State transition Probabilities Bug: All the transitions end up being the exact same (the initial distribution). This bug happens when the previous one doesn't; it happens on (3) and doesn't on (1) and (2).  
Initial transition Probabilies Bug: All the transitions end up being the exact same (the initial distribution).  

At first I thought our best bet was (3), but we stil have the last 2 bugs. Just don't use "labeled" and use "baum-welch" with 'max_iterations' set to 1.

```python
  ...
  for i in range(label_ndarray.shape[0]):
      if isinstance(label_ndarray[i], State):
          labels[i] = self.states.index(label_ndarray[i])
      else:
          labels[i] = self.state_name_mapping[label_ndarray[i]]
  ...
```

* The emission-pseudocount does not work for algorithm='labeled'.

* The emission-pseudocount is not added to states that only occur at the start of sequences, e.g. observations:`[["234", "123", "234"], ["651", "1"]]` and state_labels:`[["s234", "s123", "s234"], ["s651", "s1"]]` means that state651 will have probability of 1 for 651 and 0 for everything else.

## Coding Notes - Matlab

[Function to be used](https://www.mathworks.com/help/stats/hmmestimate.html)  
[Install the Matlab Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)  
Path of execution: /usr/local/MATLAB/R2018b/extern/engines/python  
Path of installation: ~/anaconda3/envs/matlabCompatiblePython/lib/python3.6/site-packages  
Success with: sudo /home/s/anaconda3/envs/matlabCompatiblePython/bin/python setup.py install  
(no need for --prefix="installdir" parameter)  

[Passing data to Matlab](https://www.mathworks.com/help/matlab/matlab_external/pass-data-to-matlab-from-python.html)  
[Passing arrays to Matlab](https://www.mathworks.com/help/matlab/matlab_external/matlab-arrays-as-python-variables.html)  
[Call Matlab Functions from Python](https://www.mathworks.com/help/matlab/matlab_external/call-matlab-functions-from-python.html)

## Coding Notes - HOHMM

* The states in HOHMM are mapped to the input state labels in the order that it encountered them.

* I suspect that the reason it doesn't use dummy states because it operates in the following way: for order _n_ ignore the first _n_ elements of the sequence and instead use them for the initial probability matrix, e.g. for 3rd-order ignore the first 3 "pos"/"neg" or whatever they are and built the following initial matrix: `['pos', 'neg'] ['posneg, 'pospos' ...] ['posposneg', 'pospospos' ...]`, the traditional matrix is on the 3rd list while the other 2 are reserved for the elements it ignores at the start of the sequqence.

[Documentation](https://simple-hohmm.readthedocs.io/en/latest/)

<br><br/>

## Architecture A

Even if it looks like it, it is not really a classification method and is not suited for classification tasks.
![Method A](Documentation%20Images/General%20HMM%20Method%201.png?raw=true)

## Architecture B

Pure classification-based approach.
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

(7) **Non-sequential HMM** [any-based] [★★★] [TODO] [B] [[Iglesias et al.](https://search.proquest.com/openview/2119a1817a1618e9220edc27fea5ba1b/1?pq-origsite=gscholar&cbl=2049104)] [[Yi](https://dl.acm.org/citation.cfm?id=1168646)] [[Yi et al.](https://journals.sagepub.com/doi/pdf/10.1177/0165551508092257?casa_token=QZGFQENo5t0AAAAA:Plxel2MQi8LMt8TYRA21FQW4NgACsgLTVYlXFsSSvWuC0i3lokWWA0o0Te59xBOK_8WPWMaj2So)] : Is applicable to scenarios where the transition probabilities are not a statistical reflection of the relationship among different states like in traditional HMMs. This occurs when each state represents a different information source and then the relationship between states can be rephrased as the one between different information sources. What is considered as training data is two separate data sets from different sources, which does not seem to have relevant information about their relationship. For example, imagine a HMM consisting of three hidden states, one for the title of a text, one for the body and one for keywords.

(8) **Autoregressive HMM** [ ] [ ] [ ] [ ] [Ephraim et al.](https://www.computer.org/csdl/proceedings-article/icassp/1988/00196638/12OmNynJMG9)] : Tackles the problem of capturing correlations between observed variables that are far away from each other in terms of time steps.

(9) **Input-output HMM** [ ] [ ] [ ] [ ] [[Bengio et al.](http://papers.nips.cc/paper/964-an-input-output-hmm-architecture.pdf)] : Introduces observed variables that can influence either the hidden state variables, output variables or both. This technique can be particularly helpful in the domain of supervised learning for sequential data. However, I think it requires data/information in a specific "previous-current" form.  

(10) **Bidirectional HMM** [any-based] [ ] [ ] [ ] [[Zacher et al.](http://msb.embopress.org/content/msb/10/12/768.full.pdf)] [[Arani et al.](https://digital-library.theiet.org/content/journals/10.1049/iet-cvi.2017.0645)]  

(11) **Hierarchical HMM** [ ] [ ] [ ] [ ] [[Fine et al.](https://link.springer.com/content/pdf/10.1023/A:1007469218079.pdf)]  

(12) The remaining HMM models that alter assumptions about time etc. (e.g. Semi-Markov)

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
(7) It is possible to have HMM that don't utilize sequential data at all.

<br><br/>

# To Do

* Fix general HMM code to be mapped on the states _s<sub>i</sub>_ depending on alphabetical order. However, I already have a smart implementation on `Finegrained.py` by using state names as labels instead of "pos"/"neg"/"neu".
* On the IMDb artificial labels HMM, try to train it only on emotional words instead of all the words while using a state-of-the-art lexicon/tool/whatever.
* Switch code to object-oriented, where the class has a function to plot results (e.g. functions such as 'compare_length', 'print_console_to_text', 'plot_results').
* Look for ideal smoothing factor.
* Look for ideal ensemble of HMMs such as dor product instead of average. 
* Bidrectional HMM, fuse like on the Arani paper.
* See HMM (3) discrete.  
* Dataframe sample function should take random_state as a parameter.
* Take a look at the weighting function from Quan and Ren.
* Use Stanford Sentiment Treebank and its tree structure to split it into sequences (alternative source https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/data) leading to a HMM with 2-3 sentences per document.
* Attempt different Ensembles (see Arani paper) instead of average, e.g. multiply.
* Attempt different smoothing techniques (http://www.aclweb.org/anthology/P99-1023 , Section 3.2)

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
