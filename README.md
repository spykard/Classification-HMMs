# A state-of-the-art Hidden Markov Model Framework

Code for the papers:  

[Sentiment Analysis using Novel and Interpretable Architectures of Hidden Markov Models](https://www.sciencedirect.com/science/article/abs/pii/S0950705121005943) (Elsevier 2021).

[Hidden Markov Models for Sentiment Analysis in Social Media](https://ieeexplore.ieee.org/abstract/document/8885272) (BCD 2019).

[Machine Learning Techniques for Sentiment Analysis and Emotion Recognition in Natural Language](http://hdl.handle.net/10889/12628) (Thesis).

## Getting Started

* [Introduction](#introduction)
* [Interpretability](#interpretability)
* [Dependencies](#dependencies)
* [Running the Model](#running-the-model)
* [Implementation Details](#implementation-details)
  * [HMM Algorithms](#HMM-Algorithms)
  * [Categorization of HMM Models](#Categorization-of-HMM-Models-with-emphasis-on-Classification-Tasks)
  * [Coding Notes - Overall](#Coding-Notes---Overall)
  * [Coding Notes - Pomegranate](#Coding-Notes---Pomegranate)
  * [Coding Notes - Matlab](#Coding-Notes---Matlab)
  * [Coding Notes - HOHMM](#Coding-Notes---HOHMM)
  * [Experiment Notes](#Experiment-Notes)
  * [Random Notes](#Random-Notes)
  * [To Do - Future Work](#To-Do---Future-Work)
* [Citation](#citation)
* [Credits](#credits)

## Introduction

A **framework** that implements a wide variety of **Hidden Markov Models**. The main focus of the performed experiments is mainly on natural language processing, but the proposed approach can be applied to any classification task in general.  
The number of **implemented models**, whether that is different **architectures**, **algorithms** or **high-order** Hidden Markov Model variants, is unprecedented. Some major examples are provided below.

A traditional architecture, denoted as "Approach A".

![Approach A](https://github.com/spykard/Classification-HMMs/blob/master/Documentation%20Images/Approach%20A.png?raw=true "Architecture of Approach A for training HMMs")

An architecture denoted as "Approach B", where a single supervised HMM is trained for every class *y* in the data, by utilizing a set of labels and observations. These labels can take any form such as labels of sentences, labels of phrases or even labels of words. For instance, a set of data contains *n* documents, each one consists of a set of sentences *S*, which form a sequential vector: *d =* [*s<sub>1</sub>, s<sub>2</sub>, ..., s<sub>n</sub>*]. Every one of the aforementioned sentences is annotated with a label *x*.

![Approach B](https://github.com/spykard/Classification-HMMs/blob/master/Documentation%20Images/Approach%20B.png?raw=true "Architecture of Approach B for training HMMs")

An ensemble of Models.

![Ensemble](https://github.com/spykard/Classification-HMMs/blob/master/Documentation%20Images/Ensemble.png?raw=true "Overview of the proposed ranked weighted vote Ensemble")

## Interpretability

The proposed models are not only an effective and potent tool for classification, but they also possess **extremely high interpretability**.  
The HMMs can not only indicate the sentiment of parts of a sentence but they can also showcase the way that sentiment evolves from the start to the end of a sentence. **Unlike neural networks** and most machine learning approaches that operate as a **black box**, our approach has extremely high interpretability. Interpretability is important and desired in learning systems for a wide variety of reasons. First and foremost, in most cases, it is not enough to merely get a prediction, but we also want to know the reasoning behind that prediction, especially in cases of incompleteness in problem formalization ([Arrieta et al.](https://www.sciencedirect.com/science/article/pii/S1566253519308103), [Doshi-Velez et al.](https://arxiv.org/abs/1702.08608)).  
Furthermore, the original goal of **science** was to gain knowledge, not solve problems with big datasets and black box machine learning models ([Molnar](https://christophm.github.io/interpretable-ml-book/)) that achieve low, or even no interpretability. Thus, the proposed HMMs are reliable, trust-worthy, highly interpretable predictive models.  

Some raw in-practice examples of the model operating on random sentences are provided below.

![Example #1](https://github.com/spykard/Classification-HMMs/blob/master/Documentation%20Images/Example%202.png?raw=true "Example #1 illustrating the model in-practice on a random sentence")

![Example #2](https://github.com/spykard/Classification-HMMs/blob/master/Documentation%20Images/Example%204.png?raw=true "Example #2 illustrating the model in-practice on a random sentence")

## Dependencies

**Python** Version:
```
Python >= 3.6
```
Required for Baum-Welch, Maximum a Posteriori, Viterbi algorithms and other baseline components:
```
pomegranate >= 0.11.0
```
Required for Labeled algorithm and an alternative high-order implementation different to my own:
```
SimpleHOHMM >= 0.3.0
```
Required for k-fold Cross-Validation module and performance metrics:
```
scikit-learn >= 0.20.3
```
Required for plotting the performance of models:
```
matplotlib
```
Required for n-grams and high-order implementations:
```
nltk
```

## Running the Model

The preprocessing as well as the model training are executed by running a single .py script at a time.

Important Framework Functions:

#### `HMM_Framework.build()`

``` python
def build(self, architecture, model, framework, k_fold, boosting=False,
          state_labels_pandas=[], observations_pandas=[], golden_truth_pandas=[],
          text_instead_of_sequences=[], text_enable=False, 
          n_grams=1, n_target="", n_prev_flag=False, n_dummy_flag=False, 
          pome_algorithm="baum-welch", pome_verbose=False, pome_njobs=1, pome_smoothing_trans=0.0, pome_smoothing_obs=0.0,
          pome_algorithm_t="map",
          hohmm_high_order=1, hohmm_smoothing=0.0, hohmm_synthesize=False,
          architecture_b_algorithm="formula", formula_magic_smoothing=0.0
          ):
```

#### `Ensemble_Framework.ensemble_run()`

``` python
def ensemble_run(cross_val_prediction_matrix, mapping, golden_truth, mode, weights=None, 
                 use_log_prob=True, detailed=False):
```

## Implementation Details

### HMM Algorithms

| Training      | Architecture  | Required Data Sequences                    |
| ------------- |:-------------:| ------------------------------------------ |
| Baum-Welch    | A             | observations [✓], states [X], golden truth [X] / not only supervised |
| Baum-Welch    | B             | observations [✓], states [X], golden truth [✓] / supervised          |
| Viterbi       | A             | observations [✓], states [X], golden truth [X] / not only supervised |
| Viterbi       | B             | observations [✓], states [X], golden truth [✓] / supervised          |
| Labeled       | A             | observations [✓], states [✓], golden truth [X] / not only supervised |
| Labeled       | B             | observations [✓], states [✓], golden truth [✓] / supervised          |

| Prediction           | Architecture  | Required Data Sequences                            |  
| -------------------- |:-------------:| -------------------------------------------------- |
| Maximum a Posteriori | A             | observations [✓], states [X], golden truth [X] / not only supervised |
| Viterbi              | A             | observations [✓], states [X], golden truth [X] / not only supervised |
| Forward              | B             | observations [✓], states [X], golden truth [✓] / supervised          |
| Math Formula         | B             | observations [✓], states [✓], golden truth [✓] / supervised          |

### Categorization of HMM Models with emphasis on Classification Tasks

`(number) *Name* [on what type of sequential data it operates on] [difficulty] [approach A/B] [is it implemented] [references relevant to NLP]`

(1) **Classic HMM** [any-based] [★] [A&B] [yes]  [[Rabiner](https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)]: see Section 3 of paper.

(2) **State-emission HMM** [mainly sentence-based] [★] [A&B] [yes] [[Manning et al.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.2604&rep=rep1&type=pdf)] [[Mathew](https://www.slideshare.net/thomas_a_mathew/text-categorization-using-ngrams-and-hiddenmarkovmodels)]: see Section 3 of paper.

* For example, imagine a crazy soft drink machine that prefers to output a certain drink and after each output changes state randomly.
* states: sentence polarity labels, observations: sentence polarity labels

(3) **General Mixture HMM** [mainly sentence-based] [★] [A&B] [yes]: see Section 3 of paper.

* states: constant document labels, observations: sentence polarity labels

(4) **Stationary HMM** [any-based] [★★★] [A&B] [semi] [[Liu et al.](https://www.hindawi.com/journals/mpe/2015/987189/)] [Iglesias et al.](https://search.proquest.com/openview/2119a1817a1618e9220edc27fea5ba1b/1?pq-origsite=gscholar&cbl=2049104)] [[Yi](https://dl.acm.org/citation.cfm?id=1168646)] [[Yi et al.](https://journals.sagepub.com/doi/pdf/10.1177/0165551508092257?casa_token=QZGFQENo5t0AAAAA:Plxel2MQi8LMt8TYRA21FQW4NgACsgLTVYlXFsSSvWuC0i3lokWWA0o0Te59xBOK_8WPWMaj2So)]: see Section 3 of my paper.

* Liu et al. - states: 4 states, tf-idf etc., observations: the values of the features (there is no sequence)

(5) **Multivariate HMM** [any-based] [★★★★] [A&B] [semi] [[[Tune et al.](https://arxiv.org/pdf/1305.0321.pdf)] [[Li et al.](http://vision.gel.ulaval.ca/~parizeau/Publications/P971225.pdf)]: see Section 3 of paper.

* Can be either Continuous (Method A, new formula would need to be invented for Method B) or Discrete (Method A/B).

* Note: passing more than 100 words/distributions makes it crash, so it is pointless.
* Example of Discrete: pass multiple words on each sentence (useless since too many variables and each Discrete distribution is independent from other words); a very good idea would be to pass both the word and its Part-of-Speech tag on a Spyros HMM; another idea would be to pass the 1st most relevant word on the 1st Discrete distribution, the 2nd most relevant on the 2nd and so on. 
* Example of Continuous: pass the tfidf value of the entire sentence (but how would we do that).

(6) **High-order HMM** [any-based] [★★★★★] [B] [yes] [[Quan et al.](https://www.sciencedirect.com/science/article/pii/S0020025515007057)] [[Preez](https://www.sciencedirect.com/science/article/pii/S0885230897900371)] [[Kochanski](http://kochanski.org/gpk/teaching/0401Oxford/HMM.pdf)] [[Ching et al.](https://link.springer.com/chapter/10.1007/978-3-540-45080-1_73)] [[Lee et al.](http://link-springer-com-s.vpn.whu.edu.cn:9440/content/pdf/10.1007%2F11779568_74.pdf)] : Lifts a major restriction of HMMs and allows the states to also depend on the observation/state preceding the directly previous one. The implementation might work through Kochanski's transformation, Preez's transformation and [miniHMM's](https://github.com/joshuagryphon/minihmm/blob/master/minihmm/represent.py) dummy states and initial probabilities.  

* Quan et al. (+ Multivariate) - states: emotions, observations: custom encoding with 150 possible observations per state

(7) **Clustering then HMM** [any-based] [★★★★★] [B] [yes] [[Kang et al.](https://www.sciencedirect.com/science/article/pii/S0957417417304979)]  

* Kang et al. - states: clusters, obervations: words - We can either (1) form some sort of clustering on the SVD term-to-term dot matrix like Kang did, (2) form a clustering of the documents and then predict on each word, (3) form a clustering on word2vec ([better not](https://datascience.stackexchange.com/questions/30917/k-means-clustering-of-word-embedding-gives-strange-results)).
* Other ideas - Combine (5) and (6) or something, states: labels, observations: clusters

(8) **Artificial labels then HMM - Kardakis-HMM** [any-based] [★★★] [B] [yes] [Me] : We want to perform classification but in text-related tasks we have a single label instead of a sequence of labels. In order to tackle this, take any Machine Learning classifier or tool (e.g. [Naive Bayes](https://github.com/saumyakb/CS4740-NLP-Sentiment-Analysis-using-HMM)), train it on the data and then have it predict the class label of every single word - we create artificial labels. Even though we are performing predictions on the data it was trained on, the resulting labels are VERY noisy since we are doing it on a single-word basis (e.g. get me the sentiment label of "movie" and get me the label of "review"). Then, the proposed HMM is trained on the artificial data and actually performs better than the original classifiers. For even higher performance, have 10 state-of-the-art classifiers and tools (LSTM, lexicon etc.)  predict the (artificial) label of each word and perform a majority vote.

* S. Kardakis - states: class labels, obervations: words

(9) **Autoregressive HMM** [ ] [ ] [ ] [ ] [Ephraim et al.](https://www.computer.org/csdl/proceedings-article/icassp/1988/00196638/12OmNynJMG9)] : Tackles the problem of capturing correlations between observed variables that are far away from each other in terms of time steps.

(10) **Input-output HMM** [ ] [ ] [ ] [ ] [[Bengio et al.](http://papers.nips.cc/paper/964-an-input-output-hmm-architecture.pdf)] : Introduces observed variables that can influence either the hidden state variables, output variables or both. This technique can be particularly helpful in the domain of supervised learning for sequential data. However, I think it requires data/information in a specific "previous-current" form.  

(11) **Bidirectional HMM** [ ] [ ] [ ] [ ] [[Zacher et al.](http://msb.embopress.org/content/msb/10/12/768.full.pdf)] [[Arani et al.](https://digital-library.theiet.org/content/journals/10.1049/iet-cvi.2017.0645)]  

(12) **Hierarchical HMM** [ ] [ ] [ ] [ ] [[Fine et al.](https://link.springer.com/content/pdf/10.1023/A:1007469218079.pdf)]  

(13) The remaining HMM models that alter assumptions about time etc. (e.g. Semi-Markov)

### Coding Notes - Overall

* On architecture B's 'formula' algorithm, to get log probabilities we have to normalize the result of the math formula. [Kang et al.](https://www.sciencedirect.com/science/article/pii/S0957417417304979) divide the multiplied probability score by the length of the sequence.
* On architecture B's 'formula' algorithm, the ideal magic smoothing factor is around half of the smallest possible probability of observations, code is on "print_probability_parameters()".
* Sample_weights parameter only works when n_jobs=1  

### Coding Notes - Pomegranate

* The states in Pomegranate, represented as strings _s<sub>i</sub>_ are mapped to the input state labels in an alphabetical order, e.g. `['bbb', 'aaa', 'ccc']` means: _s<sub>0</sub>='aaa', s<sub>1</sub>='bbb', s<sub>2</sub>='ccc', s<sub>3</sub>='None-start', s<sub>4</sub>='None-end'_.

``` python
  ...
  normal_states = list(sorted(normal_states, key=attrgetter('name')))
  silent_states = list(sorted(silent_states, key=attrgetter('name')))
  ...
```

* For anything that isn't algorithm='labeled': just use the state_names parameter to avoid some bugs, e.g.1. state_labels:`["bbb", "aaa", "bbb"]` then state_names=`["aaa", "bbb"]`. EVEN better, convert all state labels to s<sub>0</sub>, s<sub>1</sub>, ... s<sub>n</sub>, e.g.2. state_labels:`["s2", "s1", "s0"]` then state_names=`["s0", "s1", "s2"]`.

* For algorithm='labeled': extremely bad implementation riddled with bugs, see [.py](/Utility%20Code/HMM%20supervised%20training%20'from_samples'%20function%20Bug%20showcase.py). The '_labeled_summarize' function is a mess.  
Observation Probabilities Bug: Instead of taking into consideration the current observation, it takes the previous observation for each time step. In detail: (1) if the label names are "s0", "s1" etc. even if 'state_names' parameter is not used, the bug happens, (2) if the label names are anything else and 'state_names' is used to assist the function, the bug happens, (3) however if the label names are anything else and 'state_names' is not used, the bug doesn't happen. The bug also doesn't happen in the opposite scenario, where "s0" etc. are used, but irelevant 'state_names' are also given.  
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

* Conclusion: Labeled is actually not implemented. Whatever is executed inside viterbi and baum-welch is missing. If I make a call to _viterbi, labeled gets fixed. Even though in the docs it explicitly says that the labels parameter only work for "lableed", all algorithms take them into consideration (at the start of from_samples) and work perfectly well.

* The emission-pseudocount does not work for algorithm='labeled'.

* The emission-pseudocount is not added to states that only occur at the start of sequences, e.g. observations:`[["234", "123", "234"], ["651", "1"]]` and state_labels:`[["s234", "s123", "s234"], ["s651", "s1"]]` means that "state651" will have probability of 1 for 651 and 0 for everything else.

### Coding Notes - Matlab
 
[Installing the Matlab Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)  
Path of execution: /usr/local/MATLAB/R2018b/extern/engines/python  
Path of installation: ~/anaconda3/envs/matlabCompatiblePython/lib/python3.6/site-packages  
Successful setup with: sudo /home/s/anaconda3/envs/matlabCompatiblePython/bin/python setup.py install  
(no need for --prefix="installdir" parameter)  

[Passing data to Matlab](https://www.mathworks.com/help/matlab/matlab_external/pass-data-to-matlab-from-python.html)  
[Passing arrays to Matlab](https://www.mathworks.com/help/matlab/matlab_external/matlab-arrays-as-python-variables.html)  
[Call Matlab Functions from Python](https://www.mathworks.com/help/matlab/matlab_external/call-matlab-functions-from-python.html)
[Function to be used](https://www.mathworks.com/help/stats/hmmestimate.html)

### Coding Notes - HOHMM

* The states in HOHMM are mapped to the input state labels in the order that it encountered them.

* I suspect that the reason it doesn't use dummy states because it operates in the following way: for order _n_ ignore the first _n_ elements of the sequence and instead use them for the initial probability matrix, e.g. for 3rd-order ignore the first 3 "pos"/"neg" or whatever they are and built the following initial matrix: `['pos', 'neg'] ['posneg, 'pospos' ...] ['posposneg', 'pospospos' ...]`, the traditional matrix is on the 3rd list while the other 2 are reserved for the elements it ignores at the start of the sequqence.

* [Documentation](https://simple-hohmm.readthedocs.io/en/latest/)

### Experiment Notes

(all) Utilizing n-grams on the observations heavily increases accuracy, see console logs and graphs.  
(all) Using a higher n-gram means more new unseen transitions, see graphs.  
(all) Adding the smoothing factor doesn't affect performance that much, at least on Pomegranate.  
(all) njobs = -1 (enabling parallelization with batches) has a peculiar effect of sometimes increasing or decreasing accuracy (by around 20% in both). This might be happening because the model does not overfit/train on the training data which leads to better performance on the test data.
(all) Training on the "neg" subset of the IMDb dataset on Pomegranate completely bugs when using emission_pseudocount or higher-order represented as first-order; possibly semi-supervised learning gets enabled; sequence are slightly longer than "pos" ones. Temporary fix is to shorten the "neg" sequences by 1.  
(all) Windows Operating Systems require `__name__ == "__main__":` to run [multiprocessing](https://stackoverflow.com/questions/24374288/where-to-put-freeze-support-in-a-python-script). Windows Operating Systems also don't support SenticNet package.  
(1) Can be run at any time.  
(2) [Experimental Results of State-emission HMM.txt](Logs/Experimental%20Results%20of%20State-emission%20HMM.txt)  
(3) Can be run at any time.  
(4) It is possible to have HMMs that don't utilize sequential data at all.  
(5) Didn't work  
(6) [Experimental Results of State-emission HMM.txt](Logs/Experimental%20Results%20of%20State-emission%20HMM.txt). Also, as the order increases we have transitions that have too low of a probability; the problematic count will help with this. As the order increases we also have more empty sequences; `['neg', 'neg-pos', 'neg-pos-pos']` is an empty sequence since HOHMM doesn't use dummy states.  
(7) Idea 2 didn't work, idea 3 didn't work [Experimental Results of Clustering Solver HMM.txt](Logs/Experimental%20Results%20of%20Clustering%20Solver%20HMM.txt).  
(8) A HMM can increase the performance of any bag-of-words-based Machine Learning classifier or tool by utilizing the sequential information of text. This is done by producing artificial labels. [Experimental Results of State-emission HMM.txt](Logs/Experimental%20Results%20of%20State-emission%20HMM.txt) and [Experimental Results on Big Dataset.txt](Logs/Experimental%20Results%20on%20Big%20Dataset.txt)  

### Random Notes

* A weird [implementation](https://github.com/larsmans/seqlearn/blob/master/seqlearn/hmm.py) where all words are one-hot encoded and used as observations; since it is built like a classifier, clueless people run it with (n_sequences, seq_length=1) by throwing a tf-idf matrix as the sequence of length 1 and it kind of works. states: Part-of-Speech tags, observations: one-hot encoded words.

* A Multivariate HMM on 6 Emotions with 2 Discrete Distributions, one for the words and one for Part-of-Speech tags; or one for the words and one for the emotion intensity. This won't work across entire documents but only on sentences. states: Artificial Labels, observations: 2 sets of Discrete Distributions.

### To Do - Future Work

* Take a look at the weighting function from [Quan et al.](https://www.sciencedirect.com/science/article/pii/S0020025515007057)
* Use Stanford Sentiment Treebank and its tree structure to split it into sequences ([alternative source](https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/data)) leading to a HMM with 2-3 sentences per document.
* Attempt different Ensembles (see Arani paper) instead of average, e.g. multiply.
* Attempt different smoothing techniques from [Thede et al.](http://www.aclweb.org/anthology/P99-1023)
* Implement 'text_instead_of_sequences' on build().
* Make weights on the Ensemble be relative to the exact accuracy of each base classifier.
* Implement BIRCH before the Spherical k-Means for a better initialization

## Citation

Please cite this work using the first paper below:

```
@article{kardakis2021sentiment,
  title={Sentiment analysis using novel and interpretable architectures of Hidden Markov Models},
  author={Perikos, Isidoros and Kardakis, Spyridon and Hatzilygeroudis, Ioannis},
  journal={Knowledge-Based Systems},
  volume={229},
  pages={107332},
  year={2021},
  publisher={Elsevier}
}

@inproceedings{kardakis2019hidden,
  title={Hidden Markov Models for Sentiment Analysis in Social Media},
  author={Perikos, Isidoros and Kardakis, Spyridon and Paraskevas, Michael and Hatzilygeroudis, Ioannis},
  booktitle={2019 IEEE International Conference on Big Data, Cloud Computing, Data Science \& Engineering (BCD)},
  pages={130--135},
  year={2019},
  organization={IEEE}
}
```

## Credits

[Spyridon Kardakis](https://www.linkedin.com/in/kardakis/)