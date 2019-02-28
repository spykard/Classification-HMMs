# Hidden Markov Models

## Coding Notes

* The states on pomegranate, represented as strings _s<sub>i</sub>_ are mapped to the input states in an alphabetical order, e.g. `['bbb', 'aaa', 'ccc']` means: _s<sub>0</sub>='aaa', s<sub>1</sub>='bbb', s<sub>2</sub>='ccc'_

```python
  ...
  normal_states = list(sorted(normal_states, key=attrgetter('name')))
  silent_states = list(sorted(silent_states, key=attrgetter('name')))
  ...
```
## List of Models with emphasis on Natural Language Processing

(number) [on what type of sequential data it works on] [difficulty] [is it more or less ready] [references relevant to NLP]

(1) [any-based] [★] [yes] [[Manning et al.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.2604&rep=rep1&type=pdf)] [[Mathew](https://www.slideshare.net/thomas_a_mathew/text-categorization-using-ngrams-and-hiddenmarkovmodels)] **State-emission HMM**: An extremely basic form of HMM, used when we have no notion of class/category sequence. Definition: If the symbol emitted at time _t_ depends on both the state at time _t_ and at time _t+1_ we have what is sometimes called an arc-emission HMM. However, an alternative formulation is a state-emission HMM, where the symbol emitted at time _t_ depends just on the state at time _t_. For example, imagine a crazy soft drink machine that prefers to output a certain drink and after each output changes state randomly.

(2) [any-based] [★] [yes]  **Classic HMM**: The well-known sequential model used for Part-of-Speech tagging, Biology-related tasks, Image recognition (pixels as a sequence) etc.


(4) [any-based] [★★★★★] [yes] [[Quan et al.](https://www.sciencedirect.com/science/article/pii/S0020025515007057)] [[Preez](https://www.sciencedirect.com/science/article/pii/S0885230897900371)] **High-order HMM**: Lifts a major restriction of HMMs and allows the states to also depend on the observation/state preceding the directly previous one.

(5) [[Kang et al.](https://www.sciencedirect.com/science/article/pii/S0957417417304979)]    Clustering then HMM


(5) [any-based] **Bidirectional HMM**

(6) [ ] [ ] [ ] [[Fine et al.](https://link.springer.com/content/pdf/10.1023/A:1007469218079.pdf)] **Hierarchical HMM** 



## To Do

* Fix general HMM code to be mapped on the states _s<sub>i</sub>_ depending on alphabetical order

### Counting

* Showcase how as the order increases we have transitions that are too low probability; the problematic count will help with this.
* Showcase how as the order increases we have more empty sequences; `['neg', 'neg-pos', 'neg-pos-pos']` is an empty sequence since HOHMM doesn't use dummy states.
* Note the new unseen observation count which stays constant; the unseen count will help with this.
<br><br/>

## Conclusions

* A HMM can increase the performance of any ... by utilizing the sequential information of text
