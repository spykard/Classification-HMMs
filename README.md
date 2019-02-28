# Hidden Markov Models

## Notes on Pomegranate

* The states represented as strings _s<sub>i</sub>_ are mapped to the input states in an alphabetical order, e.g. [_'bbb', 'aaa', 'ccc'_] means _s<sub>0</sub> = "aaa"_

```python
  ...
  normal_states = list(sorted(normal_states, key=attrgetter('name')))
  silent_states = list(sorted(silent_states, key=attrgetter('name')))
  ...
```
<br><br/>
## TODO

* Fix general HMM code to be mapped on the states _s<sub>i</sub>_ depending on alphabetical order

### Counting
* Showcase how as the order increases we have transitions that are too low probability; the problematic count will help with this.
* Showcase how as the order increases we have more empty sequences; `['neg', 'neg-pos', 'neg-pos-pos']` is an empty sequence since HOHMM doesn't use dummy states.
* Note the new unseen observation count which stays constant; the unseen count will help with this.

