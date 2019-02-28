# Hidden Markov Models

## Notes on Pomegranate

* The states represented as strings _s<sub>i</sub>_ are mapped to the input states in an alphabetical order, e.g. [_"bbb", "aaa", "ccc"_] means _s<sub>0</sub> = "aaa"_

```python
  ...
  normal_states = list(sorted(normal_states, key=attrgetter('name')))
  silent_states = list(sorted(silent_states, key=attrgetter('name')))
  ...
```
<br><br/>
## TODO

* Fix general HMM code to be mapped on the states _s<sub>i</sub>_ depending on alphabetical order
* Showcase how as the order increases we have transitions that are too low probability; the problematic count might help with this
* Note the new unseen observation count which stays constant
