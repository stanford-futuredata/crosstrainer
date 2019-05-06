# CrossTrainer: Practical Domain Adaptation with Loss Reweighting

This is an implementation of the method described in "CrossTrainer: Practical Domain Adaptation with Loss Reweighting" by Justin Chen, Edward Gan, Kexin Rong, Sahaana Suri, and Peter Bailis.

### Install
The crosstrainer package can be installed using pip.

```
pip install crosstrainer
```

### Usage

CrossTrainer utilizes loss reweighting to train machine learning models using data from a target task with supplementary source data.

##### Inputs:
Base model, target data, source data.

##### Outputs:
Trained model with optimized weighting parameter alpha.

##### Example:

```python
from crosstrainer import CrossTrainer
from sklearn import linear_model

lr = linear_model.LogisticRegression()
ct = CrossTrainer(lr, k=5, delta=0.01)
lr, alpha = ct.fit(X_target, y_target, X_source, y_source)
y_pred = lr.predict(X_test)
```

More examples can be found in the tests file: ```crosstrainer/tests/test_crosstrainer.py```.