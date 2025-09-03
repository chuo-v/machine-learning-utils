# machine-learning-utils

This is a collection of Python utilities that I wrote for use in machine learning.

## Recent Updates

The following are some of the most recent updates:
- added [stacking_predictions_retriever.py](ensemble-learning/stacking/stacking_predictions_retriever.py)
  - to streamline my use of stacking, an [ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning) strategy, in Kaggle competitions (see my [kaggle-notebooks](https://github.com/chuo-v/kaggle-notebooks) repository for more details), I implemented a utility that would allow me to experiment more quickly with different combinations of base models
  - the utility saves predictions from the base models in files, and helps to keep track of the models that have non-stale predictions so that they can be reused, saving the time that would have otherwise been spent on making (the same) predictions using the estimators again