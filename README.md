# Acoustic Flat Spot Detection

This repository includes full training code.


The files are expected to be located in `data/{station}/{*.aup}`  

`augment.py` contains classes for augmenting and pre processing data during training.

`callbacks.py` return metrics, early stopping and other modules for augmenting core training. 

`extract.py` provides modules for extracting resampled and framed data from raw annotations. **Run the script before training**

`learner.py` contains the train evaluate repeat cycle

`load.py` defines data loading, normalization and group sensitive splitting

`predict.py` wraps a model after training and allows to perform inference on whole files

`training.py` is a wrapper around the learner configuring the whole learners environment


