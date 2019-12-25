# Acoustic Flat Spot Detection


In order to build a flat spot classifier we build upon a dataset of acoustic drive by train recordings. 
Annotators marked sections containing flat spots in audacity project files.

The files are expected to be located in `data/{station}/{*.aup}`  
By calling `data_build_register.py` the recordings and corresponding labels are collected to a pandas data frame.
It is then saved to `data_register.pkl` and can be used for customizable data loaders.

Baseline models using readily available sklearn implementations can be found in `model_basic.py`.
 