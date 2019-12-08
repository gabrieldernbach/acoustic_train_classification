import numpy as np


def split(data, a=0.6, b=0.8):
    """
    Create a random train, dev, test split of a pandas data frame
    """
    a, b = int(a * len(data)), int(b * len(data))
    data_shuffled = data.sample(frac=1).reset_index(drop=True)
    return np.split(data_shuffled, [a, b])
