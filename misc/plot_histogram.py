import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import ipdb

data_path = "./data/results.pkl"
data = pickle.load(open(data_path, "rb"))
n_primitives = 5
output_dir = "./data"

for k, v in data.items():
    plt.clf()
    hist, _ = np.histogram(v, np.arange(n_primitives+1))
    plt.bar(np.arange(n_primitives), hist)
    plt.title(k)
    plt.xlabel("different primitives")
    plt.ylabel("counts")
    plt.savefig(os.path.join(output_dir,"{}.jpg".format(k)))
