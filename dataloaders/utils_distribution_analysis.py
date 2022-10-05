import numpy as np
import matplotlib.pyplot as plt

def visualize_histogram(data, name):
    plt.clf()
    if len(data.shape) < 3:
        data = np.expand_dims(data, axis=2)

    rand_dim0 = np.arange(0, data.shape[0])
    rand_dim1 = np.arange(0, data.shape[1])
    np.random.shuffle(rand_dim0)
    np.random.shuffle(rand_dim1)

    values = []

    if data.shape[0] == 128:
        skip = 16
    else:
        skip = 4

    for i in range(0,data.shape[0], skip):
        for j in range(0,data.shape[1], skip):
            values.append(np.mean(data[i:i+skip,j:j+skip]))

    plt.plot([x for x in range(0,len(values))], values)
    plt.xlabel('random samples')
    plt.ylabel('values')
    plt.savefig(name)