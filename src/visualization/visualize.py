import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import src.preprocess.dataset as dataset
import src.evaluate.evaluate as evaluate


def visualize(confusion, classes):

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # NumPy uses CPU here so we need to use a CPU version
    cax = ax.matshow(confusion.cpu().numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Set up title
    plt.title("Languages the RNN Guessed Correctly")

    plt.savefig("reports/figures/confusion.png")
    
    plt.show()



visualize(evaluate.confusion, classes=dataset.alldata.labels_uniq)
