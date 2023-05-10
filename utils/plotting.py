from matplotlib import pyplot as plt

def plot_losses(epochs: int, losses: dict, title: str, output_dir=None):
    x = range(epochs)
    for key in losses.keys():
        plt.plot(x, losses[key], label='worker{}'.format(key))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if output_dir is not None:
        