import numpy as np
import matplotlib.pyplot as plt

def fplt(x, y, color, label, title, xlabel, ylabel, xstyle = "normal", ystyle = "normal"):
    # Updating parameters, creat figure
    plt.rcParams.update({
        "font.size": 16, "axes.linewidth": 2,
        "xtick.major.width": 2, "ytick.major.width": 2})
    fig = plt.figure(figsize=(6, 5), facecolor= 'white')
    plt.style.use('default')

    # Set axes
    ax = fig.add_subplot()
    if np.ndim(x) == 2:
        for i in range(len(x)):
            ax.plot(x[i], y[i], color[i], label[i])
    elif np.ndim(x) == 1:
        ax.plot(x, y, color,label)


    # Tick parameters
    ax.tick_params(direction = 'in', top = True, right = True)

    # Basical parameters
    # ax.set_xlim(np.min(x), np.max(x))
    # ax.set_ylim(np.min(y), np.max(y))
    if xstyle == "log":
        ax.set_xscale('log')
    elif xstyle != "normal":
        raise ValueError("x-style is illegal.")
    if ystyle == "log":
        ax.set_yscale('log')
    elif ystyle != "normal":
        raise ValueError("y-style is illegal.")
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title, y=1.01)
    plt.legend(loc = 'upper right')
    plt.tight_layout()

    plt.grid(True)
    plt.legend()
    plt.tick_params(axis="both", direction="in", length=8, width=1.5)
    plt.tick_params(which="minor", direction="in", length=3, width=1)
    plt.grid(which="major", alpha=0.3)
    plt.grid(which="minor", alpha=0.1)

    plt.savefig(title)