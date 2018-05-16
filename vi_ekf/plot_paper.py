import os
import numpy as np
from data import History
from plot_helper import plot_side_by_side, init_plots, get_colors
from tqdm import tqdm
import scipy.signal
import matplotlib.pyplot as plt
from pyquat import Quaternion
from import_log import import_log
import sys

fig_dir = os.path.dirname(os.path.realpath(__file__)) + "/../results/"

def plot_paper():
    data = {}

    data['PU'] = import_log(1526503296)
    data['PU+DT'] = import_log(1526503405)
    data['V'] = import_log(1526506904)
    data['PU+DT+KF'] = import_log(1526509777)

    global fig_dir
    if not os.path.isdir(fig_dir): os.system("mkdir " + fig_dir)


    plot_velocities(data)
    plot_attitude(data)
    plt.show()

def plot_velocities(data):
    global fig_dir

    colors = get_colors(len(data), plt.cm.jet)

    plt.figure(figsize=(16,10))
    ax = None
    lines = ['-' for i in range(len(data))]
    for i in range(3):
        ax = plt.subplot(3, 1, i + 1, sharex=ax)
        plt.plot(data['V'].t.vel, data['V'].vel[:, i], '--', label="truth", color=colors[0], linewidth=2)
        for j, (key, d) in enumerate(data.iteritems()):
            plt.plot(d.t.xhat, d.xhat[:, 3+i], lines[j], label=key, color=colors[j], linewidth=2)
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=False, shadow=False, ncol=5)
        plt.ylabel("m/s")
    plt.xlabel("s")
    plt.savefig(fig_dir + "velocities.pdf", bbox_inches='tight', dpi=600)


def plot_attitude(data):
    global fig_dir

    colors = get_colors(len(data), plt.cm.jet)

    plt.figure(figsize=(16,10))
    ax = None
    titles = ['roll', 'pitch', 'yaw']
    lines = ['-' for i in range(len(data))]
    for i in range(3):
        ax = plt.subplot(3,1,i+1, sharex=ax)
        plt.plot(data['V'].t.att, data['V'].euler[:, i], '--', label="truth", color=colors[0], linewidth=2)
        for j, (key, d) in enumerate(data.iteritems()):
            plt.plot(d.t.global_att, d.global_euler_hat[:,i], lines[j], label=key, color=colors[j], linewidth=2)
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=False, shadow=False, ncol=5)
        plt.ylabel(titles[i] + ' (rad)')
    plt.xlabel("s")
    plt.savefig(fig_dir + "attitude.pdf", bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    plot_paper()




