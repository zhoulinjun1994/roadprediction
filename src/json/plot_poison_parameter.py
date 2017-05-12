#!/usr/bin/env python
# encoding: utf-8
# File Name: ll.py
# Author: Jiezhong Qiu
# Create Time: 2017/02/11 16:08 # TODO:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import rcParams
import cPickle as pickle
import json
import os


import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

#rc("font", **{"family":"serif", "serif":["Times"]})
rc("ytick", labelsize = 15)
rc("xtick", labelsize = 15)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
#rcParams['text.usetex'] = True


def plot_poisson_parameter(prefix=""):
    for json_file in os.listdir("./"):
        if not json_file.endswith("%s.json" % prefix):
            continue
        with open(json_file, "rb") as f:
            link, x, y = json.load(f)
            fig = plt.figure()
            plt.plot(x, y, color="royalblue")
            days = [i*24*7 for i in xrange(int(x[-1]/24./7.) + 1)]
            for day in days:
                plt.axvline(x=day, ymin=0, ymax=600, lw=0.5, ls="--")
            plt.savefig("poisson_%d_%s.eps" % (link, prefix), format="eps", bbox_inches='tight')

if __name__ == '__main__':
    plot_poisson_parameter(prefix="1h")
