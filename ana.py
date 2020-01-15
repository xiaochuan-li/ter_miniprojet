# *_*coding:utf-8 *_*
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft,ifft


def read_file(path='scope_1.csv'):
    file = pd.read_csv(path)
    return file['second'].values, file['Ampere'].values, file['Volt'].values


def trains(s):
    s.replace('E', '*10**')
    return eval(s)


def trans_list(l):
    res = []
    for term in l:
        res.append(trains(term))
    return res


def fft_auto(time, y, te=1 / 20000, fstop=None):
    x = np.linspace(0, len(time) * te, len(time)) / (len(time) * te ** 2)
    # y_fft = np.real(fft(y)) / (len(x) / 2)
    y_fft = fft(y)
    y_angle = np.angle(y_fft)[:len(x) // 2]
    y_fft = np.abs(y_fft) / (len(x) / 2)
    y_fft = y_fft[:len(x) // 2]
    x = x[:len(x) // 2]
    if fstop is not None:
        x = x[x < fstop]
        y_fft = y_fft[:len(x)]
    return x, y_fft, y_angle


def find_pic(x, y, seuil=0.01, num=100):
    res = []
    x_ = []
    y_ = []
    y_head = y[:-1]
    y_tail = y[1:]
    diff = (y_tail - y_head) / max(y)
    diff = diff[1:] * diff[:-1]
    diff /= min(diff)
    for i, term in enumerate(diff):
        if term >= seuil:
            x_.append(x[i + 1])
            y_.append(y[i + 1])
            res.append((x[i + 1], y[i + 1]))
            num -= 1
            if num == 0:
                break
    return x_, y_


def toString(point_x, point_y):
    res = (list(zip(point_x, point_y)))
    s = ''
    for ith, term in enumerate(res):
        s += "the {} pic : Fe = {:5.3f}; A = {:5.3f}\n".format(ith, term[0], term[1])
    return s


def recompose(point_x, point_y, angle_y):

    coe = list(zip(point_x, point_y, angle_y))
    coe = [x[1]*np.exp(np.complex(0,0)) for x in coe]
    coe=np.asarray(coe)
    coe=list(zip(coe, point_x))

    def func(x):
        res = np.zeros(x.shape,dtype=np.complex)
        j=np.complex(0,1)
        for term in coe:
            res += (term[0] * np.exp(x * 2 * np.pi * term[1] * j))
        return res
    return func


def plot_fft(points, x, y):
    plt.plot(x, y, 'y', label='fft')
    plt.scatter(points[0], points[1])
    plt.show()


if __name__ == "__main__":
    time, i, v = read_file()
    te = 1 / 20000

    x, y_fft, y_angle = fft_auto(time, i)

    point_x, point_y = find_pic(x, y_fft)
    plot_fft([point_x, point_y], x, y_fft)
    res = toString(point_x, point_y)
    repaire = recompose(point_x, point_y, y_angle)
    y_i = repaire(time)
    plt.plot(time, y_i, 'r', label=u'recomposed line')
    plt.plot(time, i, label='original line')
    plt.legend()
    plt.show()

