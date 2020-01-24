# *_*coding:utf-8 *_*
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft


class DataCsv:
    def __init__(self, data_time, data_freq, freq_stop, thread):
        self.data_time = data_time
        self.data_freq = data_freq
        self.t, self.i, self.v = unzip(self.data_time)
        self.f, self.a = unzip(self.data_freq, types=['fre', 'Ampere'])
        self.delta_t = (self.t[-1] - self.t[0]) / len(self.t)
        self.delta_f = (self.f[-1] - self.f[0]) / len(self.f)
        self.freq_stop = freq_stop
        self.thread = thread
        self.fft_x, self.fft_y, self.fft_y_abs, self.recompose_y = fft_auto(self.t, self.i, self.delta_t,
                                                                            fstop=self.freq_stop)
        self.point_x, self.point_y, self.pic_index = find_pic(self.fft_x, self.fft_y_abs, thread=self.thread)
        self.y_re = gen_f_pics(self.fft_y, self.pic_index)

    def analyse(self, data_x=None, data_y=None):
        data_x = self.t if data_x is None else data_x
        data_y = self.i if data_y is None else data_y
        delta_x = (data_x[-1] - data_x[0]) / len(data_x)
        self.fft_x, self.fft_y, self.fft_y_abs, self.recompose_y = fft_auto(data_x, data_y, delta_x,
                                                                            fstop=self.freq_stop)
        self.point_x, self.point_y, self.pic_index = find_pic(self.fft_x, self.fft_y_abs, thread=self.thread)
        self.y_re = gen_f_pics(self.fft_y, self.pic_index)

    def plot_fft(self):
        plt.title("Plot des FFTs")
        plt.xlabel("frequence(Hz)")
        plt.ylabel("spectre")
        plt.axis([min(self.point_x) - 100, max(self.f) + 250, min(self.fft_y_abs) - 0.01, max(self.fft_y_abs) + 0.01])
        plt.plot(self.f, self.a, "k", label="FFT Générateur")
        plt.plot(self.fft_x, self.fft_y_abs, 'y', label='FFT Calculé')
        plt.scatter(self.point_x, self.point_y, label='Points Dominants')
        plt.legend()
        plt.show()

    def plot_recompose(self):
        plt.title("Plot des Signals")
        plt.xlabel("temps(s)")
        plt.ylabel("courant(A)")
        plt.plot(np.linspace(self.t[0], self.t[-1], len(self.recompose_y)), self.recompose_y, 'r',
                 label=u'inversed line')
        plt.plot(np.linspace(self.t[0], self.t[-1], len(self.y_re)), self.y_re, 'yo', label=u'recomposed line')
        plt.plot(self.t, self.i, "b.", label='original line')
        plt.legend()
        plt.show()

    def toString(self):
        res = list(zip(self.point_x, self.point_y))
        return '\n'.join(
            ["the {} pic : Fe = {:5.3f}; A = {:5.3f}".format(ith, term[0], term[1]) for ith, term in enumerate(res)])


class Data:
    def __init__(self, csv_path, png_path, freq_stop=None, thread=0.25):
        self.csv_path = csv_path
        self.png_path = png_path
        data_time, data_f = read_txt_file(csv_path)
        self.data_in = DataCsv(data_time=data_time, data_freq=data_f, freq_stop=freq_stop, thread=thread)
        self.png = cv2.imread(png_path)

    def show_res_gen(self):
        cv2.imshow("resultat d'osciloscope", self.png)
        cv2.waitKey()


def read_txt_file(path='Lampe.csv'):
    with open(path, 'r') as f:
        data_str_list = f.readlines()
        start_t, start_f = filter(lambda x: "x-axis" in data_str_list[x], range(len(data_str_list)))
        time_serie = data_str_list[start_t + 2:start_f]
        f_serie = data_str_list[start_f + 2:]
        data_time = np.zeros((len(time_serie)),
                             dtype=np.dtype([("time", np.float), ("Ampere", np.float), ("Volt", np.float)], ))
        data_f = np.zeros((len(f_serie)), dtype=np.dtype([("fre", np.float), ("Ampere", np.float)]))
        for i, line in enumerate(time_serie):
            data_time[i] = tuple(np.asarray([float(x) for x in line.replace('\n', '').split(',')]))
        for i, line in enumerate(f_serie):
            data_f[i] = tuple(np.asarray([float(x) if x != '' else 0 for x in line.replace('\n', '').split(',')][:2]))
        return data_time, data_f


def trains(s):
    s.replace('E', '*10**')
    return eval(s)


def trans_list(l):
    return [trains(term) for term in l]


def fft_auto(time, y, te=1 / 20000, fstop=None):
    x = np.linspace(0, len(time) * te, len(time)) / (len(time) * te ** 2)
    y_fft = fft(y)
    y_fft_abs = np.abs(y_fft) / (len(x) / 2)
    y_fft_abs = y_fft_abs[:len(x) // 2]
    x = x[:len(x) // 2]
    if fstop is not None:
        x = x[x < fstop]
        y_fft_abs = y_fft_abs[:len(x)]
        y_fft = y_fft[:len(x)]
    y_recompose = np.real(ifft(y_fft))
    return x, y_fft, y_fft_abs, y_recompose


def find_pic(x, y, thread=0.25):
    res, x_, y_ = [[], [], []]
    diff = (y[1:] - y[:-1]) / max(y)
    diff = diff[1:] * diff[:-1]
    diff /= min(diff)
    for i, term in enumerate(diff):
        if term >= thread:
            x_.append(x[i + 1])
            y_.append(y[i + 1])
            res.append(i + 1)
    return x_, y_, res


def unzip(data_input, types=None):
    if types is None:
        types = ['time', 'Ampere', 'Volt']
    return [data_input[type] for type in types]


def gen_f_pics(array_fe, list_points):
    new_array = np.zeros_like(array_fe, dtype=np.complex)
    for index in list_points:
        new_array[index] = array_fe[index]
    new_array = ifft(new_array)
    return np.real(new_array)


if __name__ == "__main__":
    data = Data("Lampe.csv", "Lamp.png", freq_stop=None)
    data.data_in.plot_fft()
    data.data_in.plot_recompose()
    # data.show_res_gen()
