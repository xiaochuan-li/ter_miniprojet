# *_*coding:utf-8 *_*
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft


class DataCsv:
    def __init__(self, data_time, data_freq, freq_stop, thread, save_path, display):
        """

        :param data_time:
        :param data_freq:
        :param freq_stop:
        :param thread:
        :param save_path:
        :param display:
        """
        self.save_path = save_path
        self.data_time = data_time
        self.data_freq = data_freq
        self.display = display
        self.t, self.i, self.v = unzip(self.data_time)
        self.f, self.a = unzip(self.data_freq, types=["fre", "Ampere"])
        self.delta_t = (self.t[-1] - self.t[0]) / len(self.t)
        self.delta_f = (self.f[-1] - self.f[0]) / len(self.f)
        self.freq_stop = freq_stop
        self.thread = thread
        self.fft_x, self.fft_y, self.fft_y_abs, self.ifft_y = fft_auto(
            self.t, self.i, self.delta_t, fstop=self.freq_stop
        )
        self.point_x, self.point_y, self.pic_index = find_pic(
            self.fft_x, self.fft_y_abs, thread=self.thread
        )
        self.y_re = gen_f_pics(self.fft_y, self.pic_index)

    def save_spectres(self, path):
        """

        :param path:
        :return:
        """
        with open(path, "w") as f:
            for index in self.pic_index:
                f.write(
                    "{:9.7} {:9.7} {:9.7}\n".format(
                        self.fft_x[index],
                        np.real(self.fft_y[index]),
                        np.imag(self.fft_y[index]),
                    )
                )

    def analyse(self, data_x=None, data_y=None):
        """

        :param data_x:
        :param data_y:
        :return:
        """
        data_x = self.t if data_x is None else data_x
        data_y = self.i if data_y is None else data_y
        delta_x = (data_x[-1] - data_x[0]) / len(data_x)
        self.fft_x, self.fft_y, self.fft_y_abs, self.ifft_y = fft_auto(
            data_x, data_y, delta_x, fstop=self.freq_stop
        )
        self.point_x, self.point_y, self.pic_index = find_pic(
            self.fft_x, self.fft_y_abs, thread=self.thread
        )
        self.y_re = gen_f_pics(self.fft_y, self.pic_index)

    def plot_fft_o(self):
        """

        :return:
        """
        plt.title("Plot des FFTs")
        plt.xlabel("frequence(Hz)")
        plt.ylabel("spectre")
        plt.axis(
            [
                min(self.point_x) - 100,
                max(self.f) + 1500,
                min(self.fft_y_abs) - 0.01,
                max(self.fft_y_abs) + 0.01,
            ]
        )
        plt.plot(self.fft_x, self.fft_y_abs, "y", label="FFT Calculé")
        plt.scatter(self.point_x, self.point_y, label="Points Dominants")
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "FFT.png"))
        if self.display:
            plt.show()
        plt.close()

    def plot_o(self):
        """

        :return:
        """
        plt.xlabel("temps(s)")
        plt.ylabel("courant(A)")
        plt.plot(self.t, self.i, "b", label="original line")
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "Original.png"))
        if self.display:
            plt.show()
        plt.close()

    def plot_r(self):
        """

        :return:
        """
        plt.xlabel("temps(s)")
        plt.ylabel("courant(A)")
        plt.plot(
            np.linspace(self.t[0], self.t[-1], len(self.y_re)),
            np.real(self.y_re) * 2,
            "y",
            label=u"recomposed line",
        )
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "RECOMPOSED.png"))
        if self.display:
            plt.show()
        plt.close()

    def plot_fft(self):
        """

        :return:
        """
        plt.title("Plot des FFTs")
        plt.xlabel("frequence(Hz)")
        plt.ylabel("spectre")
        plt.axis(
            [
                min(self.point_x) - 100,
                max(self.f) + 500,
                min(self.fft_y_abs) - 0.01,
                max(self.fft_y_abs) + 0.01,
            ]
        )
        plt.plot(self.f, self.a, "k", label="FFT Générateur")
        plt.plot(self.fft_x, self.fft_y_abs, "y", label="FFT Calculé")
        plt.scatter(self.point_x, self.point_y, label="Points Dominants")
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "FFT.png"))
        if self.display:
            plt.show()
        plt.close()

    def plot_recompose(self):
        """

        :return:
        """
        plt.title("Plot des Signals")
        plt.xlabel("temps(s)")
        plt.ylabel("courant(A)")
        plt.plot(
            np.linspace(self.t[0], self.t[-1], len(self.ifft_y)),
            self.ifft_y,
            "r",
            label=u"inversed line",
        )
        plt.plot(
            np.linspace(self.t[0], self.t[-1], len(self.y_re)),
            np.real(self.y_re),
            "yo",
            label=u"recomposed line",
        )
        plt.plot(self.t, self.i, "b.", label="original line")
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "RECOMPOSED_2.png"))
        if self.display:
            plt.show()
        plt.close()

    def toString(self):
        """

        :return:
        """
        res = list(zip(self.point_x, self.point_y))
        return "\n".join(
            [
                "the {} pic : Fe = {:5.3f}; A = {:5.3f}".format(ith, term[0], term[1])
                for ith, term in enumerate(res)
            ]
        )


class Data(DataCsv):
    def __init__(
            self,
            csv_path,
            png_path,
            freq_stop=None,
            thread=0.1,
            save_dir="result",
            display=False,
    ):
        """

        :param csv_path:
        :param png_path:
        :param freq_stop:
        :param thread:
        :param save_dir:
        :param display:
        """
        self.plot_dir = save_dir
        self.display = display
        if not os.path.isdir(self.plot_dir):
            os.mkdir(self.plot_dir)
        self.csv_path = csv_path
        self.png_path = png_path
        data_time, data_f = read_txt_file(csv_path)
        super().__init__(
            data_time=data_time,
            data_freq=data_f,
            freq_stop=freq_stop,
            thread=thread,
            save_path=self.plot_dir,
            display=self.display,
        )
        try:
            self.png = cv2.imread(png_path)
        except:
            self.png = np.zeros((480, 720))
            print("no png available")

    def show_res_gen(self):
        """

        :return:
        """
        if not self.display:
            print("Display not allowed in save mode")
        else:
            plt.imshow(self.png)
            plt.show()


def read_txt_file(path="Lampe.csv"):
    """

    :param path:
    :return:
    """
    with open(path, "r") as f:
        try:
            data_str_list = f.readlines()
            start_t, start_f = filter(
                lambda x: "x-axis" in data_str_list[x], range(len(data_str_list))
            )
            time_serie = data_str_list[start_t + 2: start_f]
            f_serie = data_str_list[start_f + 2:]
            data_time = np.zeros(
                (len(time_serie)),
                dtype=np.dtype(
                    [("time", np.float), ("Ampere", np.float), ("Volt", np.float)],
                ),
            )
            data_f = np.zeros(
                (len(f_serie)),
                dtype=np.dtype([("fre", np.float), ("Ampere", np.float)]),
            )
            for i, line in enumerate(time_serie):
                data_time[i] = tuple(
                    np.asarray([float(x) for x in line.replace("\n", "").split(",")])
                )
            for i, line in enumerate(f_serie):
                data_f[i] = tuple(
                    np.asarray(
                        [
                            float(x) if x != "" else 0
                            for x in line.replace("\n", "").split(",")
                        ][:2]
                    )
                )
            return data_time, data_f
        except ValueError as e:
            print(e, "Error when reading")
            return None, None


def trains(s):
    """

    :param s:
    :return:
    """
    s.replace("E", "*10**")
    return eval(s)


def trans_list(l):
    """

    :param l:
    :return:
    """
    return [trains(term) for term in l]


def fft_auto(time, y, te=1 / 20000, fstop=None):
    """

    :param time:
    :param y:
    :param te:
    :param fstop:
    :return:
    """
    x = np.linspace(0, len(time) * te, len(time)) / (len(time) * te ** 2)
    y_fft = fft(y)
    y_fft_abs = np.abs(y_fft) / (len(x) / 2)
    y_fft_abs = y_fft_abs[: len(x) // 2]
    x = x[: len(x) // 2]
    if fstop is not None:
        x = x[x < fstop]
        y_fft_abs = y_fft_abs[: len(x)]
        y_fft = y_fft[: len(x)]
    y_ifft = np.real(ifft(y_fft))
    return x, y_fft, y_fft_abs, y_ifft


def find_pic(x, y, thread=0.25):
    """

    :param x:
    :param y:
    :param thread:
    :return:
    """
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
    """

    :param data_input:
    :param types:
    :return:
    """
    if types is None:
        types = ["time", "Ampere", "Volt"]
    return [data_input[type] for type in types]


def gen_f_pics(array_fe, list_points):
    """

    :param array_fe:
    :param list_points:
    :return:
    """
    new_array = np.zeros_like(array_fe, dtype=np.complex)
    for index in list_points:
        new_array[index] = array_fe[index]
    new_array = ifft(new_array)
    return new_array


def check_dir(dir):
    """

    :param dir:
    :return:
    """
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return dir


class Config:
    def __init__(self, csv_dir, png_dir, freq_stop, save_dir, display=False):
        """

        :param csv_dir:
        :param png_dir:
        :param freq_stop:
        :param save_dir:
        :param display:
        """
        self.csv_dir = check_dir(csv_dir)
        self.png_dir = check_dir(png_dir)
        self.freq_stop = freq_stop
        self.save_dir = check_dir(save_dir)
        self.display = display
        self.csv_png = self.match()

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        for i, term in enumerate(self.csv_png):
            name = os.path.basename(term[0])[:-4]
            print("Traiting the data : {}".format(name))
            check_dir(os.path.join(self.save_dir, name))
            data = Data(
                term[0],
                term[1],
                freq_stop=self.freq_stop,
                save_dir=os.path.join(self.save_dir, name),
                display=self.display,
            )
            data.plot_fft_o()
            data.plot_r()
            data.plot_o()
            data.plot_recompose()
            data.save_spectres(os.path.join(self.save_dir, name, "test.txt"))

    def match(self):
        """
        
        :return:
        """
        csv_lst = list(filter(lambda x: ".csv" in x, os.listdir(self.csv_dir)))
        png_lst = [x.replace(".csv", ".png") for x in csv_lst]
        return [
            (os.path.join(self.csv_dir, x[0]), os.path.join(self.png_dir, x[1]))
            for x in list(zip(csv_lst, png_lst))
        ]


if __name__ == "__main__":
    """
    data = Data("Lampe.csv", "Lamp.png", freq_stop=None, display_dir="DISPLAY")
    data.plot_fft()
    data.plot_recompose()
    data.show_res_gen()
    data.save_spectres("test.txt")
    """
    conf = Config(
        "data//CSV", "data//PNG", freq_stop=None, display=False, save_dir="DISPLAY"
    )
    conf()
