# *_*coding:utf-8 *_*
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft


class DataCsv:
    def __init__(self, data_time, data_freq, freq_stop, thread, save_path, display):
        """
        fonction d'initialisation de la classe DataCsv
        :param data_time: suite de temps
        :param data_freq: suite de fréquence
        :param freq_stop: seuil au-delà duquel les fréquences ne sont plus prises en considérations, valeur par défaut None
        :param thread: seuil de l'offset du spectre dont on ne taite pas les valeurs en-dessous
        :param save_path: adresse de l'enregistrement
        :param display: boolean indiquant si on rafraichit l'image sur l'écran
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
        Enregistrement du spectre
        :param path: adresse de l'enregistrement
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
        Analyse des données
        :param data_x: données du temps
        :param data_y: données de la tension
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
        Affichage du spectre après FFT
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
        Affichage de l'image initiale de la tension avant tout traitement
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
        Affichage de l'image du spectre après recomposition
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
        Affichage de toutes les images de fréquence
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
        Affichage de toutes les images avant FFT
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
        Affichaage de toutes les données 
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
    """
    Sous classe de la classe DataCsv
    """
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
        Fonction de l'initialisation de la sous classe Data
        :param csv_path: Adresse du fichier CSV initial
        :param png_path: Adresse de l'image initiale
        :param freq_stop: seuil au-delà duquel les fréquences ne sont plus prises en considérations, valeur par défaut None
        :param thread: seuil de l'offset du spectre dont on ne taite pas les valeurs en-dessous
        :param save_dir: Dossier de l'enregistrement des résultats de l'analyse
        :param display: boolean indiquant si on rafraichit l'image sur l'écran
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
        Affichage de l'image initiale
        :return:
        """
        if not self.display:
            print("Display not allowed in save mode")
        else:
            plt.imshow(self.png)
            plt.show()


def read_txt_file(path="Lampe.csv"):
    """
    Lecture du fichier CSV
    :param path: Adresse du fichier
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



def fft_auto(time, y, te=1 / 20000, fstop=None):
    """
    FFT automatique
    :param time: suite du temps
    :param y: suite de tensions
    :param te: l'inverse de la fréqunce de prise d'échantillon
    :param fstop: seuil au-delà duquel les fréquences ne sont plus prises en considérations, valeur par défaut None
    :return: x(np.array)=gamme de fréquence, y_fft(np.array, Complexe)=Données complètes après FFT, y_fft_abs (np.array)=valeur absolue après FFT, y_ifft (np.array, Complexe)= iFFT après FFT  
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
    Recheche du pic à partir des données
    :param x: l'axe horizontal
    :param y: l'axe vertical
    :param thread: seuil de l'offset du spectre dont on ne taite pas les valeurs en-dessous
    :return: x_ (liste)= fréquences dont la valeur du pics > seuil, y_(liste)= valeurs>seuil, res (liste)= indexe des fréquences précédentes
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
    Tri d'une liste de trois catégories d'éléments en trois listes d'une catégorie d'éléments
    :param data_input: La liste initiale
    :param types: None en général
    :return: Un ensemble de trois listes après le tri
    """
    if types is None:
        types = ["time", "Ampere", "Volt"]
    return [data_input[type] for type in types]


def gen_f_pics(array_fe, list_points):
    """
    Retrouver les fréquences à partir des indexes contenues dans res (voir fonction find_pic)
    :param array_fe: liste complète des fréquences
    :param list_points: liste des indexes
    :return: new_array (np.array)= les fréquences
    """
    new_array = np.zeros_like(array_fe, dtype=np.complex)
    for index in list_points:
        new_array[index] = array_fe[index]
    new_array = ifft(new_array)
    return new_array


def check_dir(dir):
    """
    Vérification de l'existence de dossier et en cas de non existence, le créer
    :param dir: adresse du dossier
    :return: dir (string)= adresse du dossier
    """
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return dir


class Config:
    def __init__(self, csv_dir, png_dir, freq_stop, save_dir, display=False):
        """
        traitement pas lots des données
        :param csv_dir: Adresse du dossier des fichiers CSV initiaux
        :param png_dir: Adresse du dossier des images initiales
        :param freq_stop: seuil au-delà duquel les fréquences ne sont plus prises en considérations, valeur par défaut None
        :param thread: seuil de l'offset du spectre dont on ne taite pas les valeurs en-dessous
        :param save_dir: Dossier de l'enregistrement des résultats de l'analyse
        :param display: boolean indiquant si on rafraichit l'image sur l'écran
     
        """
        self.csv_dir = check_dir(csv_dir)
        self.png_dir = check_dir(png_dir)
        self.freq_stop = freq_stop
        self.save_dir = check_dir(save_dir)
        self.display = display
        self.csv_png = self.match()

    def __call__(self, *args, **kwargs):
        """
        Overwrite de la fonction Call 
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
        Faire un lien entre le fichier csv et le fichier png
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
