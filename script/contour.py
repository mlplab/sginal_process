# coding: utf-8


import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_img


class ContourExtraction(object):

    def __init__(self):
        pass

    def conv1d(self, img, filters):
        x = cv2.filter2D(img, cv2.CV_64F, filters[0])
        y = cv2.filter2D(img, cv2.CV_64F, filters[1])
        return np.sqrt(x ** 2 + y ** 2)

    def conv2d(self, img, filters):
        return cv2.filter2D(img, cv2.CV_64F, filters)


class ContourExtraction1D(ContourExtraction):

    def __init__(self):
        super(ContourExtraction1D, self).__init__()
        self._1d_filter      = [np.array([[0,  0, 0],
                                          [0, -1, 1],
                                          [0,  0, 0]]),
                                np.array([[0,  0, 0],
                                          [0, -1, 0],
                                          [0,  1, 0]])]
        self._prewitt_filter = [np.array([[-1,  0,  1],
                                          [-1,  0,  1],
                                          [-1,  0,  1]]),
                                np.array([[-1, -1, -1],
                                          [ 0,  0,  0],
                                          [ 1,  1,  1]])]
        self._sobel_filter =   [np.array([[-1,  0,  1],
                                          [-2,  0,  2],
                                          [-1,  0,  1]]),
                                np.array([[-1, -2, -1],
                                          [ 0,  0,  0],
                                          [ 1,  2,  1]])]

    def process(self, img, mode='prewitt'):
        if mode == 'prewitt':
            return self.prewitt(img)
        elif mode == 'sobel':
            return self.sobel(img)
        elif mode == '1d':
            return self.contour_1d(img)

    def prewitt(self, img):
        return self.conv1d(img, self._prewitt_filter)

    def sobel(self, img):
        return self.conv1d(img, self._sobel_filter)

    def contour_1d(self, img):
        return self.conv1d(img, self._1d_filter)


class ContourExtraction2D(ContourExtraction):

    def __init__(self):
        super(ContourExtraction2D, self).__init__()
        self._lap_4 = np.array([[0,  1, 0],
                                [1, -4, 1],
                                [0,  1, 0]])
        self._lap_8 = np.array([[1,  1, 1],
                                [1, -8, 1],
                                [1,  1, 1]])

    def process(self, img, mode='lap_4'):
        mode_name = mode.split('_')[0]
        if mode_name == 'lap':
            filter_mode = int(mode.split('_')[-1])
            return self.laplacian(img, filter_mode)

    def laplacian(self, img, mode=4):
        if mode == 4:
            filter_2d = self._lap_4
        elif mode == 8:
            filter_2d = self._lap_8
        return self.conv2d(img, filter_2d)


class FFT_2d(object):

    def __init__(self):
        pass

    def dft_2d(self, img):
        N = img.shape[0]
        M = img.shape[1]
        i, j = np.meshgrid(np.arange(N), np.arange(M))
        omega = np.exp(-2 * np.pi * 1j / N)
        W = np.power(omega, i * j)
        result = W.dot(img).dot(W.T)
        return result / N / M

    def idft_2d(self, data):
        data = np.fft.ifftshift(data)
        N = data.shape[0]
        M = data.shape[1]
        i, j = np.meshgrid(np.arange(N), np.arange(M))
        omega = np.exp(2 * np.pi * 1j / N)
        W = np.power(omega, i * j)
        result = W.dot(data).dot(W.T)
        return result.real

    def plot_spectral(self, data, *args, **kwargs):
        save_name = kwargs.get('save_name', None)
        data_spe = 20 * np.log(np.abs(data))
        plot_img(data_spe, save_name=save_name)
        return self

    def band_path_filter(self, img, mode, *args, **kwargs):
        assert mode in set(['low', 'high']), 'please enter the mode "low" or "high"'
        plot = kwargs.get('plot', False)
        save_name = kwargs.get('save_name', None)
        save_filter = kwargs.get('save_filter', False)
        save_filter_name = kwargs.get('save_filter_name', f'{mode}_filter.pdf')
        R = kwargs.get('R', 50)
        shape = kwargs.get('shape', 'circle')
        dft = self.dft_2d(img)
        dft = np.fft.fftshift(dft)
        if mode == 'low':
            data = self.low_path_filter(dft, shape, R, save_filter, save_filter_name)
        else:
            data = self.high_path_filter(dft, shape, R, save_filter, save_filter_name)
        return data

    def low_path_filter(self, data, shape, R, save_filter, save_filter_name):
        N, M = data.shape[0], data.shape[1]
        low_path = np.zeros((N, M))
        if shape == 'circle':
            x, y = np.meshgrid(np.arange(N), np.arange(M))
            low_path[((x - N // 2) ** 2  + (y - M // 2) ** 2) < R ** 2] = 1
        else:
            low_path[N // 2 - R : N // 2 + R, M // 2 - R: M // 2 + R] = 1
        if save_filter:
            plt.imsave(save_filter_name, low_path, cmap='gray')
        return data * low_path

    def high_path_filter(self, data, shape, R, save_filter, save_filter_name):
        N, M = data.shape[0], data.shape[1]
        high_path = np.zeros((N, M))
        if shape == 'circle':
            x, y = np.meshgrid(np.arange(N), np.arange(M))
            high_path[((x - N // 2) ** 2  + (y - M // 2) ** 2) > R ** 2] = 1
        else:
            high_path[:N // 2 - R] = 1
            high_path[:, :M // 2 - R] = 1
            high_path[N // 2 + R:] = 1
            high_path[:, M // 2 + R:] = 1
        if save_filter:
            plt.imsave(save_filter_name, high_path, cmap='gray')
        return data * high_path
