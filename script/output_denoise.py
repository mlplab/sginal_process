# coding: utf-8


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_img, create_noise
from denoise import Denoise
from contour import FFT_2d


fmt='pdf'
noise_modes = ['wave', 'point']
shape_modes = ['square', 'wave']
filter_mode = 'low'
base_name = 'base_figure'
data_path = 'figure/denoise'
os.makedirs(data_path, exist_ok=True)
figure_name = 'Lenna.png'
figure_path = os.path.join(base_name, figure_name)


img = cv2.imread(figure_path, cv2.IMREAD_GRAYSCALE)
size = (min(img.shape), min(img.shape))
fft_img = cv2.resize(img, size)
plot_img(img)




for noise_mode in noise_modes:
    img = create_noise(img, mode=noise_mode, cicle=1. / 4.)
    plot_img(img, save_name=os.path.join(data_path, f'{noise_mode}_input.{fmt}'))

    mode_name = ['median', 'gaussian']
    denoise = Denoise()
    for mode in mode_name:
        denoise_img = denoise.process(img, mode, ksize=(7, 7), std=3)
        plot_img(denoise_img, save_name=os.path.join(data_path, f'{mode}_{noise_mode}.{fmt}'))

    denoise = FFT_2d()
    for shape_mode in shape_modes:

        dft = denoise.dft_2d(img)
        dft = np.fft.fftshift(dft)
        denoise.plot_spectral(dft, save_name=os.path.join(data_path, f'dft_{filter_mode}_{shape_mode}_spelctral.{fmt}'))
        idft = denoise.band_path_filter(img, mode=filter_mode, shape='squaer', R=20,
                                        save_filter=True,
                                        save_filter_name=os.path.join(data_path, f'{filter_mode}_{shape_mode}_filter.{fmt}'))
        denoise.plot_spectral(idft, save_name=os.path.join(data_path, f'idft_{filter_mode}_{shape_mode}_spelctral.{fmt}'))
        ifft_img = denoise.idft_2d(idft)
        plot_img(ifft_img, save_name=os.path.join(data_path, f'fft_denoise_{noise_mode}_{shape_mode}.{fmt}'))
