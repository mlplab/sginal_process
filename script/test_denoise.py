# coding: utf-8


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_img, create_noise
from denoise import Denoise
from contour import FFT_2d


fmt='pdf'
noise_mode = 'point'
filter_mode = 'low'
shape_mode = 'square'
base_name = 'base_figure'
data_path = 'figure'
figure_name = 'Lenna.png'
figure_path = os.path.join(base_name, figure_name)


img = cv2.imread(figure_path, cv2.IMREAD_GRAYSCALE)
plot_img(img, save_name=os.path.join(data_path, f'Lenna.{fmt}'))
contour = FFT_2d()
size = (min(img.shape), min(img.shape))
fft_img = cv2.resize(img, size)
dft = contour.dft_2d(fft_img)
dft = np.fft.fftshift(dft)
contour.plot_spectral(dft, save_name=os.path.join('figure', f'Lenna_dft.{fmt}'))
img = create_noise(img, mode=noise_mode, cicle=1. / 2.)
plot_img(img)



mode_name = ['median', 'gaussian']
denoise = Denoise()
for mode in mode_name:
    denoise_img = denoise.process(img, mode, ksize=(7, 7), std=3)
    plot_img(denoise_img, save_name=os.path.join(data_path, f'{mode}_{noise_mode}.{fmt}'))


contour = FFT_2d()
size = (min(img.shape), min(img.shape))
fft_img = cv2.resize(img, size)
dft = contour.dft_2d(fft_img)
dft = np.fft.fftshift(dft)
contour.plot_spectral(dft)
print(os.path.join(data_path, f'{filter_mode}_filter.{fmt}'))
idft = contour.band_path_filter(img, mode=filter_mode, shape='squaer', R=50,
                                save_filter=True,
                                save_filter_name=os.path.join(data_path, f'{filter_mode}_{shape_mode}_filter.{fmt}'))
contour.plot_spectral(idft)
ifft_img = contour.idft_2d(idft)
plot_img(ifft_img) # , save_name=os.path.join(data_path, f'fft_denoise_{noise_mode}.{fmt}'))
