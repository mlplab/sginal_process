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
low_path_dft = contour.band_path_filter(img, filter_mode, save_filter=True, save_filter_name=os.path.join(data_path, f'low_pass_filter_{shape_mode}.pdf'), shape=shape_mode, R=25)
contour.plot_spectral(low_path_dft, save_name=os.path.join(data_path, f'{filter_mode}_pass_dft_2d.pdf'),)
idft = contour.idft_2d(low_path_dft)
plot_img(idft.real, save_name=os.path.join(data_path, f'{filter_mode}_pass_idft_2d.pdf'))
