# coding: utf-8


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_img
from contour import ContourExtraction1D, ContourExtraction2D, FFT_2d


fmt = 'pdf'
filter_mode = 'high'
shape_mode = 'square'
data_path = 'figure'
figure_name = 'Lenna.png'
figure_path = os.path.join(data_path, figure_name)


img = cv2.imread(figure_path, cv2.IMREAD_GRAYSCALE)
plot_img(img)


contour = ContourExtraction1D()
for mode in ['prewitt', 'sobel']:
    contour_img = contour.process(img, mode)
    # contour_img = contour.laplacian(img)
    plot_img(contour_img, save_name=os.path.join(data_path, f'{mode}.{fmt}'))


contour = ContourExtraction2D()
for mode in ['lap_4', 'lap_8']:
    contour_img = contour.process(img, mode)
    plot_img(contour_img, save_name=os.path.join(data_path, f'{mode}.{fmt}'))


contour = FFT_2d()
size = (min(img.shape), min(img.shape))
fft_img = cv2.resize(img, size)
dft = contour.dft_2d(fft_img)
dft = np.fft.fftshift(dft)
contour.plot_spectral(dft, save_name=f'fft_countour_spectral.{fmt}')
idft = contour.band_path_filter(img, mode=filter_mode, shape='squaer', R=50,
                                save_filter=True,
                                save_filter_name=os.path.join(data_path, f'{filter_mode}_{shape_mode}_filter.{fmt}'))
contour.plot_spectral(idft, save_name=os.path.join(data_path, f'high_path_spectral.{fmt}'))
ifft_img = contour.idft_2d(idft)
plot_img(ifft_img, save_name=os.path.join(data_path, f'fft_contour.{fmt}'))
