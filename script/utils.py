# coding: utf-8


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(65535)


def plot_img(img, *args, **kwargs):

    cmap = kwargs.get('cmap', 'gray')
    save_name = kwargs.get('save_name', None)
    if save_name is None:
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        plt.show()
    else:
        plt.imsave(save_name, img, cmap=cmap)


def create_noise(img, mode='point', **kwargs):
    img = (img / img.max()).astype(np.float32)

    if mode == 'point':
        black_ratio = .01
        white_ratio = .01
        noise = np.random.choice((0, 1, 255), img.shape, p=(black_ratio, 1 - (black_ratio + white_ratio), white_ratio))
    elif mode == 'wave':
        cicle = kwargs.get('cicle', 1. / 8.)
        x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
        noise = np.zeros_like(img).astype(np.float64)
        noise[x, y] = np.sin(x * cicle * np.pi) + np.cos((y + 1) * cicle * np.pi)

    img *= (noise * 255)
    img = img.clip(0, 255).astype(np.uint8)
    return img



def dft_1d(data):
    N = data.shape[0]
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    W = np.exp(-(2 * i * j  * np.pi) * 1j / N)
    result = data.dot(W)
    return result


def idft_1d(data):
    N = data.shape[0]
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    W = np.exp((2 * i * j  * np.pi) * 1j / N)
    result = data.dot(W)
    return result / N


def dft_2d(img):
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
