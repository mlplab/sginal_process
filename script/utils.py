# coding: utf-8


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(65535)


def plot_img(img, *args, **kwargs):

    save_name = kwargs.get('save_name', None)
    if save_name is None:
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        plt.imsave(save_name, img, cmap='gray')


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
