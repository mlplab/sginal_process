# coding: utf-8


import cv2
import numpy as np


class Denoise(object):

    def __init__(self):
        pass

    def process(self, img, mode='median', **kwargs):
        if mode =='median':
            ksize = kwargs.get('ksize', (5, 5))
            return cv2.medianBlur(img, ksize[0])
        elif mode == 'gaussian':
            ksize = kwargs.get('ksize', (5, 5))
            std = kwargs.get('std', 1)
            return cv2.GaussianBlur(img, ksize, std)
