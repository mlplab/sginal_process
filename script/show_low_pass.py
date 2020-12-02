# coding: utf-8


import cv2
import numpy as np
import matplotlib.pyplot as plt


def low_pass_img(img, cut_x, cut_y):
    '''
    画像へのローパスフィルタの実装

    Parameters
    ----------
    img: 入力する画像
    cut_x   : x座標のカットオフ周波数
    cut_y   : y座標のカットオフ周波数

    Returns
    -------
    idft_img: ローパスフィルタを行った画像
    '''

    img = np.array(img, np.float32)

    spectral = np.fft.fft2(img)  # 高速フーリエ変換による周波数領域への変換
    spectral = np.fft.fftshift(spectral)  # 低周波成分を画像中心になるようにシフト移動

    spectral[:cut_x] = 0  # ローパスフィルタによる処理(x方向)
    spectral[-cut_x:] = 0
    spectral[:, :cut_y] = 0  # ローパスフィルタによる処理(x方向)
    spectral[:, -cut_y:] = 0

    spectral = np.fft.ifftshift(spectral)  # 周波数成分の並びを戻す
    idft_img = np.fft.ifft2(spectral)  # 高速フーリエ逆変換
    idft_img = idft_img.real  # 出力する実部
    return idft_img

