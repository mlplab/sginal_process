# coding: utf-8


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import dft_1d, idft_1d


save_dir = 'figure'


sns.set()


x = np.linspace(0, 1., 21)
y = np.sin(x * 2 * np.pi)
xx = np.linspace(0, 1., 1000)
yy = np.sin(xx * 2 * np.pi)


plt.figure(figsize=(16, 9))
plt.stem(x, y)
plt.plot(xx, yy, linestyle='--')
plt.xlabel(r'$t$', fontsize=24)
plt.ylabel(r'$f(t)$', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(save_dir, 'sin.pdf'))
plt.close()


dft = dft_1d(y)
plt.figure(figsize=(16, 9))
plt.stem(np.abs(dft)[:dft.shape[0] // 2])
plt.xlabel(r'$\omega$', fontsize=24)
plt.ylabel('spectral', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(save_dir, 'sin_dft.pdf'))
plt.close()
