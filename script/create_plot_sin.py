# coding: utf-8


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import dft_1d, idft_1d


save_dir = 'figure'


sns.set()


x = np.linspace(0, 1., 31)
y = 2 * np.sin(x * 2 * np.pi) + 3 * np.cos(6 * x * 2 * np.pi) + 4 * np.cos(10 * x * 2 * np.pi)
xx = np.linspace(0, 1., 1000)
yy = 2 * np.sin(xx * 2 * np.pi) + 3 * np.cos(6 * xx * 2 * np.pi) + 4 * np.cos(10 * xx * 2 * np.pi)


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
plt.figure(figsize=(8, 6))
plt.stem(np.abs(dft)[:dft.shape[0] // 2])
plt.xlabel(r'$\omega$', fontsize=24)
plt.ylabel(r'$F(\omega)$', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(save_dir, 'sin_dft.pdf'))
plt.close()


dft = dft_1d(y)
dft[:3] = 0
dft[8: dft.shape[0] // 2] = 0
dft[dft.shape[0] // 2: -8] = 0
dft[-3:] = 0
plt.figure(figsize=(8, 6))
plt.stem(np.abs(dft)[:dft.shape[0] // 2])
plt.xlabel(r'$\omega$', fontsize=24)
plt.ylabel(r'$F(\omega)$', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(save_dir, 'band_pass_dft.pdf'))
plt.close()


idft = idft_1d(dft)
plt.figure(figsize=(8, 6))
plt.stem(x, idft.real)
plt.plot(xx, 3 * np.cos(6 * 2 * np.pi * xx), linestyle='--', label=r'$\cos(6 \times 2 \pi t)$')
plt.xlabel(r'$t$', fontsize=24)
plt.ylabel(r'$f(t)$', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend()
plt.savefig(os.path.join(save_dir, 'band_pass_idft.pdf'))
plt.close()


dft = dft_1d(y)
dft[:7] = 0
dft[-7:] = 0
plt.figure(figsize=(8, 6))
plt.stem(np.abs(dft)[:dft.shape[0] // 2])
plt.xlabel(r'$\omega$', fontsize=24)
plt.ylabel(r'$F(\omega)$', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(save_dir, 'high_pass_dft.pdf'))
plt.close()


idft = idft_1d(dft)
plt.figure(figsize=(8, 6))
plt.stem(x, idft.real)
plt.plot(xx, 4 * np.cos(10 * 2 * np.pi * xx), linestyle='--', label=r'$\cos(10 \times 2 \pi t)$')
plt.xlabel(r'$t$', fontsize=24)
plt.ylabel(r'$f(t)$', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend()
plt.savefig(os.path.join(save_dir, 'high_pass_idft.pdf'))
plt.close()


dft = dft_1d(y)
dft[4: -4] = 0
plt.figure(figsize=(8, 6))
plt.stem(np.abs(dft)[:dft.shape[0] // 2])
plt.xlabel(r'$\omega$', fontsize=24)
plt.ylabel(r'$F(\omega)$', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(save_dir, 'low_pass_dft.pdf'))
plt.close()


idft = idft_1d(dft)
plt.figure(figsize=(8, 6))
plt.stem(x, idft.real)
plt.plot(xx, 2 * np.sin(2 * np.pi * xx), linestyle='--', label=r'$\sin(2 \pi t)$')
plt.xlabel(r'$t$', fontsize=24)
plt.ylabel(r'$f(t)$', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend()
plt.savefig(os.path.join(save_dir, 'low_pass_idft.pdf'))
plt.close()


dft = dft_1d(y)
idft = idft_1d(dft)
plt.figure(figsize=(8, 6))
plt.stem(x, idft.real)
plt.xlabel(r'$t$', fontsize=24)
plt.ylabel(r'$f(t)$', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig(os.path.join(save_dir, 'sin_idft.pdf'))
plt.close()
