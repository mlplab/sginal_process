# coding: utf-8


import os
import matplotlib.pyplot as plt


mode = 'contour'
load_dir = '/'.join(('figure', mode))
save_dir = 'documents'
save_file_name = '/'.join([save_dir, f'output_{mode}.tex'])


output_txt = [r'\begin{figure}[h]']
img_list = os.listdir(load_dir)
img_list.sort()


output_txt += [r'\includegraphics[clip, width=.25\textwidth]{' + f'{"/".join([load_dir, name])}' + '}' for name in img_list]
output_txt.append(r'\end{figure}')
with open(save_file_name, 'w') as f:
    for txt in output_txt:
        f.writelines(txt + '\n')
