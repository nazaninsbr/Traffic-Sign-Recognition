import os
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


all_conf_matrix_files = ['../Generated_Files/'+x for x in os.listdir('../Generated_Files/') if x.endswith('_confusion_matrix.txt')]

for file in all_conf_matrix_files:
	print(file)
	array_of_items = []
	with open(file, 'r') as fp:
		content = [line.replace('[', '').replace(']', '').strip() for line in fp.read().split('], ')]
		for line in content:
			line_items = [int(val.strip()) for val in line.split(',')]
			array_of_items.append(line_items)
		df_cm = pd.DataFrame(array_of_items, range(len(array_of_items)), range(len(array_of_items)))
		plt.figure(figsize = (20,20))
		sn.heatmap(df_cm, annot=True, annot_kws={"size": 6})
		plt.savefig(file.replace('.txt', '_image.png'))