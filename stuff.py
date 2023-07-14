import matplotlib.pyplot as plt
from typing import Iterator
from fastai.learner import Learner

def plot_grid(imgs: Iterator, y: int = None, x: int = None, size=(1, 1), names=None):
	x, y, = x or len(imgs), y or 1
	fig, axs = plt.subplots(y, x, figsize=(x * size[1], y * size[0]))  # Increase the figsize for larger overall plot
	# fig, axs = plt.subplots(y, x)
	fig.subplots_adjust(hspace=0, wspace=0)
	fig.tight_layout()
	for num, img in enumerate(imgs):
		ax = axs.flatten()[num]
		# plt.subplot(y, x, num + 1)
		ax.axis('off')
		ax.imshow(img[0].T)
		try:
			ax.set_title(imgs.dataset.classes[img[1]])
		except AttributeError:
			ax.set_title(names[num])
		if num == x * y - 1:
			break
	plt.show()
def manual_seed(seed: int, cuda=False):
	import numpy as np, torch, random
	random.seed(seed)  # Python
	np.random.seed(seed)  # Numpy vars
	torch.manual_seed(seed)  # PyTorch vars
	# torch.use_deterministic_algorithms(True)  # raises error which i have no idea how to solve
	if cuda:  # GPU vars
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
# by Ignacio Oguiza from https://forums.fast.ai/t/plotting-metrics-after-learning/69937/3, modified
def plot_metrics(learner: Learner, nrows=None, ncols=None, figsize=None, file=None, **kwargs):
	from fastai.imports import np, math, plt
	from fastai.torch_core import subplots
	metrics = np.stack(learner.recorder.values)
	names = learner.recorder.metric_names[1:-1]
	n = len(names) - 1
	if nrows is None and ncols is None:
		nrows = int(math.sqrt(n))
		ncols = int(np.ceil(n / nrows))
	elif nrows is None: nrows = int(np.ceil(n / ncols))
	elif ncols is None: ncols = int(np.ceil(n / nrows))
	figsize = figsize or (ncols * 6, nrows * 4)
	fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
	axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
	for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
		ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
		ax.set_ylim(bottom=0)
		ax.set_title(name if i > 1 else 'losses')
		ax.legend(loc='best')
	if file:
		plt.savefig(str(learner.path) + '/' + file, bbox_inches='tight')
		plt.close(fig)
	else:
		plt.show()
def generate_name(*args):
	from torch import nn, optim
	table = {nn.CrossEntropyLoss: 'CEL', optim.Adam: 'Adam', optim.SGD: 'SGD', nn.MSELoss: 'MSE'}  # object to str conversion table
	return '_'.join([table[arg] if arg in table else str(arg) for arg in args])
def inject_css(file, css='DarkReader.css', folder='old/'):
	from bs4 import BeautifulSoup
	with open(folder + file, 'r', encoding='utf8') as f:
		html = BeautifulSoup(f, 'lxml')
	css = html.new_tag('link', rel="stylesheet", href=css)
	html.head.append(css)
	with open(folder + file, 'w', encoding='utf8') as f:
		f.write(html.prettify())
def display():
	import IPython
	print('code from stuff.py:')
	return IPython.display.Code(filename='stuff.py', language='python3')
def colab_stuff():
	import os
	from google.colab import drive
	drive.mount('/content/gdrive')
	os.chdir('/content/gdrive/MyDrive/APS360')


if __name__ == '__main__':
	inject_css('Lab4 Data Imputation.html')
