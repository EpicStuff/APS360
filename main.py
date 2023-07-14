import torch, stuff, fastai.callback.schedule, numpy as np
from typing import Iterator, BinaryIO, Sequence
from pathlib import Path
from functools import partial as wrap
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import models as m
from torchinfo import summary
from fastai.callback.all import ShowGraphCallback, EarlyStoppingCallback, CSVLogger, SaveModelCallback
from fastai.optimizer import OptimWrapper
from fastai.metrics import accuracy
from fastai.learner import Learner
from fastai.data.core import DataLoaders


class CombinedDataset(Dataset):
	def __init__(self, images: list[np.ndarray], labels: torch.Tensor):
		self.images = torch.stack(list(map(lambda x: torch.from_numpy(x).to(torch.float16), images)))
		self.labels = labels.to(torch.int8)  # to reduce file size when saved
	def __len__(self):
		return len(self.images)
	def __getitem__(self, idx: int):
		return self.images[idx], self.labels[idx]
	def to(self, who: str, what):
		if 'images' in who:
			self.images = self.images.to(what)
		if 'labels' in who:
			self.labels = self.labels.to(what)
		return self
class Model(nn.Module):
	def __init__(self, p: float, parent, size: int) -> None:
		super().__init__()
		# for layer in parent.children():
		# 	for param in layer.parameters():
		# 		param.requires_grad = False
		self.parent = parent
		self.featurizer = nn.Sequential(
			nn.Flatten(),
		)
		self.classifier = nn.Sequential(
			nn.Linear(size, 1024),
			nn.Dropout(p),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.Dropout(p),
			nn.ReLU(),
			nn.Linear(512, 7),
		)
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			x = self.parent(x)
		x = self.featurizer(x)
		x = self.classifier(x)
		return x

def main() -> None:
	# reproducibility
	stuff.manual_seed(64, True)
	# hyperparameters, you should also tweak the layers in the model
	batch_size = 1024
	num_epochs = 5
	loss = torch.nn.CrossEntropyLoss
	opt = torch.optim.Adam
	parent = m.alexnet(weights=m.AlexNet_Weights.DEFAULT).features  # not the actual model but the model were transferring learning from
	dropout_prob = 0.3
	# load data
	data_train, data_val = load_data(compression='')
	# init model and train
	model = Model(dropout_prob, parent, 256 * 6 * 6)
	# debug
	summary(model.parent, (1, 3, 224, 224))
	summary(model.classifier, (1, 256 * 6 * 6))
	# training
	name = stuff.generate_name(batch_size, loss, opt, 'alexnet', dropout_prob)
	train(name, model, num_epochs, batch_size, data_train, data_val, loss, opt)
def load_data(files: Sequence[str] = ['train', 'valid'], path: str = 'data/', compression='') -> Iterator[Dataset]:
	'loads processed data'
	import pickle
	from rich.progress import open
	# # over complicated (and kinda sketchy) solution to get pickle.load to work
	# globals()['CombinedDataset'] = __import__('ipynb.fs.defs.Data', fromlist=['CombinedDataset']).CombinedDataset
	# placeholder function that does nothing
	def helper(x: BinaryIO) -> BinaryIO: return x
	# open and overwrite `func` if bz2 compression is used
	if compression == '.bz2':
		from bz2 import BZ2File
		helper = BZ2File  # type: ignore
	# for each file specified, load data
	for file in files:
		file = path + file + '.pkl' + compression
		with open(file, 'rb', description='Loading: ' + file) as file:
			yield pickle.load(helper(file)).to('labels', int)
def train(name: str, model: nn.Module, num_epochs: int, batch_size: int, data_train, data_val, loss, opt, verbose: int = 0, float16: bool = True):
	# "helpers"
	Path('models').mkdir(exist_ok=True)
	callbacks = [ShowGraphCallback(), EarlyStoppingCallback(patience=16), CSVLogger('models/' + name + '.csv'), SaveModelCallback(fname=name)]
	# wrapping data
	data = DataLoaders(DataLoader(data_train, batch_size, True), DataLoader(data_val, batch_size, True))
	# wrapping optimizer
	opt = wrap(OptimWrapper, opt=opt)
	# defining learner and training, using mixed precision training to speed things ups
	learner = Learner(data, model, loss(), opt, metrics=accuracy)  # type: ignore
	if float16:
		learner = learner.to_fp16()
	else:
		data_train = data_train.to('images', torch.float32)
		data_val = data_val.to('images', torch.float32)
	with learner.no_logging():
		learner.fit_one_cycle(num_epochs, cbs=callbacks)
	# print results
	if verbose:
		for name, val in zip(learner.recorder.metric_names[1:], learner.recorder.values[-1]):
			print(name, ': ', val, ', ', sep='', end='')
	return learner
def test():
	# reproducibility
	stuff.manual_seed(64, True)
	# load data
	data_test = list(load_data(['test'], compression=''))[0].to('images', torch.float32)
	# setup the model
	parent = m.alexnet(weights=m.AlexNet_Weights.DEFAULT).features  # not the actual model but the model were transferring learning from
	dropout_prob = 0.3
	batch_size = 1024
	loss = torch.nn.CrossEntropyLoss
	opt = wrap(OptimWrapper, opt=torch.optim.Adam)
	# init model and train
	model = Model(dropout_prob, parent, 256 * 6 * 6)
	# load model
	model.load_state_dict(torch.load('models/1024_CEL_Adam_alexnet_0.3.pth'))
	data = DataLoaders(DataLoader(data_test, batch_size, True), DataLoader(data_test, batch_size, True))
	learner = Learner(data, model, loss(), opt, metrics=accuracy)
	# test model
	return learner.validate()


if __name__ == '__main__':
	# main()
	print(test())
