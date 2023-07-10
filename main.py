import torch, stuff, fastai.callback.schedule
from typing import Iterator, BinaryIO, Sequence
from pathlib import Path
from functools import partial as wrap
from ipynb.fs.defs.Data import CombinedDataset
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import models as m
from torchinfo import summary
from fastai.callback.all import ShowGraphCallback, EarlyStoppingCallback, CSVLogger, SaveModelCallback
from fastai.optimizer import OptimWrapper
from fastai.metrics import accuracy
from fastai.learner import Learner
from fastai.data.core import DataLoaders

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
	num_epochs = 50
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
def load_data(files: Sequence[str] = ['train', 'valid'], path: str = 'data/', compression='.bz2') -> Iterator[Dataset]:
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
def train(name: str, model: nn.Module, num_epochs: int, batch_size: int, data_train, data_val, loss, opt, verbose: int = 0):
	# "helpers"
	Path('models').mkdir(exist_ok=True)
	callbacks = [ShowGraphCallback(), EarlyStoppingCallback(patience=16), CSVLogger('models/' + name + '.csv'), SaveModelCallback(fname=name)]
	# wrapping data
	data = DataLoaders(DataLoader(data_train, batch_size, True), DataLoader(data_val, batch_size, True))
	# wrapping optimizer
	opt = wrap(OptimWrapper, opt=opt)
	# defining learner and training, using mixed precision training to speed things ups
	learner = Learner(data, model, loss(), opt, metrics=accuracy).to_fp16()  # type: ignore
	with learner.no_logging():
		learner.fit_one_cycle(num_epochs, cbs=callbacks)
	# print results
	if verbose:
		for name, val in zip(learner.recorder.metric_names[1:], learner.recorder.values[-1]):
			print(name, ': ', val, ', ', sep='', end='')
	return learner
def test():
	# def transfer_learning(data: Dataset, model, batch_size: int = 2048, device='cpu') -> Iterator[Tensor]:
	# 'extracts features from data using transfer learning'
	# with torch.no_grad():
	# 	model = model.to(device)
	# 	for imgs, labels in DataLoader(data, batch_size):
	# 		imgs = imgs.to(device).to(torch.float32)
	# 		yield model.features(imgs).to('cpu')
	# # transfer learning
	# data_train.images = list(transfer_learning(data_train, model, device='cuda'))
	# data_val.images = list(transfer_learning(data_val, model, device='cuda'))
	pass


if __name__ == '__main__':
	main()
