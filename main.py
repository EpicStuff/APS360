import torch, torchinfo, stuff, fastai.callback.schedule
from typing import Tuple
from torch import nn
from torch.utils.data import DataLoader, Dataset
from fastai.callback.all import ShowGraphCallback, EarlyStoppingCallback, CSVLogger, SaveModelCallback
from fastai.optimizer import OptimWrapper
from fastai.metrics import accuracy
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from functools import partial as wrap

class CombinedDataset(Dataset):
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = self.images[idx]
		label = self.labels[idx]
		return image, label
class Model(nn.Module):
	def __init__(self) -> None:
		# self.featurizer = nn.Sequential()  # deprecated, doing transfer learning instead
		self.classifier = nn.Sequential(

		)
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x = self.featurizer(x)
		x = self.classifier(x)
		return x

def main() -> None:
	# reproducibility
	stuff.manual_seed(64, True)
	# hyperparameters
	batch_size = 256
	num_epochs = 75
	loss = torch.nn.CrossEntropyLoss
	opt = torch.optim.Adam
	# "helpers"
	callbacks = [ShowGraphCallback(), EarlyStoppingCallback(patience=10), CSVLogger('model/model.csv'), SaveModelCallback()]
	# load data
	data_train, data_val, data_test = load_data()
	print(data_train)  # testing
	input()  # testing
	# train
	train(num_epochs, batch_size, data_train, data_val, loss, opt, callbacks)
def load_data(path: str = 'data', compression='.bz2') -> Tuple[Dataset, Dataset, Dataset]:
	'returns processed data'
	import pickle, bz2
	return pickle.load(bz2.open(path + '/train.pkl' + compression, 'rb')), pickle.load(bz2.open(path + '/valid.pkl' + compression, 'rb')), pickle.load(bz2.open(path + '/test.pkl' + compression, 'rb'))
def train(num_epochs: int, batch_size: int, data_train, data_val, loss, opt, callbacks: list):
	# defining model
	model = Model()
	# wrapping data
	data = DataLoaders(DataLoader(data_train, batch_size, True), DataLoader(data_val, batch_size, True))
	# wrapping optimizer
	opt = wrap(OptimWrapper, opt=opt)
	# defining learner and training, using mixed precision training to speed things ups
	learner = Learner(data, model, loss(), opt, metrics=accuracy).to_fp16()
	with learner.no_logging():
		learner.fit_one_cycle(num_epochs, cbs=callbacks)
	# print results
	for name, val in zip(learner.recorder.metric_names[1:], learner.recorder.values[-1]):
		print(name, ': ', val, ', ', sep='', end='')


if __name__ == '__main__':
	main()
