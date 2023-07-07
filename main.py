import torch, torchinfo, stuff, fastai.callback.schedule
from typing import Iterator, BinaryIO
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import alexnet, AlexNet_Weights
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
	def __init__(self, p: float) -> None:
		# self.featurizer = nn.Sequential()  # deprecated, doing transfer learning instead
		self.classifier = nn.Sequential(
			nn.Linear(256 * 6 * 6, 1024),
			nn.Dropout(p),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.Dropout(p),
			nn.ReLU(),
			nn.Linear(512, 7),
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
	model = alexnet(weights=AlexNet_Weights.DEFAULT)  # not the actual model but the model were transferring learning from
	# "helpers"
	callbacks = [ShowGraphCallback(), EarlyStoppingCallback(patience=10), CSVLogger('model/model.csv'), SaveModelCallback(fname='model/model')]
	# load data
	data_train, data_val = load_data(compression='')
	# transfer learning
	data_train, data_val = transfer_learning((data_train, data_val), model)
	# init model and train
	model = Model(0.3)
	train(model, num_epochs, batch_size, data_train, data_val, loss, opt, callbacks)
def load_data(path: str = 'data/', files: list[str] = ['train', 'valid'], compression='.bz2') -> Iterator[Dataset]:
	'loads processed data'
	import pickle
	from rich.progress import open
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
			yield pickle.load(helper(file))

	# return pickle.load(bz2.open(path + '/train.pkl' + compression, 'rb')), pickle.load(bz2.open(path + '/valid.pkl' + compression, 'rb')), pickle.load(bz2.open(path + '/test.pkl' + compression, 'rb'))
def transfer_learning(datas: list | tuple, model) -> list[Dataset]:
	return [model.features(data) for data in datas]
def train(model: nn.Module, num_epochs: int, batch_size: int, data_train, data_val, loss, opt, callbacks: list):
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
