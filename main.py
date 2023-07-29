import torch, stuff, fastai.callback.schedule, numpy as np
from typing import Iterator, BinaryIO, Sequence
from pathlib import Path
from functools import partial as wrap
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models as m
from torchinfo import summary
from fastai.callback.all import ShowGraphCallback, EarlyStoppingCallback, CSVLogger, SaveModelCallback
from fastai.optimizer import OptimWrapper
from fastai.metrics import accuracy
from fastai.learner import Learner
from fastai.data.all import DataLoaders, DataLoader
from fastai.vision.all import ImageDataLoaders


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
	def __init__(self, parent, featurizer, classifier) -> None:
		super().__init__()
		self.parent = parent
		self.featurizer = featurizer
		self.classifier = classifier
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			x = self.parent(x)
		x = self.featurizer(x)
		x = self.classifier(x)
		return x


# hyperparameters
params = {
	'batch_size': 256,
	'num_epochs': 50,
	'loss': torch.nn.CrossEntropyLoss,
	'opt': torch.optim.Adam,
	'parent': m.convnext_base(weights='DEFAULT'),
	'parent_name': 'convnext_base_modified',
	'featurizer': nn.Sequential(),
	'dropout_prob': 0,
}
params['parent'].classifier[2] = nn.Identity()  # disable parent classifier
params['parent_out'] = lambda: summary(params['parent'], (1, 3, 224, 224)).summary_list[-1].output_size[1]
params['name'] = lambda: stuff.generate_name(params['parent_name'], params['dropout_prob'])
params['classifier'] = nn.Sequential(
	nn.Dropout(params['dropout_prob']),
	nn.Linear(params['parent_out'](), 2),
)

def main(params: dict = params, data: list | None = None):
	# reproducibility
	stuff.manual_seed(64, True)
	# load data
	data_train, data_val = data or load_data()
	# init model
	model = Model(params['parent'], params['featurizer'], params['classifier'])
	# debug
	summary(model, (1, 3, 224, 224))
	# training
	return train(params['name'](), model, params['num_epochs'], params['batch_size'], data_train, data_val, params['loss'], params['opt'])
def load_data(files: Sequence[str] = ['train', 'valid'], path: str = 'data/', compression='') -> Iterator[Dataset]:
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
			yield pickle.load(helper(file)).to('labels', int)
def train(name: str, model: nn.Module, num_epochs: int, batch_size: int, data_train, data_val, loss, opt, verbose: int = 0, float16: bool = True):
	# "helpers"
	Path('models').mkdir(exist_ok=True)
	callbacks = [ShowGraphCallback(), EarlyStoppingCallback(patience=3), CSVLogger('models/' + name + '.csv'), SaveModelCallback(fname=name)]
	# wrapping data
	data = ImageDataLoaders(DataLoader(data_train, batch_size, shuffle=True), DataLoader(data_val, batch_size, shuffle=True))
	# wrapping optimizer
	opt = wrap(OptimWrapper, opt=opt)
	# defining learner and training, using mixed precision training to speed things ups
	learner = Learner(data, model, loss(), opt, metrics=accuracy)  # type: ignore
	if float16:
		learner = learner.to_fp16()
	else:
		data_train = data_train.to('images', torch.float32)
		data_val = data_val.to('images', torch.float32)
	learner.fit_one_cycle(num_epochs, cbs=callbacks)
	# print results
	if verbose:
		for name, val in zip(learner.recorder.metric_names[1:], learner.recorder.values[-1]):
			print(name, ': ', val, ', ', sep='', end='')
	return learner
def load_model(name: str, params: dict = params):
	# init model and train
	model = Model(params['dropout_prob'], params['parent'], params['parent_out'])
	# load model
	model.load_state_dict(torch.load(f'models/{name}.pth'))
	model.eval()
	return model
def test_model(name: str, params: dict = params):
	# reproducibility
	stuff.manual_seed(64, True)
	# load data
	data_test = list(load_data(['test'], compression=''))[0].to('images', torch.float32)
	# init model
	model = Model(params['dropout_prob'], params['parent'], params['parent_out'])
	# load model
	model.load_state_dict(torch.load(f'models/{name}.pth'))
	data = DataLoaders(DataLoader(data_test, params['batch_size'], True), DataLoader(data_test, params['batch_size'], True))
	learner = Learner(data, model, params['loss'](), params['opt'], metrics=accuracy)
	# test model
	return learner.validate()


if __name__ == '__main__':
	main()
