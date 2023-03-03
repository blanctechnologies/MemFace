import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

class NeuralRender(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.save_hyperparameters()

		self.N = None
		self.d_k = None
		self.d_v = None
		self.keys = nn.Embedding(N, d_k)
		self.values = nn.Embedding(N, d_v)

		self.encoder = Encoder()
		self.explicitmem = ExplicitMem(keys, values)
		self.generator = Generator()
	
	def forward(self, x):
		return output
	
	def training_step(self, batch, batch_idx):
		x, y = batch

	def validation_step(self, batch, batch_idx):
		val_loss = None
		self.log('val_loss', val_loss)
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
		return optimizer
	
class Encoder(nn.Module):
	def __init__(self):
		
	def forward(self, x):

class ExplicitMem(nn.Module):
	def __init__(self, ):
		super().__init__()
		self.keys = keys
		self.values = values

	def forward(self, x):

class Generator(nn.Module):
	def __init__(self):

	def forward(self, x):

def buildExplicitMem(vid_folderpath):
	landmarks_path = ''
	images_path = ''


	
