import os
import json
from typing import List

import hydra
from torch.cuda.amp import autocast
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from deepspeech.pytorch.deepspeech_pytorch.model import DeepSpeech
from deepspeech.pytorch.deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech.pytorch.deepspeech_pytorch.decoder import Decoder
from deepspeech.pytorch.deepspeech_pytorch.data_loader import ChunkSpectrogramParser
from deepspeech.pytorch.deepspeech_pytorch.utils import load_decoder, load_model


pl.seed_everything(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Audio2Exp(pl.LightningModule):
	def __init__(self, encoder, implicitmem, decoder):
		super().__init__()
		self.save_hyperparameters()


		self.encoder = encoder
		self.implicitmem = implicitmem
		self.decoder = decoder # f_dec
		self.query = None # f_enc(A), dim = T x d_k
		self.keys = None # dim = m x h_a
		self.values = None # dim = m x h_a
		self.m = 1000 # number of keys and values => output of f_enc is 1000

	def forward(self, a, mask=None, return_attn=None):
		# transformer with single head scaled dot product attn	
		scaled_dot_product()
		return

	def training_step(self, batch, batch_idx):
		x, y = batch
		loss = None # fill in later
		self.log("loss", loss) 
		return loss

	def validation_step(self, batch, batch_idx):
		val_loss = None # fill in later
		self.log("val_loss", val_loss)

	def configure_optimizers(self):
		# 1e-4 during training, 5e-6 during adaptation(200 epoch)
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) 
		return optimizer
	
	def scaled_dot_product(self, q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

		
class Encoder(nn.Module):
	def __init__(self):
		super().__init__()
		self. = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

  def forward(self, x):
    return self.l1(x)


class Decoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
	def forward(self, x):
		return self.l1(x)


class ImplicitMem(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = x
	def forward(self, x):
		return self.l1(x)


# model
audio2exp = Audio2Enc(Encoder(), ImplicitMem(), Decoder())

# train model
trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model=audio2exp, train_dataloaders=train_loader)


