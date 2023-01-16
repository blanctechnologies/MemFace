import os
import json
from typing import List

import hydra
from torch.cuda.amp import autocast
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
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

	def forward(self):
		# transformer with single head scaled dot product attn	
		self.feature_extractor.eval()
		scaled_dot_product()
		return

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		l2_exp = torch.nn.MSELoss(y_hat, y)
		# we need to pull Om along with y somehow
		# to get Om_hat we need a separate function:
		# step 1: get y_hat expressions
		# step 2: put y_hat expressions into 3d_face reconstruction
		# step 3: extract Om_hat
		l2_vtx = torch.nn.MSELoss(Om_hat, Om) # dim(Om) = T × h_v × 3
		lmem_reg = 1/(self.m*(self.m - 1))*(torch.sum(pairwise_cosine_similarity(self.keys, reduction='sum')) + torch.sum(pairwise_cosine_similarity(self.values, reduction='sum')))

		loss = l2_exp + l2_vtx + 0.1 * lmem_reg
		self.log("loss", loss) 
		
		return loss

	def validation_step(self, batch, batch_idx):
		val_loss = None # LATER
		self.log("val_loss", val_loss)

	def configure_optimizers(self):
		# 1e-4 training, 5e-6 adaptation(200 epoch)
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) 
		return optimizer
	
	def attn(self, q, k, v, mask=None):
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
		# change input accordingly to the size of the audio embedding 29 -> ?
		# Q1: should there be Relu between Linear and LayerNorm
		self.l1 = nn.Sequential(nn.Linear(29, 64), nn.ReLU(), nn.LayerNorm(64), nn.Dropout())
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


class LSTMEncoder(nn.Module):
	# will try this one later
	def __init__(self):
		super().__init__()
		self.input_dim = 29 # LATER, num of audio features, T x 29, where 29 is dim of audio features, T - num of frames
		self.hidden_dim = 64
		self.n_layers = 1
		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)

  def forward(self, x):
		h_0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
		c_0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())

		output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
    return final_hidden_state[-1])


# model
audio2exp = Audio2Enc(Encoder(), ImplicitMem(), Decoder())

# train model
trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model=audio2exp, train_dataloaders=train_loader)


