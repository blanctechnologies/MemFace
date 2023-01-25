import os
import json
from typing import List

import torch
import hydra
from torch.cuda.amp import autocast
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


wandb_logger = WandbLogger(name='Audio2Exp',project='MemFace')
pl.seed_everything(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Audio2Exp(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.save_hyperparameters()

		self.M = 1000 # number of keys and values => output of f_enc is 1000
		self.d_k = 64
		self.d_v = 64
		self.keys = nn.Embedding(M, d_k)
		self.values = nn.Embedding(M, d_v)

		self.encoder = Encoder()
		self.implicitmem = ImplicitMem(keys, values)
		self.decoder = Decoder() 

	def forward(self, x):
		encoded_audiofeature = self.encoder(x)
		output = self.decoder(encoded_audiofeature + self.implicitmem(encoded_audiofeature))
		return output

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		l2_exp = torch.nn.MSELoss(y_hat, y)
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
	

class ImplicitMem(nn.Module):
	def __init__(self, keys, values, d_model=64, d_k=64, d_v=64, dropout=0.1, M=1000):
		super().__init__()
		
		self.keys = keys
		self.values = values
		self.d_k = d_k
		self.d_v = d_v
		self.w_q = nn.Linear(d_model, d_k, bias=False)
		self.w_k = nn.Linear(d_model, d_k, bias=False)
		self.w_v = nn.Linear(d_model, d_v, bias=False)
		self.w_o = nn.Linear(d_model, d_v, bias=False)

		self.dropout = nn.Dropout(dropout)

	def forward(self, query):
		q = self.w_q(query)
		k = self.w_k(self.keys)
		v = self.w_v(self.values)

		scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
		attention_weights = torch.softmax(scores, dim=-1)
		attention = torch.matmul(attention_weights, v)
		output = self.w_o(attention)
		output = self.dropout(attention)
		
		return output


class Encoder(nn.Module):
	def __init__(self):
		super().__init__()
		# change input accordingly to the size of the audio embedding 29 -> ?
		# Q1: should there be Relu between Linear and LayerNorm
		self.l1 = nn.Sequential(nn.Linear(29, 64), nn.ReLU(), nn.LayerNorm(64), nn.Dropout(), PositionalEncoding())
	def forward(self, x):
		return self.l1(x)


class PositionalEncoding(nn.Module):
	def __init__(self, d_model=64, dropout=0.1, max_len=5000):
		# change max len to max T, waiting on this info from Anni
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		"""
		Args:
			x: Tensor, shape [batch_size, seq_len, embedding_dim]
		"""
		x = x + self.pe[:x.size(0)]
		return self.dropout(x)


class Decoder(nn.Module):
	def __init__(self, d_model=64, nhead=8, num_layers=3):
		# TransformerEncoder or ConformerEncoder
		super().__init__()
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.fc = nn.Linear(d_model, 50)
	def forward(self, x):
		x = self.transformer_encoder(x)
		x = self.fc(x)
		return x


'''
# model
audio2exp = Audio2Enc(Encoder(), ImplicitMem(), Decoder())

# train model
trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")], logger= wandb_logger, gpus=1, distributed_backend='dp')
trainer.fit(model=audio2exp, train_dataloaders=train_loader)
'''
d_k = 64
d_v = 64
M = 1000
keys = nn.Embedding(M, d_k)
values = nn.Embedding(M, d_v)
implicitmem = ImplicitMem(keys, values)


print(implicitmem)
