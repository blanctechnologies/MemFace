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
from utils import get_Om

from datasets import Audio2ExpDataModule

wandb_logger = WandbLogger(name='Audio2Exp',project='MemFace')
pl.seed_everything(42, workers=True)
 #torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Audio2Exp(pl.LightningModule):
	def __init__(self):
		super().__init__()
		# self.save_hyperparameters()
		self.keys = nn.Embedding(M, d_k)
		self.values = nn.Embedding(M, d_v)

		self.M = 1000 # number of keys and values => output of f_enc is 1000
		self.d_k = 64
		self.d_v = 64
		
		
		self.encoder = Encoder()
		self.implicitmem = ImplicitMem(self.keys, self.values)
		self.decoder = Decoder() 

	def forward(self, audio_embed, sequence_lengths):
		encoded_audiofeature = self.encoder(audio_embed, sequence_lengths)
		output = self.decoder(encoded_audiofeature + self.implicitmem(encoded_audiofeature, sequence_lengths), sequence_lengths)
		return output

	def training_step(self, batch, batch_idx):
		# during first half of training alterating the learning of memory vs other parameters
		# how the first half is determined?
		# maybe it's done manually, just comment out this block in some time, but still how
		# it's determined?
		if batch_idx % 2 == 0:
				for param in self.implicitmem.parameters():
						param.requires_grad = False
				for param in self.encoder.parameters():
						param.requires_grad = True
				for param in self.decoder.parameters():
						param.requires_grad = True
		else:
				for param in self.implicitmem.parameters():
						param.requires_grad = True
				for param in self.encoder.parameters():
						param.requires_grad = False
				for param in self.decoder.parameters():
						param.requires_grad = False
		
		packed_audio_embed, packed_exp, packed_pose, packed_shape, packed_landmarks3d, sequence_lengths = batch
		packed_audio_embed = packed_audio_embed.to(device)
		packed_exp = packed_exp.to(device)
		packed_pose = packed_pose.to(device)
		packed_shape = packed_shape.to(device)
		packed_landmarks3d = packed_landmarks3d.to(device)


		padded_audio_embed, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_audio_embed, batch_first=True)
		padded_exp, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_exp, batch_first=True)
		padded_pose, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_pose, batch_first=True)
		padded_shape, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_shape, batch_first=True)
		padded_landmarks3d, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_landmarks3d, batch_first=True)
		
		# audio_embed, exp, pose, shape, landmarks3d = batch.to(device)
		exp_hat = self(audio_embed, sequence_length)
		
		l2_exp = torch.nn.MSELoss(exp_hat, exp)
		landmarks3d_hat = get_Om(pose, shape, exp_hat)
		# landmarks3d_hat = landmarks3d
		l2_vtx = torch.nn.MSELoss(landamrks3d_hat, landmarks3d) # dim(Om) = T × h_v × 3
		lmem_reg = 1/(self.m*(self.m - 1))*(torch.sum(pairwise_cosine_similarity(self.keys, reduction='sum')) + torch.sum(pairwise_cosine_similarity(self.values, reduction='sum')))

		loss = l2_exp + l2_vtx + 0.1 * lmem_reg
		self.log("loss", loss) 
		
		return loss

	def validation_step(self, batch, batch_idx):
		audio_embed, exp, pose, shape, landmarks3d = batch
		exp_hat = self(audio_embed)
		
		l2_exp = torch.nn.MSELoss(exp_hat, exp)
		landmarks3d_hat = get_Om(pose, shape, exp_hat)
		l2_vtx = torch.nn.MSELoss(landamrks3d_hat, landmarks3d) # dim(Om) = T × h_v × 3
		lmem_reg = 1/(self.m*(self.m - 1))*(torch.sum(pairwise_cosine_similarity(self.keys, reduction='sum')) + torch.sum(pairwise_cosine_similarity(self.values, reduction='sum')))

		val_loss = l2_exp + l2_vtx + 0.1 * lmem_reg
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

	def forward(self, query, sequence_lengths):
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
		self.l1 = nn.Linear(329, 64)
		self.relu = nn.ReLU()
		self.layernorm = nn.LayerNorm(64)
		self.dropout = nn.Dropout()
		self.pos_encoding = DynamicPositionalEncoding()
	
	def forward(self, x, sequence_lengths):
		x = self.l1(x)

		return output


class DynamicPositionalEncoding(nn.Module):
	def __init__(self, d_model=64, dropout=0.1, max_len=5000):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.d_model = d_model
		self.max_len = max_len


	def forward(self, x):
		# x.shape should be = (batch_size, seq_len, d_model=64)
		print(f'positional encoding forward(): x.shape = {x.shape}')
		sequence_length = x.size(1)
		position = torch.arange(sequence_length, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model)))
		pe = torch.zeros(sequence_length, d_model)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		x = x + pe
		result = self.dropout(x)
		# result.shape should be = (batch_size, seq_len, d_model=64)
		print(f'positional encoding forward(): result.shape = {result.shape}')

		return result

class Decoder(nn.Module):
	def __init__(self, d_model=64, nhead=1, num_layers=2):
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
if __name__ == '__main__':
		d_k = 64
		d_v = 64
		M = 1000


		torch.multiprocessing.set_start_method('spawn')
		audio2exp = Audio2Exp()
		datamodule = Audio2ExpDataModule()
		datamodule.setup()
		train_dataloader = datamodule.train_dataloader()
		val_dataloader = datamodule.val_dataloader()

		# callbacks=[EarlyStopping(monitor="val_loss", mode="min")], 
		trainer = pl.Trainer(fast_dev_run=True, logger=wandb_logger, devices=1, accelerator="gpu")
		trainer.fit(audio2exp, train_dataloader, val_dataloader)











