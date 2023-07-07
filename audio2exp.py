import os
import json
from typing import List
import math

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

from emoca.gdl_apps.EMOCA.utils.load import load_model
from emoca.gdl.utils.FaceDetector import FAN
from emoca.gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
from emoca.gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test, decode
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning.callbacks import ModelCheckpoint

wandb_logger = WandbLogger(name='Audio2Exp',project='MemFace')
pl.seed_everything(42, workers=True)
#torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
checkpoint_callback = ModelCheckpoint(
    dirpath="/root/MemFace/MemFace/version_None/checkpoints",
    filename="model-{epoch:02d}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,  # Save the best model
    save_last=True  # Save the last epoch's model
)

class Audio2Exp(pl.LightningModule):
	def __init__(self):
		super().__init__()
		# self.save_hyperparameters()
		# self.keys = nn.Embedding(M, d_k)
		# self.values = nn.Embedding(M, d_v)
		
		self.keys = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1000, 64)))
		self.values = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1000, 64)))
		self.M = 1000 # number of keys and values => output of f_enc is 1000
		self.d_k = 64
		self.d_v = 64
		
		
		self.encoder = Encoder()
		self.implicitmem = ImplicitMem(self.keys, self.values)
		self.decoder = Decoder() 
		# emoca initialization

		path_to_models = "/root/MemFace/emoca/assets/EMOCA/models"
		model_name = 'EMOCA_v2_lr_mse_20'
		mode = 'detail'

		self.emoca, conf = load_model(path_to_models, model_name, mode)
		self.emoca.cuda()
		self.emoca.eval()

	def forward(self, packed_audio_embed):

		audiofeature = self.encoder(packed_audio_embed)
		# encoded_audiofeature is packed, cause it's easier, we need to unpack it
				
		unpacked_audiofeature, lengths = torch.nn.utils.rnn.pad_packed_sequence(audiofeature, batch_first=True)
		print(f'unpacked_audiofeature after encoding and padding packed seq: {unpacked_audiofeature.shape}')
		# implicitmem will unpack audiofeature and return unpacked result
		output = self.decoder(unpacked_audiofeature + self.implicitmem(audiofeature), lengths)
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
		
		packed_audio_embed, packed_exp, packed_pose, packed_shape, packed_landmarks3d = batch
		# packed_audio_embed = packed_audio_embed.to(device)
		# packed_exp = packed_exp.to(device)
		# packed_pose = packed_pose.to(device)
		# packed_shape = packed_shape.to(device)
		# packed_landmarks3d = packed_landmarks3d.to(device)

		audio_embed, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_audio_embed, batch_first=True)
		exp, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_exp, batch_first=True)
		pose, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_pose, batch_first=True)
		shape, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_shape, batch_first=True)
		landmarks3d, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_landmarks3d, batch_first=True)
		print(f'training loop audio_embed.shape: {audio_embed.shape}')	
		# audio_embed, exp, pose, shape, landmarks3d = batch.to(device)
		exp_hat = self(packed_audio_embed)
		# exp_hat, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_exp_hat, batch_first=True)
		
		mse_loss = torch.nn.MSELoss()	
		l2_exp = mse_loss(exp_hat, exp)
		print(f'training loop pose.shape after forward pass and before get_Om: {pose.shape}')
		landmarks3d_hat = get_Om(pose, shape, exp_hat, self.emoca, batch_size=16)
		# landmarks3d_hat = landmarks3d
		print(f'training loop landmarks3d.shape before squeeze: {landmarks3d.shape}')
		landmarks3d = torch.squeeze(landmarks3d, 2)
		print('---- after get_Om ----')
		print(f'training loop landmarks3d.shape after squezze: {landmarks3d.shape}')
		print(f'training loop landmarks3d_hat.shape: {landmarks3d_hat.shape}')
		l2_vtx = mse_loss(landmarks3d_hat, landmarks3d) # dim(Om) = T × h_v × 3
		corr_keys = F.cosine_similarity(self.keys.unsqueeze(1), self.keys.unsqueeze(0), dim=-1)
		corr_values = F.cosine_similarity(self.values.unsqueeze(1), self.values.unsqueeze(0), dim=-1)
		lmem_reg = 1/(self.M*(self.M - 1))*(torch.sum(corr_keys) + torch.sum(corr_values))

		loss = l2_exp + l2_vtx + 0.1 * lmem_reg
		self.log("loss", loss) 
		
		return loss

	def validation_step(self, batch, batch_idx):
		packed_audio_embed, packed_exp, packed_pose, packed_shape, packed_landmarks3d = batch
		# packed_audio_embed = packed_audio_embed.to(device)
		# packed_exp = packed_exp.to(device)
		# packed_pose = packed_pose.to(device)
		# packed_shape = packed_shape.to(device)
		# packed_landmarks3d = packed_landmarks3d.to(device)


		audio_embed, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_audio_embed, batch_first=True)
		exp, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_exp, batch_first=True)
		pose, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_pose, batch_first=True)
		shape, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_shape, batch_first=True)
		landmarks3d, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_landmarks3d, batch_first=True)
		
		# audio_embed, exp, pose, shape, landmarks3d = batch.to(device)
		print(f'val audio_embed.shape before forward pass: {audio_embed.shape}')
		exp_hat = self(packed_audio_embed)
		# exp_hat, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_exp_hat, batch_first=True)
		landmarks3d = torch.squeeze(landmarks3d, 2)	
		mse_loss = torch.nn.MSELoss()	
		l2_exp = mse_loss(exp_hat, exp)
		print(f'val pose.shape after forward pass and before get_Om: {pose.shape}')
		landmarks3d_hat = get_Om(pose, shape, exp_hat, self.emoca, batch_size=16)
		
		# landmarks3d_hat = landmarks3d
		print(f'val landmarks3d_hat.shape after get_Om(): {landmarks3d_hat.shape}')
		print(f'val landmarks3d.shape after self(exp): {landmarks3d.shape}')
		l2_vtx = mse_loss(landmarks3d_hat, landmarks3d) # dim(Om) = T × h_v × 3
		corr_keys = F.cosine_similarity(self.keys.unsqueeze(1), self.keys.unsqueeze(0), dim=-1)
		corr_values = F.cosine_similarity(self.values.unsqueeze(1), self.values.unsqueeze(0), dim=-1)
		lmem_reg = 1/(self.M*(self.M - 1))*(torch.sum(corr_keys) + torch.sum(corr_values))

		loss = l2_exp + l2_vtx + 0.1 * lmem_reg
		self.log("val_loss", loss)
		print(f'TRAINING LOOP LEFTOVERS:{torch.cuda.ipc_collect()}') 
		torch.cuda.ipc_collect()
		

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
		
		self.w_q = nn.Parameter(torch.randn(64, 64))
		self.w_k = nn.Parameter(torch.randn(64, 64))
		self.w_v = nn.Parameter(torch.randn(64, 64))
		self.w_o = nn.Parameter(torch.randn(1, 64))
		self.dropout = nn.Dropout(dropout)

	def forward(self, query):
		# if query is a frame of a sequence, but not a sequence, we need to rewrite that
		# query right now is a batch of sequences: [batch_size, max_seq_len, dim=64] and it's unpacked
		q_list_unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(query, batch_first=True)
		max_len = lengths.max()
		# - flatten the q_list into [batch_size*max_seq_len, dim]
		# lengths: [batch_size]
		# you can double check what's the shape of q_list_unpacked
		print(f'q_list_unpacked.shape in ImplicitMem: {q_list_unpacked.shape}')
		q_list_unpacked_flat = q_list_unpacked.reshape(-1, 64)	
		print(f'q_list_unpacked and reshaped into 2d matrix in ImplicitMem: {q_list_unpacked_flat.shape}')
		q = torch.matmul(q_list_unpacked_flat, self.w_q)
		k = torch.matmul(self.keys, self.w_k)
		v = torch.matmul(self.values, self.w_v)
		# - mask needs to be applied as well for scoresd
		# ? is k.transpose(-2, -1) same as k.T
		
		# ? how to get mask
		# - perhaps we need to use lengths
		# if we matmul padded tensors in q with k, it will zero out
		scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
		
		# and for softmax we do need mask, the mask will set the padded elements to a very large negative value
		mask = (scores != 0).float()
		attention_weights = torch.softmax(scores, dim=-1)
		attention = torch.matmul(attention_weights, v)
		output = attention * self.w_o
		output = self.dropout(attention)
		
		# do we need to unflatten the output of dim (q_len_flat, 64) back into batches?
		# batch_size = self.batch_size
		batch_size = 16
		print(f'output.shape inside ImplicitMem: {output.shape}')
		orig_shape_output = output.view(batch_size, max_len, 64)
		return orig_shape_output


class Encoder(nn.Module):
	def __init__(self):
		super().__init__()
		# change input accordingly to the size of the audio embedding 29 -> ?
		# Q1: should there be Relu between Linear and LayerNorm
		self.l1 = nn.Linear(392, 64)
		self.relu = nn.ReLU()
		self.layernorm = nn.LayerNorm(64)
		self.dropout = nn.Dropout()
		self.pos_encoding = DynamicPositionalEncoding()
	
	def forward(self, x):
		unpacked_data, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
		print(f'lengths: {lengths}')
		print(f'lengths.shape: {lengths.shape}')
		print(f'unpacked_data.shape: {unpacked_data.shape}')
		x = self.l1(unpacked_data)
		print(f'x.shape after l1: {x.shape}')
		x = self.relu(x)
		print(f'x.shape after relu: {x.shape}')
		x = self.layernorm(x)
		x = self.dropout(x)
		x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
		x = self.pos_encoding(x)
	
		return x


class DynamicPositionalEncoding(nn.Module):
	def __init__(self, d_model=64, dropout=0.1, max_len=5000):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.max_len = max_len
		self.d_model = d_model
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, d_model)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, packed_seq):
		# Extract the data and lengths from the PackedSequence
		data, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=True)
		
		# Add the positional encoding to the data tensor
		pos_enc = self.pe[:, :data.size(1)]
		pos_enc = pos_enc.to(device=data.device)
		data = data + pos_enc
		print(f'x.shape after encoding, but before encoder packing: {data.shape}')
		
		# Pack the data tensor back into a PackedSequence
		packed_seq = torch.nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=True)
		return packed_seq


class Decoder(nn.Module):
	def __init__(self, d_model=64, nhead=1, num_layers=2):
		# TransformerEncoder or ConformerEncoder
		super().__init__()
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.fc = nn.Linear(d_model, 50)
	def forward(self, x, sequence_lengths):
		# pack the padded sequences back into PackedSequence, since transformer encoder can operate on them
		# sorted_sequence_lengths, sorted_indices = torch.sort(sequence_lengths, descending=True)
		# sorted_padded_sequences = x[sorted_indices]
		# packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(sorted_padded_sequences, lengths=sorted_sequence_lengths, batch_first=True)
		x = self.transformer_encoder(x)
		# now we need to unpack the x, since fc only works on unpacked sequences
		# x, lenghts = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
		x = self.fc(x)
		# x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
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

		torch.cuda.empty_cache()
		# torch.multiprocessing.set_start_method('spawn')
		audio2exp = Audio2Exp()
		datamodule = Audio2ExpDataModule()
		datamodule.setup()
		train_dataloader = datamodule.train_dataloader()
		val_dataloader = datamodule.val_dataloader()
		
		# callbacks=[EarlyStopping(monitor="val_loss", mode="min")], 
		# fast_dev_run=True,
		# EarlyStopping(monitor="val_loss", mode="min", patience=20)
		trainer = pl.Trainer(strategy = DDPStrategy(find_unused_parameters=True), default_root_dir='checkpoints', callbacks=[checkpoint_callback], logger=wandb_logger, accelerator="gpu", devices=4)
		trainer.fit(audio2exp, train_dataloader, val_dataloader)











