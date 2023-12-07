# import os
# import json
# from typing import List
# import math

import torch
# import hydra
# from torch.cuda.amp import autocast
from torch import nn
# from torch import optim, nn, utils, Tensor
# import torch.nn.functional as F
# from torchmetrics.functional import pairwise_cosine_similarity
# from torchvision.transforms import ToTensor
import pytorch_lightning as pl
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from utils import get_Om


from datasets import Audio2ExpDataModule

from emoca.gdl_apps.EMOCA.utils.load import load_model
# from emoca.gdl.utils.FaceDetector import FAN
from emoca.gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
from emoca.gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test, decode

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

import torchmetrics
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from positional_encodings.torch_encodings import (
		 PositionalEncoding1D,
		 PositionalEncoding2D,
		 PositionalEncoding3D,
		 Summer,
)

torch.cuda.empty_cache()
wandb_logger = WandbLogger(name='Audio2Exp', project='MemFace')
pl.seed_everything(42, workers=True)
# torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Audio2Exp(pl.LightningModule):
		def __init__(self, learning_rate):
				super().__init__()
				# self.save_hyperparameters()
				# self.keys = nn.Embedding(M, d_k)
				# self.values = nn.Embedding(M, d_v)

				# self.keys = torch.nn.Parameter(torch.randn(1000, 64))
				# self.values = torch.nn.Parameter(torch.randn(1000, 64))
				# self.keys = nn.Linear(1000, 64)
				# self.values = nn.Linear(1000, 64)
				self.M = 1000  # number of keys and values => output of f_enc is 1000
				self.d_k = 64
				self.d_v = 64
				self.learning_rate = learning_rate
				self.training_mode = 'fit'
				self.encoder = Encoder()
				# self.implicitmem = ImplicitMem(self.keys, self.values)
				self.implicitmem = ImplicitMem()
				self.decoder = Decoder()

				# emoca initialization

				path_to_models = "/home/avocoral/MemFace/emoca/assets/EMOCA/models"
				model_name = 'EMOCA_v2_lr_mse_20'
				mode = 'detail'

				self.emoca, conf = load_model(path_to_models, model_name, mode)
				self.emoca.cuda()
				self.emoca.eval()
				for param in self.emoca.parameters():
						param.requires_grad = False

		def forward(self, packed_audio_embed):
				print(f'packed_audio_embed in forward of Audio2Exp: {packed_audio_embed}')
				audiofeature = self.encoder(packed_audio_embed)
				print(f'audiofeature in forward of Audio2Exp: {audiofeature}')
				# encoded_audiofeature is packed, cause it's easier, we need to unpack it

				unpacked_audiofeature, lengths = torch.nn.utils.rnn.pad_packed_sequence(audiofeature, batch_first=True)
				print(f'unpacked_audiofeature after encoding and padding packed seq: {unpacked_audiofeature.shape}')
				# implicitmem will unpack audiofeature and return unpacked result
				output = self.decoder(unpacked_audiofeature + self.implicitmem(audiofeature), lengths)
				return output

		# def L_reg(self, K, V):
		#		corr_K = torch.sum(pairwise_cosine_similarity(K))  # Use the metric
		#		corr_V = torch.sum(pairwise_cosine_similarity(V))  # Use the metric
		#		return (corr_K + corr_V)/(self.M * (self.M - 1))	# Average of the two correlations

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
				batch_size = audio_embed.size(0)
				# compare landmarks3d from file and generated one.
				print(f'landmarks3d.shape: {landmarks3d.shape}')
				print(f'landmarks3d[0] from file: {landmarks3d[0]}')
				landmarks3d_generated = get_Om(pose, shape, exp, self.emoca, batch_size=batch_size)
				print(f'landmarks3d[0] generated from exp, pose,shape: {landmarks3d_generated[0]}')

				# audio_embed, exp, pose, shape, landmarks3d = batch.to(device)
				exp_hat = self.forward(packed_audio_embed)
				# exp_hat, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_exp_hat, batch_first=True)

				mse_loss = torch.nn.MSELoss()
				l2_exp = mse_loss(exp_hat, exp)
				# batch_size = 8
				# print(f'training loop pose.shape after forward pass and before get_Om: {pose.shape}')
				landmarks3d_hat = get_Om(pose, shape, exp_hat, self.emoca, batch_size=batch_size)
				# landmarks3d_hat = landmarks3d
				# print(f'training loop landmarks3d.shape before squeeze: {landmarks3d.shape}')
				landmarks3d = torch.squeeze(landmarks3d, 2)
				# print('---- after get_Om ----')
				# print(f'training loop landmarks3d.shape after squezze: {landmarks3d.shape}')
				# print(f'training loop landmarks3d_hat.shape: {landmarks3d_hat.shape}')

				l2_vtx = mse_loss(landmarks3d_hat, landmarks3d)  # dim(Om) = T × h_v × 3
				print(f'self.implicitmem.keys.shape: {self.implicitmem.keys.shape}')
				print(f'self.implicitmem.values.shape: {self.implicitmem.values.shape}')
				# corr_keys = F.cosine_similarity(self.implicitmem.keys.unsqueeze(1), self.implicitmem.keys.unsqueeze(0), dim=-1)
				# corr_values = F.cosine_similarity(self.implicitmem.values.unsqueeze(1), self.implicitmem.values.unsqueeze(0), dim=-1)
				corr_keys = sim_matrix(self.implicitmem.keys, self.implicitmem.keys)
				corr_values = sim_matrix(self.implicitmem.values, self.implicitmem.values)

				# corr_keys = F.cosine_similarity(self.implicitmem.keys.unsqueeze(1), self.implicitmem.keys.unsqueeze(0), dim=2)
				# corr_values = F.cosine_similarity(self.implicitmem.values.unsqueeze(1), self.implicitmem.values.unsqueeze(0), dim=2)
				print(f'corr_keys.shape: {corr_keys.shape}')
				print(f'corr_values.shape: {corr_values.shape}')
				# corr_keys = pairwise_cosine_similarity(self.implicitmem.keys, self.implicitmem.keys)
				# corr_values = pairwise_cosine_similarity(self.implicitmem.values, self.implicitmem.values)
				lmem_reg = 1 / (self.M * (self.M - 1)) * (torch.sum(corr_keys) + torch.sum(corr_values))
				# lmem_reg = self.L_reg(self.implicitmem.keys, self.implicitmem.values)
				# K = self.implicitmem.keys
				# V = self.implicitmem.values
				# lmem_reg = (torch.sum(pairwise_cosine_similarity(K)) + torch.sum(pairwise_cosine_similarity(V)))/(self.M * (self.M - 1))# Use the metric
				# corr_V = torch.sum(pairwise_cosine_similarity(V))  # Use the metric
				# (corr_K + corr_V)/(self.M * (self.M - 1))  # Average of the two correlations
				print(f'lmem_reg:{lmem_reg}')
				# lmem_reg.backward()

				loss = l2_exp + 10 * l2_vtx + lmem_reg
				self.log("l2_exp", l2_exp)
				self.log("l2_vtx", l2_vtx)
				self.log("lmem_reg", lmem_reg)
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
				# batch_size = 8
				batch_size = audio_embed.size(0)
				# audio_embed, exp, pose, shape, landmarks3d = batch.to(device)
				exp_hat = self(packed_audio_embed)
				# exp_hat, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_exp_hat, batch_first=True)
				landmarks3d = torch.squeeze(landmarks3d, 2)
				mse_loss = torch.nn.MSELoss()
				l2_exp = mse_loss(exp_hat, exp)
				landmarks3d_hat = get_Om(pose, shape, exp_hat, self.emoca, batch_size=batch_size)

				# landmarks3d_hat = landmarks3d
				l2_vtx = mse_loss(landmarks3d_hat, landmarks3d)  # dim(Om) = T × h_v × 3
				# corr_keys = F.cosine_similarity(self.keys.unsqueeze(1), self.keys.unsqueeze(0), dim=-1)
				# corr_values = F.cosine_similarity(self.values.unsqueeze(1), self.values.unsqueeze(0), dim=-1)
				# lmem_reg = 1/(self.M*(self.M - 1))*(torch.sum(corr_keys) + torch.sum(corr_values))
				corr_keys = sim_matrix(self.implicitmem.keys, self.implicitmem.keys)
				corr_values = sim_matrix(self.implicitmem.values, self.implicitmem.values)
				# corr_keys = F.cosine_similarity(self.implicitmem.keys.unsqueeze(1), self.implicitmem.keys.unsqueeze(0), dim=2)
				# corr_values = F.cosine_similarity(self.implicitmem.values.unsqueeze(1), self.implicitmem.values.unsqueeze(0), dim=2)
				print(f'corr_keys.shape: {corr_keys.shape}')
				print(f'corr_values.shape: {corr_values.shape}')
				# corr_keys = pairwise_cosine_similarity(self.implicitmem.keys, self.implicitmem.keys)
				# corr_values = pairwise_cosine_similarity(self.implicitmem.values, self.implicitmem.values)
				lmem_reg = 1 / (self.M * (self.M - 1)) * (torch.sum(corr_keys) + torch.sum(corr_values))
				# lmem_reg.backward()
				print(f'lmem_reg:{lmem_reg}')
				val_loss = l2_exp + 10 * l2_vtx + lmem_reg
				self.log("val_loss", val_loss)
				torch.cuda.ipc_collect()

				return val_loss

		def configure_optimizers(self):
				# 1e-4 training, 5e-6 adaptation(200 epoch)
				# optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
				# if self.training_mode == 'fit':
				#		 optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
				# elif self.training_mode == 'adaptation':
				#			optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
				optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
				return optimizer

		def train_dataloader(self):
				datamodule = Audio2ExpDataModule()
				datamodule.setup()
				train_dataloader = datamodule.train_dataloader()
				return train_dataloader



class ImplicitMem(nn.Module):
		def __init__(self, d_model=64, d_k=64, d_v=64, dropout=0.1, M=1000):
				super().__init__()

				self.keys = nn.Parameter(torch.randn(1000, 64), requires_grad=True)
				self.values = nn.Parameter(torch.randn(1000, 64), requires_grad=True)
				self.batch_size = 16
				# self.keys = keys
				# self.values = values
				self.d_k = d_k
				self.d_v = d_v

				self.w_q = nn.Linear(in_features=64, out_features=1)
				self.w_k = nn.Linear(in_features=64, out_features=1)
				self.w_v = nn.Linear(in_features=64, out_features=1)
				self.w_o = nn.Linear(in_features=1, out_features=64)

				# self.w_q = nn.Parameter(torch.randn(64, 64), requires_grad = True)
				# self.w_k = nn.Parameter(torch.randn(64, 64), requires_grad = True)
				# self.w_v = nn.Parameter(torch.randn(64, 64), requires_grad = True)
				# self.w_o = nn.Parameter(torch.randn(1, 64), requires_grad = True)
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
				# q = torch.matmul(q_list_unpacked_flat, self.w_q)
				q = self.w_q(q_list_unpacked_flat)
				# print(f'q.shape after q = self.w_q(q_list_unpacked_flat): {q.shape}')
				# k = self.keys(self.w_k)
				# v = self.values(self.w_v)
				# k = torch.matmul(self.keys, self.w_k)
				k = self.w_k(self.keys)
				print(f'k.shape after k = self.w_k(self.keys): {k.shape}')
				# v = torch.matmul(self.values, self.w_v)
				# v = self.w_v(self.values)
				# print(f'v.shape after v = self.w_v(self.values): {v.shape}')
				# - mask needs to be applied as well for scoresd
				# ? is k.transpose(-2, -1) same as k.T

				# ? how to get mask
				# - perhaps we need to use lengths
				# if we matmul padded tensors in q with k, it will zero out
				scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
				print(f'scores.shape: {scores.shape}')
				# and for softmax we do need mask, the mask will set the padded elements to a very large negative value
				# mask = (scores != 0).float()
				attention_weights = torch.softmax(scores, dim=-1)
				print(f'attention_weights: {attention_weights.shape}')
				# attention = torch.matmul(attention_weights, v)

				attention = torch.matmul(attention_weights, self.values)
				attention = self.w_v(attention)
				output = self.w_o(attention)
				# output = attention * self.w_o
				print(f'attention.shape after attention = torch.matmul(attention_weights, v): {attention.shape}')
				attention = self.w_o(attention)
				print(f'attention.shape after attention = self.w_o(attention): {attention.shape}')
				output = self.dropout(attention)
				print(f'output.shape after output = self.dropout(attention): {output.shape}')

				# do we need to unflatten the output of dim (q_len_flat, 64) back into batches?
				# print(f'q_list_unpacked.shape: {q_list_unpacked.shape}')
				# print(f'output.shape inside ImplicitMem: {output.shape}')
				orig_shape_output = output.view(self.batch_size, max_len, 64)
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
				# self.pos_encoding = DynamicPositionalEncoding()
				self.pos_encoding = PositionalEncoding1D(64)

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
				print(f'x.shape before packing and pos_encoding: {x.shape}')
				x = self.pos_encoding(x)
				x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
				# x = self.pos_encoding(x, lengths)

				return x


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


def sim_matrix(a, b, eps=1e-8):
		"""
		added eps for numerical stability
		"""
		a.requires_grad = True
		b.requires_grad = True

		a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
		a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
		b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
		sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
		return sim_mt


if __name__ == '__main__':
		d_k = 64
		d_v = 64
		M = 1000
		torch.multiprocessing.set_start_method('spawn')
		audio2exp = Audio2Exp(learning_rate=1e-4)
		datamodule = Audio2ExpDataModule()
		datamodule.setup()
		train_dataloader = datamodule.train_dataloader()
		val_dataloader = datamodule.val_dataloader()
		checkpoint_callback = ModelCheckpoint(
				dirpath="/home/avocoral/MemFace/checkpoints",
				filename="model-{epoch:02d}",
				monitor="val_loss",
				mode="min",
				save_top_k=1,  # Save the best model
		)
		swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

		for name, param in audio2exp.named_parameters():
				print(name, param.requires_grad)
		plugin = DDPPlugin(find_unused_parameters=True)
		trainer = pl.Trainer(
				auto_lr_find=True,
				gradient_clip_val=0.5,
				max_epochs=999,
				plugins=plugin,
				callbacks=[checkpoint_callback, swa_callback],
				default_root_dir='checkpoints',
				logger=wandb_logger,
				accelerator="gpu",
				devices=1,
		)
		
		# Run learning rate finder
		lr_finder = trainer.tuner.lr_find(audio2exp)
		print(f'lr finder results: {lr_finder.results}')


		# Plot with
		# fig = lr_finder.plot(suggest=True)
		# fig.show()
		new_lr = lr_finder.suggestion()
		audio2exp.hparams.lr = new_lr

		trainer.fit(audio2exp, train_dataloader, val_dataloader)
