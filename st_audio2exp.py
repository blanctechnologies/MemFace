# import os
# import json
# from typing import List
# import math
import streamlit as st
from loguru import logger
from pandas import Series

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
from lightning.pytorch.callbacks import LearningRateFinder
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from utils import get_Om


from datasets import Audio2ExpDataModule

from emoca.gdl_apps.EMOCA.utils.load import load_model

# from emoca.gdl.utils.FaceDetector import FAN
# from emoca.gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
# from emoca.gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test, decode

from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy

# import torchmetrics
# from torchmetrics.functional.pairwise import pairwise_cosine_similarity
# from torch.optim import Optimizer
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
    # PositionalEncoding2D,
    # PositionalEncoding3D,
    # Summer,
)

import warnings
logger.warning("* * * WARNING * * * All warnings suppressed!")
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()
wandb_logger = WandbLogger(name='Audio2Exp', project='MemFace')
pl.seed_everything(42, workers=True)
# torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

st.set_page_config(page_title="Audio2Exp", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
ST_HEADER = st.empty()
COL1, COL2 = st.columns([1, 3])
LOSS_REPORT = COL1.empty()
ENCODER_CONTAINER = COL2.empty()
COL2.divider()
TRAINING_CONTAINER = COL2.empty()
COL2.divider()
IMPLICITMEM_CONTAINER = COL2.empty()
COL2.divider()
FORWARD_CONTAINER = COL2.empty()


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

        self.loss_func_list = []

        self.list_l2_exp = []
        self.list_l2_vtx = []
        self.list_lmem_reg = []
        self.list_loss = []

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
        # st.write(f'packed_audio_embed in forward of Audio2Exp: {packed_audio_embed}')
        st_forward = FORWARD_CONTAINER.container()
        st_forward.subheader('Audio2Exp - forward step')
        st_forward.write('packed_audio_embed in forward of Audio2Exp:')
        st_forward.write(packed_audio_embed)
        audiofeature = self.encoder(packed_audio_embed)
        # st.write(f'audiofeature in forward of Audio2Exp: {audiofeature}')
        # encoded_audiofeature is packed, cause it's easier, we need to unpack it

        unpacked_audiofeature, lengths = torch.nn.utils.rnn.pad_packed_sequence(audiofeature, batch_first=True)
        # st.write(f'unpacked_audiofeature after encoding and padding packed seq: {unpacked_audiofeature.shape}')
        # implicitmem will unpack audiofeature and return unpacked result
        output = self.decoder(unpacked_audiofeature + self.implicitmem(audiofeature), lengths)
        return output

    # def L_reg(self, K, V):
    #       corr_K = torch.sum(pairwise_cosine_similarity(K))  # Use the metric
    #       corr_V = torch.sum(pairwise_cosine_similarity(V))  # Use the metric
    #       return (corr_K + corr_V)/(self.M * (self.M - 1))    # Average of the two correlations

    def training_step(self, batch, batch_idx):
        # during first half of training alterating the learning of memory vs other parameters
        # how the first half is determined?
        # maybe it's done manually, just comment out this block in some time, but still how
        # it's determined?
        st_training = TRAINING_CONTAINER.container()
        st_training.subheader('Audio2Exp - training step')
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
        audio_embed._unique_name = "audio_embed"
        exp, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_exp, batch_first=True)
        exp._unique_name = "exp"
        pose, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_pose, batch_first=True)
        pose._unique_name = "pose"
        shape, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_shape, batch_first=True)
        shape._unique_name = "shape"
        landmarks3d, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_landmarks3d, batch_first=True)
        landmarks3d._unique_name = "landmarks3d"
        batch_size = audio_embed.size(0)
        # compare landmarks3d from file and generated one.
        st_training.write(f'landmarks3d.shape: {landmarks3d.shape}')
        st_training.write(f'landmarks3d[0] from file: {landmarks3d[0]}')
        landmarks3d_generated = get_Om(pose, shape, exp, self.emoca, batch_size=batch_size)
        st_training.write('landmarks3d[0] generated from exp, pose,shape:')
        st_training.write(landmarks3d_generated[0])

        # audio_embed, exp, pose, shape, landmarks3d = batch.to(device)
        exp_hat = self.forward(packed_audio_embed)
        # exp_hat, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_exp_hat, batch_first=True)

        mse_loss = torch.nn.MSELoss()
        l2_exp = mse_loss(exp_hat, exp)
        # batch_size = 8
        # st.write(f'training loop pose.shape after forward pass and before get_Om: {pose.shape}')
        landmarks3d_hat = get_Om(pose, shape, exp_hat, self.emoca, batch_size=batch_size)
        # landmarks3d_hat = landmarks3d
        # st.write(f'training loop landmarks3d.shape before squeeze: {landmarks3d.shape}')
        landmarks3d = torch.squeeze(landmarks3d, 2)
        # st.write('---- after get_Om ----')
        # st.write(f'training loop landmarks3d.shape after squezze: {landmarks3d.shape}')
        # st.write(f'training loop landmarks3d_hat.shape: {landmarks3d_hat.shape}')

        l2_vtx = mse_loss(landmarks3d_hat, landmarks3d)  # dim(Om) = T × h_v × 3
        st_training.write(f'self.implicitmem.keys.shape: {self.implicitmem.keys.shape}')
        st_training.write(f'self.implicitmem.values.shape: {self.implicitmem.values.shape}')
        # corr_keys = F.cosine_similarity(self.implicitmem.keys.unsqueeze(1), self.implicitmem.keys.unsqueeze(0), dim=-1)
        # corr_values = F.cosine_similarity(self.implicitmem.values.unsqueeze(1), self.implicitmem.values.unsqueeze(0), dim=-1)
        corr_keys = sim_matrix(self.implicitmem.keys, self.implicitmem.keys)
        corr_values = sim_matrix(self.implicitmem.values, self.implicitmem.values)

        # corr_keys = F.cosine_similarity(self.implicitmem.keys.unsqueeze(1), self.implicitmem.keys.unsqueeze(0), dim=2)
        # corr_values = F.cosine_similarity(self.implicitmem.values.unsqueeze(1), self.implicitmem.values.unsqueeze(0), dim=2)
        st_training.write(f'corr_keys.shape: {corr_keys.shape}')
        st_training.write(f'corr_values.shape: {corr_values.shape}')
        # corr_keys = pairwise_cosine_similarity(self.implicitmem.keys, self.implicitmem.keys)
        # corr_values = pairwise_cosine_similarity(self.implicitmem.values, self.implicitmem.values)
        lmem_reg = 1 / (self.M * (self.M - 1)) * (torch.sum(corr_keys) + torch.sum(corr_values))
        # lmem_reg = self.L_reg(self.implicitmem.keys, self.implicitmem.values)
        # K = self.implicitmem.keys
        # V = self.implicitmem.values
        # lmem_reg = (torch.sum(pairwise_cosine_similarity(K)) + torch.sum(pairwise_cosine_similarity(V)))/(self.M * (self.M - 1))# Use the metric
        # corr_V = torch.sum(pairwise_cosine_similarity(V))  # Use the metric
        # (corr_K + corr_V)/(self.M * (self.M - 1))  # Average of the two correlations
        st_training.write(f'lmem_reg:{lmem_reg}')
        # lmem_reg.backward()

        loss = l2_exp + 10 * l2_vtx + lmem_reg
        self.log("l2_exp", l2_exp)
        self.log("l2_vtx", l2_vtx)
        self.log("lmem_reg", lmem_reg)
        self.log("loss", loss)

        self.list_l2_exp.append(float(l2_exp.cpu()))
        self.list_l2_vtx.append(float(l2_vtx.cpu()))
        self.list_lmem_reg.append(float(lmem_reg.cpu()))
        self.list_loss.append(float(loss.cpu()))

        return loss

    def validation_step(self, batch, batch_idx):
        st_validation = LOSS_REPORT.container()
        st_validation.subheader("Loss function")
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
        exp_hat = self.forward(packed_audio_embed)
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
        st_validation.write(f'corr_keys.shape: {corr_keys.shape}')
        st_validation.write(f'corr_values.shape: {corr_values.shape}')
        # corr_keys = pairwise_cosine_similarity(self.implicitmem.keys, self.implicitmem.keys)
        # corr_values = pairwise_cosine_similarity(self.implicitmem.values, self.implicitmem.values)
        lmem_reg = 1 / (self.M * (self.M - 1)) * (torch.sum(corr_keys) + torch.sum(corr_values))
        # lmem_reg.backward()
        st_validation.write(f'lmem_reg:{lmem_reg}')
        val_loss = l2_exp + 10 * l2_vtx + lmem_reg
        self.loss_func_list.append(float(val_loss.cpu()))
        st_validation.line_chart(self.loss_func_list)
        self.log("val_loss", val_loss)

        st_validation.write('Graph - l2_exp')
        st_validation.line_chart(self.list_l2_exp)
        
        st_validation.write('Graph - list_l2_vtx')
        st_validation.line_chart(self.list_l2_exp)
        
        st_validation.write('Graph - list_lmem_reg')
        st_validation.line_chart(self.list_l2_exp)
        
        st_validation.write('Graph - list_loss')
        st_validation.line_chart(self.list_l2_exp)

        torch.cuda.ipc_collect()

        return val_loss

    def configure_optimizers(self):
        # 1e-4 training, 5e-6 adaptation(200 epoch)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
        # if self.training_mode == 'fit':
        #        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # elif self.training_mode == 'adaptation':
        #           optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
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
        st_forward = IMPLICITMEM_CONTAINER.container()
        st_forward.subheader('ImplicitMem - forward')
        # if query is a frame of a sequence, but not a sequence, we need to rewrite that
        # query right now is a batch of sequences: [batch_size, max_seq_len, dim=64] and it's unpacked
        q_list_unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(query, batch_first=True)
        
        max_len = lengths.max()
        # - flatten the q_list into [batch_size*max_seq_len, dim]
        # lengths: [batch_size]
        # you can double check what's the shape of q_list_unpacked
        st_forward.write(f'q_list_unpacked.shape in ImplicitMem: {q_list_unpacked.shape}')
        q_list_unpacked_flat = q_list_unpacked.reshape(-1, 64)
        st_forward.write(f'q_list_unpacked and reshaped into 2d matrix in ImplicitMem: {q_list_unpacked_flat.shape}')
        # q = torch.matmul(q_list_unpacked_flat, self.w_q)
        q = self.w_q(q_list_unpacked_flat)
        # st.write(f'q.shape after q = self.w_q(q_list_unpacked_flat): {q.shape}')
        # k = self.keys(self.w_k)
        # v = self.values(self.w_v)
        # k = torch.matmul(self.keys, self.w_k)
        k = self.w_k(self.keys)
        st_forward.write(f'k.shape after k = self.w_k(self.keys): {k.shape}')
        # v = torch.matmul(self.values, self.w_v)
        # v = self.w_v(self.values)
        # st.write(f'v.shape after v = self.w_v(self.values): {v.shape}')
        # - mask needs to be applied as well for scoresd
        # ? is k.transpose(-2, -1) same as k.T

        # ? how to get mask
        # - perhaps we need to use lengths
        # if we matmul padded tensors in q with k, it will zero out
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        st_forward.write(f'scores.shape: {scores.shape}')
        # and for softmax we do need mask, the mask will set the padded elements to a very large negative value
        # mask = (scores != 0).float()
        attention_weights = torch.softmax(scores, dim=-1)
        st_forward.write(f'attention_weights: {attention_weights.shape}')
        # attention = torch.matmul(attention_weights, v)

        attention = torch.matmul(attention_weights, self.values)
        attention = self.w_v(attention)
        output = self.w_o(attention)
        # output = attention * self.w_o
        st_forward.write(f'attention.shape after attention = torch.matmul(attention_weights, v): {attention.shape}')
        attention = self.w_o(attention)
        st_forward.write(f'attention.shape after attention = self.w_o(attention): {attention.shape}')
        output = self.dropout(attention)
        st_forward.write(f'output.shape after output = self.dropout(attention): {output.shape}')

        # do we need to unflatten the output of dim (q_len_flat, 64) back into batches?
        # st.write(f'q_list_unpacked.shape: {q_list_unpacked.shape}')
        # st.write(f'output.shape inside ImplicitMem: {output.shape}')
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
        st_encoder = ENCODER_CONTAINER.container()
        st_encoder.subheader('Encoder — forward')
        st_encoder.write(f'lengths: {lengths}')
        st_encoder.write(f'lengths.shape: {lengths.shape}')
        st_encoder.write(f'unpacked_data.shape: {unpacked_data.shape}')
        x = self.l1(unpacked_data)
        st_encoder.write(f'x.shape after l1: {x.shape}')
        x = self.relu(x)
        st_encoder.write(f'x.shape after relu: {x.shape}')
        x = self.layernorm(x)
        x = self.dropout(x)
        st_encoder.write(f'x.shape before packing and pos_encoding: {x.shape}')
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


class FineTuneLearningRateFinder(LearningRateFinder):
    """Fine tuning for non-linear loss funcition,
    when the loss function become too small or too large, we could adjust it"""
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


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
        pass
        # ** DEBUG ** Skip for now
        # st.write(name, param.requires_grad)
        # ^^^^^ * * * * * * * 
    # plugin = DDPPlugin(find_unused_parameters=True)
    # plugin = DDPStrategy()
    auto_lr_finder = FineTuneLearningRateFinder(milestones=(5, 10))
    trainer = pl.Trainer(
        # auto_lr_find=True,
        gradient_clip_val=0.5,
        max_epochs=412,
        # plugins=plugin,
        # strategy="ddp",
        strategy=DDPStrategy(find_unused_parameters=True),
        # callbacks=[checkpoint_callback, swa_callback, auto_lr_finder],
        callbacks=[checkpoint_callback, swa_callback],  # Let disable Learning Rate Finder until we made model learning okay
        default_root_dir='checkpoints',
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
    )

    # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(audio2exp)
    # st.write(f'lr finder results: {lr_finder.results}')

    # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # new_lr = lr_finder.suggestion()
    # audio2exp.hparams.lr = new_lr

    ST_HEADER.write("✅  Start training!")
    trainer.fit(audio2exp, train_dataloader, val_dataloader)
