import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pandas as pd
from torchvision.io import read_image

# based on AVSpeech dataset
class Audio2ExpDataset(Dataset):
		def __init__(self, audio_embed_dir : str, coeff_dir : str, transform=None):
				self.audio_embed_dir = audio_embed_dir
				self.coeff_dir = coeff_dir
				self.transform = transform

		def __len__(self):
				return len(os.listdir(self.audio_embed_dir))

		def __getitem__(self, idx):
				audio_embed_filepath = os.join(self.audio_embed_dir, os.listdir(self.audio_embed_dir)[idx])
				audio_embed = torch.load(audio_embed_filepath)
				coeff_folderpath = os.join(self.coeff_dir, os.listdir(self.coeff_dir)[idx])
				exp_coeff = torch.from_numpy(numpy.load(os.join(coeff_folderpath, 'exp.npy')))
				pose_coeff = torch.from_numpy(numpy.load(os.join(coeff_folderpath, 'pose.npy')))
				shape_coeff = torch.from_numpy(numpy.load(os.join(coeff_folderpath, 'shape.npy')))
				cam_coeff = torch.from_numpy(numpy.load(os.join(coeff_folderpath, 'cam.npy')))
				landmarks3d = torch.from_numpy(numpy.load(os.join(coeff_folderpath, 'landmarks3d.npy')))
				coeff = {'exp_coeff': exp_coeff, 'pose_coeff': pose_coeff,\
									'shape_coeff': shape_coeff, 'cam_coeff': cam_coeff, 'landmarks3d': landmarks3d}\
				return audio_embed, coeff


class Audio2ExpDataModule(pl.LightningDataModule):
		def __init__(self, audio_dir: str = '/mnt/sda/AVSpeech/audio_encodings', coeff_dir: str = '/mnt/sda/AVSpeech/coeffs', batch_size: int = 32):
				super().__init__()
				self.audio_dir = audio_dir
				self.coeff_dir = coeff_dir

		def setup(self, stage: str):
				if stage == 'fit':
						AVSpeech = Audio2ExpDataset(self.audio_dir, self.coeff_dir)
						self.train, self.val = random_split(AVSpeech, [0.9, 0.1])

		def train_dataloader(self):
				return DataLoader(self.train, batch_size=self.batch_size, num_workers=8)

		def val_dataloader(self):
				return DataLoader(self.val, batch_size=self.batch_size, num_workers=8)


# TODO: Neural Rendering dataset
class NeuralRenderingDataset(Dataset):
		def __init__(self, audio_embed_dir, video_dir, transform=None):
				self.audio_embed_dir = audio_embed_dir
				self.video_dir = video_dir

		def __len__(self):
				return len(os.listdir(self.audio_embed_dir))

		def __getitem__(self, idx):
				
				sample = {'audio_embed': audio_embed, 'exp_coeff': exp_coeff, 'pose_coeff': pose_coeff,\
									'shape_coeff': shape_coeff, 'cam_coeff': cam_coeff, 'landmarks3d': landmarks3d,\
									'orig_frame': orig_frame,'masked_frame': masked_frame, 'flame_frame': flame_frame}
				return sample

