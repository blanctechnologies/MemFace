import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np


# based on AVSpeech dataset
class Audio2ExpDataset(Dataset):
		def __init__(self, audio_embed_dir : str, coeff_dir : str, transform=None):
				self.audio_embed_dir = audio_embed_dir
				self.coeff_dir = coeff_dir
				self.transform = transform

		def __len__(self):
				return len(os.listdir(self.audio_embed_dir))

		def __getitem__(self, idx):
				audio_embed_filepath = os.path.join(self.audio_embed_dir, os.listdir(self.audio_embed_dir)[idx])
				audio_embed = torch.load(audio_embed_filepath)
				print(f'audio embedding:  {audio_embed[1]}')
				print(f'audio embedding len: {len(audio_embed)}')
				print(f'audio embedding [0] shape: {audio_embed[0].shape}')
				print(f'audio embedding [1] shape: {audio_embed[1].shape}')
			
				
				coeff_folderpath = os.path.join(self.coeff_dir, os.listdir(self.coeff_dir)[idx])
				exp	= []
				pose = []
				shape = []
				Om = []

				for i, frame_name in enumerate(os.listdir(coeff_folderpath)):
						exp.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'exp.npy'))))
						pose.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'pose.npy'))))
						shape.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'shape.npy'))))
						Om.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'landmarks3d.npy'))))
						
				coeff = {'exp_coeff': torch.stack(exp), 'pose_coeff': torch.stack(pose),\
									'shape_coeff': torch.stack(shape), 'landmarks3d': torch.stack(Om)}
				return audio_embed, coeff


class Audio2ExpDataModule(pl.LightningDataModule):
		def __init__(self, audio_dir: str = '/mnt/sda/AVSpeech/audio_encodings_test', coeff_dir: str = '/mnt/sda/AVSpeech/video_test', batch_size: int = 32):
				super().__init__()
				self.audio_dir = audio_dir
				self.coeff_dir = coeff_dir

		def setup(self, stage = 'fit'):
				if stage == 'fit':
						AVSpeech = Audio2ExpDataset(self.audio_dir, self.coeff_dir)
						# self.train, self.val = random_split(AVSpeech, [0.7, 0.3])
						self.train = AVSpeech

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


if __name__ == '__main__':
		datamodule = Audio2ExpDataModule()
		datamodule.setup()
		# print(f'len: {len(datamodule.train)}')
		print(f'element #0: {datamodule.train.__getitem__(1)}')
