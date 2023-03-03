import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
import torch.nn.functional as F

# based on AVSpeech dataset
class Audio2ExpDataset(Dataset):
		def __init__(self, audio_embed_dir : str, coeff_dir : str, transform=None):
				self.audio_embed_dir = audio_embed_dir
				self.coeff_dir = coeff_dir
				self.transform = transform

		def __len__(self):
				return len(os.listdir(self.coeff_dir))

		def __getitem__(self, idx):
				print(f'ifx: {idx}')
				sample_name = os.listdir(self.coeff_dir)[idx]
				audio_embed_filepath = os.path.join(self.audio_embed_dir, sample_name+'.pt')
				audio_embed = torch.cat(torch.load(audio_embed_filepath), dim=0)
				print(f'audio embedding name: {audio_embed_filepath}')
				# print(f'foldername = {sample_name[:-3]}')
				coeff_folderpath = os.path.join(self.coeff_dir, sample_name)
				# print(f'coeff_folderpath: {coeff_folderpath}')

				# print(f'audio_embed shape: {audio_embed.shape}')
				# print(f'audio_embed: {audio_embed}')	
								
				exp	= []
				pose = []
				shape = []
				Om = []

				for i, frame_name in enumerate(os.listdir(coeff_folderpath)):
						exp.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'exp.npy'))))
						pose.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'pose.npy'))))
						shape.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'shape.npy'))))
						Om.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'landmarks3d.npy'))))
						
				exp = torch.stack(exp)
				pose = torch.stack(pose)
				shape = torch.stack(shape)
				landmarks3d = torch.stack(Om)
				# coeff = torch.cat(torch.stack(exp), torch.stack(pose), torch.stack(shape), torch.squeeze(torch.stack(Om)))
				video_frames_num = exp.shape[0]
				
				# print(f'video_frames_num: {video_frames_num}')
				audio_embed = F.interpolate(audio_embed.T.unsqueeze(0), size=[video_frames_num], mode='nearest').squeeze(0).T

				# print(f'new audio_embed shape orig: {audio_embed.shape}')
				# print(f"exp_coeff shape orig: {exp.shape}")
				# print(f"pose_coeff shape orig: {pose.shape}")
				# print(f"shape_coeff shape orig: {shape.shape}")
				# print(f"landmarks3d shape orig: {landmarks3d.shape}")
				return audio_embed, exp, pose, shape, landmarks3d


class Audio2ExpDataModule(pl.LightningDataModule):
		def __init__(self, audio_dir: str = '/mnt/sda/AVSpeech/audio_encodings', coeff_dir: str = '/mnt/sda/AVSpeech/video', batch_size: int = 1):
				super().__init__()
				self.audio_dir = audio_dir
				self.coeff_dir = coeff_dir
				self.batch_size = batch_size

		def setup(self, stage = 'fit'):
				if stage == 'fit':
						AVSpeech = Audio2ExpDataset(self.audio_dir, self.coeff_dir)
						proportions = [.85, .15]
						lengths = [int(p * len(AVSpeech)) for p in proportions]
						lengths[-1] = len(AVSpeech) - sum(lengths[:-1])
						self.train, self.val = random_split(AVSpeech, lengths)
						# self.train = AVSpeech

		def train_dataloader(self):
				return DataLoader(self.train, batch_size=self.batch_size, num_workers=16, collate_fn=self.collate_fn)

		def val_dataloader(self):
				return DataLoader(self.val, batch_size=self.batch_size, num_workers=16, collate_fn=self.collate_fn)
		
		def collate_fn(self, batch):
				batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

				audio_embed, exp, pose, shape, landmarks3d = zip(*batch)
				audio_embed_lengths = [len(seq) for seq in audio_embed]
				exp_lengths = [len(seq) for seq in exp]
				pose_lengths = [len(seq) for seq in pose]
				shape_lengths = [len(seq) for seq in shape]
				landmarks3d_lengths = [len(seq) for seq in landmarks3d]
				# print(f"audio_embed_len[0]: {audio_embed_lengths[0]}")
				# print(f"exp_lengths[0]: {exp_lengths[0]}")
				# print(f"pose_lengths[0]: {pose_lengths[0]}")
				# print(f"shape_lengths[0]: {shape_lengths[0]}")
				# print(f"landmarks3d_lengths[0]: {landmarks3d_lengths[0]}")

				padded_audio_embed = torch.nn.utils.rnn.pad_sequence(audio_embed, batch_first=True)
				padded_exp = torch.nn.utils.rnn.pad_sequence(exp, batch_first=True)
				padded_pose = torch.nn.utils.rnn.pad_sequence(pose, batch_first=True)
				padded_shape = torch.nn.utils.rnn.pad_sequence(shape, batch_first=True)
				padded_landmarks3d = torch.nn.utils.rnn.pad_sequence(landmarks3d, batch_first=True)
				
				device = 'cuda:0'

				packed_audio_embed = torch.nn.utils.rnn.pack_padded_sequence(padded_audio_embed, audio_embed_lengths, batch_first=True, enforce_sorted=False).to(device)
				packed_exp = torch.nn.utils.rnn.pack_padded_sequence(padded_exp, exp_lengths, batch_first=True, enforce_sorted=False).to(device)
				packed_pose = torch.nn.utils.rnn.pack_padded_sequence(padded_pose, pose_lengths, batch_first=True, enforce_sorted=False).to(device)
				packed_shape = torch.nn.utils.rnn.pack_padded_sequence(padded_shape, shape_lengths, batch_first=True, enforce_sorted=False).to(device)
				packed_landmarks3d = torch.nn.utils.rnn.pack_padded_sequence(padded_landmarks3d, landmarks3d_lengths, batch_first=True, enforce_sorted=False).to(device)

				# padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
				# mask = (padded_batch != 0)
				# packed_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_batch, mask.sum(1), batch_first=True, enforce_sorted=False)
				return packed_audio_embed, packed_exp, packed_pose, packed_shape, packed_landmarks3d, audio_embed_lengths

if __name__ == '__main__':
		datamodule = Audio2ExpDataModule()
		datamodule.setup()
		device = 'cuda:0'
		# print(f'len: {len(datamodule.train)}')
		train_loader = datamodule.train_dataloader()	
		
		first_batch = next(iter(train_loader))
		packed_audio_embed, packed_exp, packed_pose, packed_shape, packed_landmarks3d, sequence_lengths = first_batch

		audio_embed, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_audio_embed, batch_first=True)
		exp, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_exp, batch_first=True)
		pose, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_pose, batch_first=True)
		shape, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_shape, batch_first=True)
		landmarks3d, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_landmarks3d, batch_first=True)

		
		print(f'sequence_lengths: {sequence_lengths}')
		

		# take the 2nd sample from batch, crop the padding | double check the indexing !
		# audio_embed = audio_embed[1][:int(audio_embed_lengths[0])]
		# exp = exp[1][:int(exp_lengths[0])]
		# pose = pose[1][:int(pose_lengths[0])]
		# shape = shape[1][:int(shape_lengths[0])]
		# landmarks3d = landmarks3d[1][:int(landmarks3d_lengths[0])]
		unpacked_audio_embed = [seq[:seq_len] for seq, seq_len in zip(audio_embed, sequence_lengths)]
		unpacked_exp = [seq[:seq_len] for seq, seq_len in zip(exp, sequence_lengths)]
		unpacked_pose = [seq[:seq_len] for seq, seq_len in zip(pose, sequence_lengths)]
		unpacked_shape = [seq[:seq_len] for seq, seq_len in zip(shape, sequence_lengths)]
		unpacked_landmarks3d = [seq[:seq_len] for seq, seq_len in zip(landmarks3d, sequence_lengths)]
		i = 2
		print('-------- AFTER SLICING ----------')
		print(f"audio_embed shape: {unpacked_audio_embed[i].shape}")
		print(f"exp shape: {unpacked_exp[i].shape}")
		print(f"pose shape: {unpacked_pose[i].shape}")
		print(f"shape shape: {unpacked_shape[i].shape}")
		print(f"landmarks3d shape: {unpacked_landmarks3d[i].shape}")
