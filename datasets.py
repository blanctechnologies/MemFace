import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import pickle

class NeuralRenderingDataset(Dataset):
		def __init__(self, data_dir='/home/avocoral/Downloads/Obamaset/Obama_vid/Obama', metadata_dir='/home/avocoral/Downloads/Obamaset/Obama_meta'):
				self.data_dir = data_dir
				self.metadata_dir = metadata_dir
				self.resolution = 450

		def __len__(self):
				# -25 frames, cause we don't take any of the last 25 frames as first frame of seq
				return len(os.listdir(self.data_dir)) - 26

		def __getitem__(self, idx):
				# one elem is 25 continious frames of ref video + masked original
				
				frames_dir = [str(idx+i).zfill(6)+'_000' for i in range(1, 26)]
				# print(f'idx: {idx}')
				# print(f'print dataset len: {self.__len__()}')
				# print(f'frames_dir: {frames_dir}')
				orig_images = []
				landmarks3d_final = []
				masked_ref_images = []
				print(f'len(frames_dir): {len(frames_dir)}')
				for i, frame_dir in enumerate(frames_dir):

					img_path = os.path.join(self.metadata_dir, 'cropped_frames', f'{frame_dir[:-4]}.png')
					# print(f'img_path:{img_path}')
					mask_path = os.path.join(self.data_dir, frame_dir, 'mask.png')
					ref_path = os.path.join('/home/avocoral/MemFace/test_folder', f'geometry_detail_{frame_dir}.png')
					# print(f'ref_path: {ref_path}')
					landmarks3d_path = os.path.join(self.data_dir, frame_dir, 'landmarks3d.npy')
					# Check if the files exists
					if not os.path.exists(landmarks3d_path):
									raise FileNotFoundError(f"Some files are missing for frame {frame_dir}")
					landmarks2d_path = os.path.join(self.metadata_dir, f'new_landmarks/{frame_dir}.pkl')
					# load landmarks3d
					landmarks3d = torch.from_numpy(np.load(landmarks3d_path))[:, 48:, :]
					
					# masking face
					original_image = cv2.imread(img_path)[:450, :450, :]
					# print(f'original_image.shape: {original_image.shape}')
					ref_image = cv2.imread(ref_path)[:450, :450, :]
					# print(f'ref_image.shape: {ref_image.shape}')
					mask_image = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
					# print(f'mask_image.shape: {mask_image.shape}')

					# Convert the mask image to binary
					_, binary_mask = cv2.threshold(mask_image, thresh=1, maxval=255, type=cv2.THRESH_BINARY)

					# Invert the binary mask image
					inverted_mask = cv2.bitwise_not(binary_mask)

					# Multiply the inverted mask with the original image
					masked_image = cv2.bitwise_and(original_image, original_image, mask=inverted_mask)
					
					# mask the mouth area
					objects = []
					with (open(landmarks2d_path, "rb")) as openfile:
						while True:
							try:
								objects.append(pickle.load(openfile))
							except EOFError:
								break
					
					mouth_landmarks = torch.Tensor([[[-int(objects[0][i][0]), -int(objects[0][i][1])] for i in range(len(objects[0])) if 48 <= i < 69]])
					
					# lip region crop + 1pixel boundary around 
					minX = int(mouth_landmarks[:, :, 0].min() - 1)
					maxX = int(mouth_landmarks[:, :, 0].max() + 1)
					minY = int(mouth_landmarks[:, :, 1].min() - 1)
					maxY = int(mouth_landmarks[:, :, 1].max() + 1)
					masked_image = cv2.rectangle(masked_image, (minX, minY), (maxX, maxY), (0, 0, 0), -1)	
				
					# image_path = f"/home/avocoral/MemFace/test_folder/masked_image{i}.png"
					# cv2.imwrite(image_path, masked_image)
					
					# channel_wise concat of masked image and ref image
					b1, g1, r1 = cv2.split(masked_image)
					b2, g2, r2 = cv2.split(ref_image)
					merged_image = cv2.merge((r1, g1, b1, r2, g2, b2))
					merged_image_tensor = torch.from_numpy(merged_image).permute(2, 0, 1)
					
					# should i transpose ref image and masked image as well? - for consistancy - yes
					masked_ref_images.append(merged_image_tensor)
					original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
					orig_images.append(torch.from_numpy(original_image).permute(2, 0, 1))
					landmarks3d_final.append(landmarks3d)
				
				# stack and normalize
				# mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
				# std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
				# orig_images = (torch.stack(orig_images) - mean) / std
				# landmarks3d = torch.stack(landmarks3d_final)
				# masked_ref_images = ((torch.stack(masked_ref_images) - mean.repeat(1, 2, 1, 1)) / std.repeat(1, 2, 1, 1))
				
				# without normalization
				orig_images = torch.stack(orig_images)
				landmarks3d = torch.stack(landmarks3d_final)
				masked_ref_images = torch.stack(masked_ref_images)
				print(f'masked_ref_images.shape, orig_images.shape, landmarks3d.shape = {masked_ref_images.shape}, {orig_images.shape}, {landmarks3d.shape}')	
				
				# return (masked_img, ref_image), original_image, landmarks3d 
				return masked_ref_images, orig_images, landmarks3d


class NeuralRenderingDataModule(pl.LightningDataModule):
		def __init__(self, data_dir='/home/avocoral/Downloads/Obamaset/Obama_vid/Obama', metadata_dir='/home/avocoral/Downloads/Obamaset/Obama_meta', inference_data_path=None, inference_metadata_path=None, batch_size=2):
				super().__init__()
				self.data_dir = data_dir
				self.metadata_dir = metadata_dir
				self.batch_size = batch_size
				self.inference_data_path = inference_data_path
				self.inference_metadata_path = inference_metadata_path
				self.transform = transforms.Compose([
						transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the pixel values
				])

		def setup(self, stage = 'fit'):
				if stage == 'fit':
						NRDataset = NeuralRenderingDataset(self.data_dir, self.metadata_dir)
						proportions = [.85, .15]
						lengths = [int(p * len(NRDataset)) for p in proportions]
						lengths[-1] = len(NRDataset) - sum(lengths[:-1])
						self.train, self.val = random_split(NRDataset, lengths)
				elif stage == 'inference':
						self.inference_dataset = NeuralRenderingDataset(self.inference_data_path, self.inference_metadata_path)				
				else:
						raise ValueError(f"Invalid stage name: {stage}")

		def train_dataloader(self):
				return DataLoader(self.train, batch_size=self.batch_size, num_workers=24, drop_last=True)

		def val_dataloader(self):
				return DataLoader(self.val, batch_size=self.batch_size, num_workers=24, drop_last=True)
		
		def inference_dataloader(self):
				return DataLoader(self.inference_dataset, batch_size=1, shuffle=False)


# based on AVSpeech dataset
class Audio2ExpDataset(Dataset):
		def __init__(self, audio_embed_dir : str, coeff_dir : str, transform=None):
				self.audio_embed_dir = audio_embed_dir
				self.coeff_dir = coeff_dir
				self.transform = transform

		def __len__(self):
				# return len(os.listdir(self.coeff_dir))
				return 1

		def __getitem__(self, idx):
				print(f'ifx: {idx}')
				sample_name = os.listdir(self.coeff_dir)[idx]
				# audio_embed_filepath = os.path.join(self.audio_embed_dir, sample_name+'.pt')
				audio_embed_filepath = os.path.join(self.audio_embed_dir, 'Obama.pt')
				# audio_embed = torch.cat(torch.load(audio_embed_filepath), dim=0)
				audio_embed = torch.load(audio_embed_filepath)
				print(f'audio_embed shape: {audio_embed.shape}')
				print(f'audio embedding name: {audio_embed_filepath}')
				# print(f'foldername = {sample_name[:-3]}')
				# coeff_folderpath = os.path.join(self.coeff_dir, sample_name)
				coeff_folderpath = self.coeff_dir
				# print(f'coeff_folderpath: {coeff_folderpath}')
				# audioembeds every 50ms, so 1 second of audio is 50 elements, 30 seconds is 1500
				audio_embed = audio_embed[:1500]
				print(f'audio_embed shape: {audio_embed.shape}')
				# print(f'audio_embed: {audio_embed}')	
								
				exp	= []
				pose = []
				shape = []
				Om = []
				sample_len = len(os.listdir(coeff_folderpath))
				
				for i, frame_name in enumerate(os.listdir(coeff_folderpath)):
						if i>=900:
								continue
						exp.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'exp.npy'))))
						pose.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'pose.npy'))))
						shape.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'shape.npy'))))
						# Om_frame = torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'landmarks3d.npy')))
						# print(f'landmarks_frame.shape: {Om_frame.shape}')
						Om.append(torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'landmarks3d.npy')))[:, 48:, :])
				
				# print(f'Om[0].shape: {Om[0].shape}')
						
				exp = torch.stack(exp)
				pose = torch.stack(pose)
				shape = torch.stack(shape)
				landmarks3d = torch.stack(Om)
				# coeff = torch.cat(torch.stack(exp), torch.stack(pose), torch.stack(shape), torch.squeeze(torch.stack(Om)))
				video_frames_num = exp.shape[0]
				
				# print(f'video_frames_num: {video_frames_num}')
				audio_embed = F.interpolate(audio_embed.T.unsqueeze(0), size=[video_frames_num], mode='nearest').squeeze(0).T

				# print(f'new audio_embed shape orig: {audio_embed.shape}')
				print(f"exp_coeff shape orig: {exp.shape}")
				# print(f"pose_coeff shape orig: {pose.shape}")
				# print(f"shape_coeff shape orig: {shape.shape}")
				# print(f"landmarks3d shape orig: {landmarks3d.shape}")
				return audio_embed, exp, pose, shape, landmarks3d


class Audio2ExpDataModule(pl.LightningDataModule):
		def __init__(self, audio_dir: str = '/home/avocoral/Downloads/Obamaset/Obama_audio', coeff_dir: str = '/home/avocoral/Downloads/Obamaset/Obama_vid/Obama', batch_size: int = 1):
				super().__init__()
				self.audio_dir = audio_dir
				self.coeff_dir = coeff_dir
				self.batch_size = batch_size
				self._has_setup_fit = False
				self._has_setup_adaptation = False
				self._has_setup_inference = False

		def setup(self, stage = 'fit'):
				# if stage == 'fit':
				# 		AVSpeech = Audio2ExpDataset(self.audio_dir, self.coeff_dir)
				# 		proportions = [.85, .15]
				# 		lengths = [int(p * len(AVSpeech)) for p in proportions]
				# 		lengths[-1] = len(AVSpeech) - sum(lengths[:-1])
				# 		self.train, self.val = random_split(AVSpeech, lengths)
				# 		# self.train = AVSpeech
				# 		self._has_setup_fit = True
				if stage == 'fit':
						AVSpeech = Audio2ExpDataset(self.audio_dir, self.coeff_dir)
						self.train = AVSpeech
						self._has_setup_fit = True
				elif stage == 'adaptation':
						AVSpeech = Audio2ExpDataset(self.audio_dir, self.coeff_dir)
						self.train = AVSpeech
						# self.val = AVSpeech[1]
						self._has_setup_adaptation = True
				elif stage == 'inference':
						AVSpeech = Audio2ExpDataset(self.audio_dir, self.coeff_dir)
						self.inference = AVSpeech
						self._has_setup_inference = True


		def train_dataloader(self):
				return DataLoader(self.train, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn, drop_last=True)

		def val_dataloader(self):
				return DataLoader(self.val, batch_size=1, num_workers=0, collate_fn=self.collate_fn)
		
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

				packed_audio_embed = torch.nn.utils.rnn.pack_padded_sequence(padded_audio_embed, audio_embed_lengths, batch_first=True, enforce_sorted=False).to('cuda')
				packed_exp = torch.nn.utils.rnn.pack_padded_sequence(padded_exp, exp_lengths, batch_first=True, enforce_sorted=False).to('cuda')
				packed_pose = torch.nn.utils.rnn.pack_padded_sequence(padded_pose, pose_lengths, batch_first=True, enforce_sorted=False).to('cuda')
				packed_shape = torch.nn.utils.rnn.pack_padded_sequence(padded_shape, shape_lengths, batch_first=True, enforce_sorted=False).to('cuda')
				packed_landmarks3d = torch.nn.utils.rnn.pack_padded_sequence(padded_landmarks3d, landmarks3d_lengths, batch_first=True, enforce_sorted=False).to('cuda')

				# padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
				# mask = (padded_batch != 0)
				# packed_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_batch, mask.sum(1), batch_first=True, enforce_sorted=False)
				return packed_audio_embed, packed_exp, packed_pose, packed_shape, packed_landmarks3d

if __name__ == '__main__':
		# datamodule = Audio2ExpDataModule()
		# datamodule.setup()
		# device = 'cuda:0'
		# print(f'len: {len(datamodule.train)}')
		# train_loader = datamodule.train_dataloader()	
		# 
		# first_batch = next(iter(train_loader))
		# packed_audio_embed, packed_exp, packed_pose, packed_shape, packed_landmarks3d, sequence_lengths = first_batch

		# audio_embed, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_audio_embed, batch_first=True)
		# exp, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_exp, batch_first=True)
		# pose, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_pose, batch_first=True)
		# shape, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_shape, batch_first=True)
		# landmarks3d, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_landmarks3d, batch_first=True)

		# 
		# print(f'sequence_lengths: {sequence_lengths}')
		# 

		# # take the 2nd sample from batch, crop the padding | double check the indexing !
		# audio_embed = audio_embed[1][:int(audio_embed_lengths[0])]
		# exp = exp[1][:int(exp_lengths[0])]
		# pose = pose[1][:int(pose_lengths[0])]
		# shape = shape[1][:int(shape_lengths[0])]
		# landmarks3d = landmarks3d[1][:int(landmarks3d_lengths[0])]
		# unpacked_audio_embed = [seq[:seq_len] for seq, seq_len in zip(audio_embed, sequence_lengths)]
		# unpacked_exp = [seq[:seq_len] for seq, seq_len in zip(exp, sequence_lengths)]
		# unpacked_pose = [seq[:seq_len] for seq, seq_len in zip(pose, sequence_lengths)]
		# unpacked_shape = [seq[:seq_len] for seq, seq_len in zip(shape, sequence_lengths)]
		# unpacked_landmarks3d = [seq[:seq_len] for seq, seq_len in zip(landmarks3d, sequence_lengths)]
		# i = 2
		# print('-------- AFTER SLICING ----------')
		# print(f"audio_embed shape: {unpacked_audio_embed[i].shape}")
		# print(f"exp shape: {unpacked_exp[i].shape}")
		# print(f"pose shape: {unpacked_pose[i].shape}")
		# print(f"shape shape: {unpacked_shape[i].shape}")
		# print(f"landmarks3d shape: {unpacked_landmarks3d[i].shape}")

		dataset = NeuralRenderingDataset()
		a = dataset[0]
		print(f'shape a: {a[0].shape}, {a[1].shape}, {a[2].shape}')

		datamodule = NeuralRenderingDataModule()
		datamodule.setup()
		device = 'cuda:0'
		train_loader = datamodule.train_dataloader()
		first_batch = next(iter(train_loader))
		masked_ref_images, orig_images, landmarks3d = first_batch
		print(f'masked_ref_images.shape: {masked_ref_images.shape}')
		print(f'orig_images.shape: {orig_images.shape}')
		print(f'landmarks3d.shape: {landmarks3d.shape}')
