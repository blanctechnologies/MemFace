import torch
from audio2exp import Audio2Exp
from preprocessing import getAudioEncoding
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import subprocess
import librosa
from datasets import Audio2ExpDataModule, Audio2ExpDataset
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
import os
import numpy as np
import re
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip

from pathlib import Path
from emoca.gdl_apps.EMOCA.utils.io import decode
from gdl.utils.lightning_logging import _fix_image
from skimage.io import imsave
from emoca.gdl.datasets.ImageTestDataset import TestData
from emoca.gdl_apps.EMOCA.utils.load import load_model
from emoca.gdl.utils.FaceDetector import FAN
from emoca.gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
from emoca.gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test, decode

def torch_img_to_np(img):
	return img.detach().cpu().numpy().transpose(1, 2, 0)


def natural_sort_key(s):
    # Key function to sort filenames naturally (e.g., frame_1.jpg, frame_2.jpg, ..., frame_10.jpg)
	return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def extractAudioEncoding(audio_filepath, audio_embedding_filepath):
	device = 'cuda'
	processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
	model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
	sample_rate = 16000
	waveform, _ = librosa.load(audio_filepath, sr=sample_rate)
	transcript, all_logits = getAudioEncoding(waveform, processor, model)
	all_logits = torch.cat(all_logits, dim=0)
	print(f'all_logits.shape after stacking: {all_logits.shape}')
	torch.save(all_logits, audio_embedding_filepath)
	
	print(f'transcript: {transcript}')
	return all_logits


# extract original crops
def extract_original_crops():
	pass



def audio2expression(audio_embed_filepath):
	device = 'cuda:0'

	# audio2exp = Audio2Exp()
	# checkpoints = torch.load('/home/avocoral/MemFace/adaptation_checkpoints/MemFace/168jb4lf/checkpoints/epoch=999-step=999.ckpt')
	# audio2exp.load_state_dict(checkpoints["state_dict"])
	# audio2exp.to('cuda:0')
	# audio2exp.eval()

	# read audio embedding
	# audio_embed = torch.load(audio_embed_filepath)
	# print(f'audio_embed.shape after its loaded: {audio_embed.shape}')
	# audio_embed = torch.cat(torch.load(audio_embed_filepath), dim=0)
	# audio_embed = audio_embed[:500]
	# print(f'audio_embed before interpolation: {audio_embed.shape}')
	# audio_embed = F.interpolate(audio_embed.T.unsqueeze(0), size=[300], mode='nearest').squeeze(0).T
	# print(f'audio_embed after interpolation, before inference: {audio_embed.shape}')

	# audio_embed = audio_embed.unsqueeze(0)
	# print(f'audio_embed after adding batch dim: {audio_embed.shape}')
	# collate_fn part

	# audio_embed_lengths = [len(seq) for seq in audio_embed]
	# padded_audio_embed = torch.nn.utils.rnn.pad_sequence(audio_embed, batch_first=True)
	# packed_audio_embed = torch.nn.utils.rnn.pack_padded_sequence(padded_audio_embed, audio_embed_lengths, batch_first=True, enforce_sorted=False).to('cuda')

	# with torch.no_grad():
	# 	new_exp = audio2exp(packed_audio_embed)
	
	# test exp - high res geometry_detail.imgs
	coeff_dir ='/home/avocoral/Downloads/Obamaset/Obama_vid/Obama'
	new_exp = []
	for i, frame_name in enumerate([str(i).zfill(6)+'_000' for i in range(301, 601)]):
		new_exp.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'exp.npy'))))
	new_exp = torch.stack(new_exp)

	new_exp = new_exp.squeeze(0)
	print(f'new_exp.shape: {new_exp.shape}')
	# return new_exp
	# take old pose and shape and create tensor [new exp, old pose, old shape]
	coeff_dir ='/home/avocoral/Downloads/Obamaset/Obama_vid/Obama'
	frame_num = len(os.listdir(coeff_dir))
	frame_names = [str(i).zfill(6)+'_000' for i in range(frame_num-300+1, frame_num+1)]
	# print(f'total num of inference frames: {len(frame_names)}')
	# print(f'first frame name: {frame_names[0]}')
	# print(f'last frame name: {frame_names[-1]}')


	old_pose = []
	old_shape = []
	old_tex = []
	old_cam = []
	old_detail = []
	for i, frame_name in enumerate(frame_names):
	 	old_pose.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'pose.npy'))))
	 	old_shape.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'shape.npy'))))
	 	old_tex.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'tex.npy'))))
	 	old_cam.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'cam.npy'))))
	 	old_detail.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'detail.npy'))))

	old_pose = torch.stack(old_pose)
	old_shape = torch.stack(old_shape)
	old_tex = torch.stack(old_tex)
	old_cam = torch.stack(old_cam)
	old_detail = torch.stack(old_detail)

	# print(f'old_pose.shape: {old_pose.shape}')
	# print(f'old_shape.shape: {old_shape.shape}')

	# # generate render images for [new_exp, old_shape, old_pose]
	path_to_models = "/home/avocoral/MemFace/emoca/assets/EMOCA/models"
	model_name = 'EMOCA_v2_lr_mse_20'
	mode = 'detail'
	final_out_folder = '/home/avocoral/MemFace/test_folder'
	final_out_folder = Path(final_out_folder)
	emoca, conf = load_model(path_to_models, model_name, mode)
	emoca.cuda()
	emoca.eval()

	vals = dict()
	vals["expcode"] = new_exp[0].unsqueeze(0).to('cuda')
	# 	# .unsqueeze(0).to('cuda')
	vals["shapecode"] = old_shape[0].unsqueeze(0).to('cuda')
	vals["posecode"] = old_pose[0].unsqueeze(0).to('cuda')
	vals["texcode"] =	old_tex[0].unsqueeze(0).to('cuda')
	vals["cam"] = old_cam[0].unsqueeze(0).to('cuda')
	vals["lightcode"] = torch.from_numpy(np.load('/home/avocoral/Downloads/Obamaset/Obama_vid_with_light/dataset_preprocessed/Obama_vid/000001_000/light.npy')).unsqueeze(0).to('cuda')
	vals["detailcode"] = old_detail[0].unsqueeze(0).to('cuda')
	vals['detailemocode'] = None
	print(f'vals["expcode"].shape: {vals["expcode"].shape}')
	print(f'vals["posecode"].shape: {vals["posecode"].shape}')
	print(f'vals["shapecode"].shape: {vals["shapecode"].shape}')

	# test_frames = ['/home/avocoral/Downloads/Obamaset/Obama_vid/Obama/006294_000/inputs.png']
	# testdata = TestData(test_frames, iscrop=True, face_detector='fan')
	# print(f"testdata[0]['image']: {testdata[0]['image'].shape}")
	# print(f'len(testdata): {len(testdata)}')
	# input_tensor = testdata[0]['image']

	# Define the target size
	# target_size = (512, 512)

	# Resize the input tensor to the target size
	# resized_tensor = F.interpolate(input_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
	# resized_tensor = resized_tensor.squeeze(0)
	# testdata[0]['image']
	# vals["images"] = torch.randn((3, 512, 512)).unsqueeze(0).repeat(300, 1, 1, 1).to('cuda')
	vals["images"] = torch.randn((3, 512, 512)).unsqueeze(0).to('cuda')

	vals, visdict = decode(emoca, vals, training=False)
		
	imsave(final_out_folder / f"geometry_detail_{i}.png", _fix_image(torch_img_to_np(visdict['geometry_detail'][0])))
		# for i, image in enumerate(visdict['geometry_detail'].view(300, 3, 512, 512)):
			# imsave(final_out_folder / f"geometry_detail_{i}.png", _fix_image(torch_img_to_np(image)))
		# frames_to_video('/home/avocoral/MemFace/test_folder', "/home/avocoral/MemFace/Obama_ref_adapted.mp4", fps=30)
		# 	print(f'Frame {i} is ready!')
		
	# add audio_track


if __name__ == "__main__":
	# audio_filepath = '/home/avocoral/MemFace/MemoryFaceTest.wav'
	audio_embedding_filepath = '/home/avocoral/MemFace/MemoryFaceTest.pt'
	# logits = extractAudioEncoding(audio_filepath, audio_embedding_filepath)
	# print(f'len(logits): {len(logits)}')
	# for i in range(len(logits)):
	# 	print(f'logits[{i}].shape: {logits[i].shape}')

	if torch.cuda.is_available():
		torch.cuda.empty_cache()  # Clean GPU memory cache
	audio2expression(audio_embedding_filepath)
	# input_folder_path = '/home/avocoral/MemFace/test_folder'
	# output_video_path = "Obama_ref_adapted.mp4"
	# frames_to_video(input_folder_path, output_video_path, fps=30)












