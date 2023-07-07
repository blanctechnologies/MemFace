# NR INFERENCE
# 1. adapt NR model to new video, mode: training
# 	- input: NR model weights, 15-30 sec adaptation preprocessed video
# 	- output: NR adapted_model weights
# 	- lr = 1e−4, 50 epochs
#	2. do inference on new video, mode: inference
#		- input: NR adapted_model weights, full preprocessed source video, full target ref video
# 	- output: source transfered to target ref video

from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import pytorch_lightning as pl

import torch
import torch.nn as nn
import pytorch_lightning as pl

from datasets import NeuralRenderingDataModule
from neuralrendering import NeuralRender

import shutil
import os

class AdaptedNeuralRender(NeuralRender):
	pass 


def build_inference_dataset(orig_dataset_filepath='/home/avocoral/Downloads/Obamaset/Obama_vid'):
	output_filepath = '/home/avocoral/Downloads/Obamaset/inference_dataset'
	source_frames_dir = [str(i).zfill(6)+'_000' for i in range(1, 450)]
	target_frames_dir = [str(i).zfill(6)+'_000' for i in range(5844, 6295)]
	print(f'len(source_frames_dir): {len(source_frames_dir)}')
	print(f'len(target_frames_dir): {len(target_frames_dir)}')
	for i, folder in enumerate(source_frames_dir):
		# move
		source_frame_path = os.path.join(orig_dataset_filepath, folder)
		shutil.move(source_frame_path, output_filepath)
		print(f'folder {folder} moved.')

		# delete source geometry_coarse.png
		os.remove(os.path.join(output_filepath, folder, 'geometry_coarse.png'))
		print(f'source geometry_coarse.png deleted.')
		
		# move target geometry_coarse.png
		target_frame_path = os.path.join(orig_dataset_filepath, folder, 'geometry_coarse.png')
		target_frame_destination = os.path.join(output_filepath, target_frames_dir[i], 'geometry_coarse.png')
		shutil.move(target_frame_path, target_frame_destination)
		print(f'target geometry_coarse.png from {folder} moved to {source_folders[k]}')


def adapt_NR_model():
	adaptation_datapath = ''
	model = AdaptedNeuralRender()

	pretrained_weights_path = "path/to/pretrained/weights.ckpt"
	pretrained_state_dict = torch.load(pretrained_weights_path)
	model.load_state_dict(pretrained_state_dict)
	
	datamodule = NeuralRenderingDataModule()
	datamodule.setup()
	
	checkpoint_callback = ModelCheckpoint(dirpath="path/to/save/directory", filename="fine_tuned_model_{epoch:02d}")	
	trainer = Trainer(max_epochs=50, callbacks=[checkpoint_callback], gpus=[0], accelerator="gpu")
	trainer.fit(model, train_dataloader)


def inference():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load the trained model
	checkpoint_filepath = '/home/avocoral/MemFace/checkpoints/MemFace/yz2d7all/checkpoints/yz2d7all.ckpt'
	model = NeuralRender.load_from_checkpoint(checkpoint_filepath)
	model = model.to(device)
	model.eval()  # Set the model to evaluation mode

	datamodule = NeuralRenderingDataModule(inference_data_path='', inference_metadata_path='')  # Replace with your own DataModule initialization
	datamodule.setup(stage='inference')
	

	# Iterate over the data loader and make predictions
	for batch in data_loader:
		inputs = batch.to(device)
		with torch.no_grad():
			outputs = model(inputs)	

if __name__ == "__main__":
	build_inference_dataset()
