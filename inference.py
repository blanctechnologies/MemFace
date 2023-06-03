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


class AdaptedNeuralRender(NeuralRender):
	pass 


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
	checkpoint_filepath = ''
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


