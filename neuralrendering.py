import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import utils
from utils import construct_explicitmem
from datasets import NeuralRenderingDataModule
import torchvision
# from pytorch_lightning.loggers import TensorBoardLogger
import wandb
import numpy as np

from ConvLSTM_pytorch.convlstm import ConvLSTM
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2

import torchvision.transforms.functional as TF

# Load a pre-trained VGG16 model
device='cuda'
vgg = models.vgg16(pretrained=True).features[:23]
vgg.to(device)
vgg.eval()

# vgg_model = models.vgg19(pretrained=True).features
# vgg_model.to(device)
# vgg_model.eval()

transform = transforms.ToPILImage()
mse_loss = torch.nn.MSELoss()
# logger = TensorBoardLogger("logs", name="my_experiment")
wandb_logger = WandbLogger(name='NRmodel',project='MemFace')
pl.seed_everything(42, workers=True)



class VGGLoss(nn.Module):
    def __init__(self, feature_layers):
        super(VGGLoss, self).__init__()
        self.feature_layers = feature_layers
        self.vgg_model = models.vgg19(pretrained=True).features
        self.vgg_model.eval()
        self.vgg_layers = nn.Sequential(*list(self.vgg_model.children())[:max(feature_layers) + 1])
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        x_features = self.vgg_layers(x)
        y_features = self.vgg_layers(y)

        loss = 0
        for layer_idx in self.feature_layers:
            loss += self.criterion(x_features[layer_idx], y_features[layer_idx])

        return loss


class VGGPerceptualLoss(torch.nn.Module):
		def __init__(self, resize=False):
				super(VGGPerceptualLoss, self).__init__()
				blocks = []
				blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
				blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
				blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
				blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
				for bl in blocks:
						for p in bl.parameters():
								p.requires_grad = False
				self.blocks = torch.nn.ModuleList(blocks)
				self.transform = torch.nn.functional.interpolate
				self.resize = resize
				self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
				self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

		def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
				if input.shape[1] != 3:
						input = input.repeat(1, 3, 1, 1)
						target = target.repeat(1, 3, 1, 1)
				input = (input-self.mean) / self.std
				target = (target-self.mean) / self.std
				if self.resize:
						input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
						target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
				loss = 0.0
				x = input
				y = target
				for i, block in enumerate(self.blocks):
						x = block(x)
						y = block(y)
						if i in feature_layers:
								loss += torch.nn.functional.l1_loss(x, y)
						if i in style_layers:
								act_x = x.reshape(x.shape[0], x.shape[1], -1)
								act_y = y.reshape(y.shape[0], y.shape[1], -1)
								gram_x = act_x @ act_x.permute(0, 2, 1)
								gram_y = act_y @ act_y.permute(0, 2, 1)
								loss += torch.nn.functional.l1_loss(gram_x, gram_y)
				return loss

vgg = VGGPerceptualLoss().to("cuda:0")

class NeuralRender(pl.LightningModule):
	def __init__(self):
		super(NeuralRender, self).__init__()
		self.N = 300

		self.ImageEncoder = ImageEncoder()
		self.ExplicitMem = ExplicitMem()
		self.ConvLSTM = ConvLSTM(input_dim=384, hidden_dim=384, kernel_size=(3, 3), num_layers=1, batch_first=True)
		self.ImageDecoder = ImageDecoder()
		self.discriminator = Discriminator()
	
	def forward(self, masked_ref_images, landmarks3d, batch_idx):
		encoded_images, skip_connections = self.ImageEncoder(masked_ref_images.float())
		memory_images, exp_mem_image_1st, attention_score = self.ExplicitMem(landmarks3d.float(), batch_idx)
		if batch_idx % 500 == 0:
			exp_mem_image = wandb.Image(TF.to_pil_image((exp_mem_image_1st * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()), caption=f'exp_mem image, score:{attention_score}')
			self.logger.experiment.log({"exp_mem_image" : [exp_mem_image]})
		
		# memory_images: (60, 384, 4, 7) - > (60, 384, 7, 7) - shape of encoded_images
		reshaped_memory_images = memory_images.reshape(60, 384, 3, 7)
		
		print(f'encoded_images.shape: {encoded_images.shape}')
		print(f'reshaped_memory_images.shape: {reshaped_memory_images.shape}')
		# reshaped_tensor = reshaped_memory_images.view(-1, reshaped_memory_images.shape[2], reshaped_memory_images.shape[3])
		# print(f'reshaped_tensor.shape just before interpolation: {reshaped_tensor.shape}')
		interpolated_memory_images = F.interpolate(reshaped_memory_images, size=(7, 7), mode='bilinear', align_corners=False)
		# interpolated_memory_images = interpolated_tensor.view(input_shape[0], input_shape[1], 7, 7)
	
		print(f'interpolated_memory_images.shape: {interpolated_memory_images.shape}')
		

		final_encoded_images = encoded_images + interpolated_memory_images
		final_encoded_images = final_encoded_images.reshape(2, 30, 384, 7, 7)
		print(f'final_encoded_images.shape before LSTM: {final_encoded_images.shape}')
		# here we should reshape (bs*T, ...) -> (bs, T, ...)
		lstm_output, state = self.ConvLSTM(final_encoded_images)
		print(f'len(lstm_output[-1]):{len(lstm_output[-1])})')
		print(f'len(state[-1]): {len(state[-1])}')
		print(f'convlstm_output[-1][-1].shape: {lstm_output[-1][-1].shape}')
		print(f'state.shape: {state[-1][-1].shape}')
		
		first_elements = [tensor[0] for tensor in lstm_output]
		second_elements = [tensor[1] for tensor in lstm_output]
		
		convlstm_result = torch.cat((torch.cat(first_elements, dim=0), torch.cat(second_elements, dim=0)), dim=0)
		print(f'convlstm_result.shape before reshape: {convlstm_result.shape}')
		
		# and here we should reshape (bs, T, ...) -> (bs*T, ...)
		convlstm_result = convlstm_result.reshape(60, 384, 7, 7)
		print(f'convlstm_result.shape right before ImageDecoder: {convlstm_result.shape}')
		output_images_hat = self.ImageDecoder(convlstm_result, skip_connections)
		
		return output_images_hat
	
	def training_step(self, batch, batch_idx):
		masked_ref_images, output_images, landmarks3d = batch
		# reshape
			
		masked_ref_images = masked_ref_images.view(60, 6, 224, 224)
		output_images = output_images.view(60, 3, 224, 224)
		landmarks3d = landmarks3d.squeeze().view(60, 20, 3)

		output_images_hat = self(masked_ref_images, landmarks3d, batch_idx)
		device = 'cuda'	
		output_images = output_images.float().to(device)
		output_images_hat = output_images_hat.float().to(device)
		print(f'output_images.shape: {output_images.shape}')
		print(f'output_images_hat.shape: {output_images_hat.shape}')
		print(f'output_images.device: {output_images.device}')
		print(f'output_images_hat.device: {output_images_hat.device}')
		# vgg_loss = vgg(output_images_hat, output_images, feature_layers=[0, 1, 2, 3], style_layers=[])

		# Calculate the VGG loss
		# feature_layers = [2, 9, 16, 23]  # Example feature layers to compute loss
		# vgg_loss = VGGLoss(feature_layers)
		# vgg_loss_final = vgg_loss(output_images, output_images_hat)


		vgg_loss = vgg(output_images, output_images_hat, feature_layers=[2], style_layers=[0, 1, 2, 3])
		print(f'vgg_loss_final: {vgg_loss}')
		print(f'vgg_loss_final.shape: {vgg_loss.shape}')
		mse_loss_batch = mse_loss(output_images, output_images_hat)
		print(f'mse_loss_batch: {mse_loss_batch}')
		print(f'mse_loss_batch.shape: {mse_loss_batch.shape}')
		l_rec = (mse_loss(output_images, output_images_hat) + vgg_loss)
		l_d_adv = discriminator_loss(self.discriminator, output_images, output_images_hat)
		l_nr_adv = generator_loss(self.discriminator, output_images_hat)
		# 20*l_rec
		# loss = 1*l_rec + 1*l_d_adv + 1*l_nr_adv
		loss = 1*l_rec
		self.log("l_rec", l_rec)
		self.log("l_d_adv", l_d_adv)
		self.log("l_nr_adv", l_nr_adv)
		self.log("vgg_loss", vgg_loss)
		self.log("mse_loss_batch", mse_loss_batch)
		self.log("loss", loss)	
		
		# save input, output and output_hat  images
		if batch_idx % 500 == 0:
			ref_image = wandb.Image(TF.to_pil_image((masked_ref_images[0][3:]).to(torch.uint8).permute(1, 2, 0).cpu().numpy()), caption='ref_image')
			masked_image = wandb.Image(TF.to_pil_image((masked_ref_images[0][0:3]).to(torch.uint8).permute(1, 2, 0).cpu().numpy()), caption='masked_image')
			output_image = wandb.Image(TF.to_pil_image((output_images[0]).to(torch.uint8).permute(1, 2, 0).cpu().numpy()), caption='output_image')
			output_hat_image = wandb.Image(TF.to_pil_image((output_images_hat[0]).to(torch.uint8).permute(1, 2, 0).cpu().numpy()), caption='output_hat_image')
			self.logger.experiment.log({"ref_input" : [ref_image]})
			self.logger.experiment.log({"masked_input" : [masked_image]})
			self.logger.experiment.log({"output" : [output_image]})
			self.logger.experiment.log({"output_hat" : [output_hat_image]})


	def validation_step(self, batch, batch_idx):
		masked_ref_images, output_images, landmarks3d = batch
		# reshape
		masked_ref_images = masked_ref_images.view(60, 6, 224, 224)
		output_images = output_images.view(60, 3, 224, 224)
		landmarks3d = landmarks3d.squeeze().view(60, 20, 3)

		output_images_hat = self(masked_ref_images, landmarks3d, batch_idx)
		device = 'cuda'	
		output_images = output_images.float().to(device)
		output_images_hat = output_images_hat.float().to(device)
		print(f'output_images.shape: {output_images.shape}')
		print(f'output_images_hat.shape: {output_images_hat.shape}')
		print(f'output_images.device: {output_images.device}')
		print(f'output_images_hat.device: {output_images_hat.device}')
		# vgg_loss = vgg(output_images_hat, output_images, feature_layers=[0, 1, 2, 3], style_layers=[])
		vgg_loss = vgg(output_images, output_images_hat, feature_layers=[2], style_layers=[0, 1, 2, 3])
		# feature_layers = [2, 9, 16, 23]  # Example feature layers to compute loss
		# vgg_loss = VGGLoss(feature_layers)
		# vgg_loss_final = vgg_loss(output_images, output_images_hat)
		l_rec = (mse_loss(output_images, output_images_hat) + vgg_loss)
		l_d_adv = discriminator_loss(self.discriminator, output_images, output_images_hat)
		l_nr_adv = generator_loss(self.discriminator, output_images_hat)
		# 20*l_rec
		# val_loss = 1*l_rec + 1*l_d_adv + 1*l_nr_adv
		val_loss = 1*l_rec
		self.log("val_loss", val_loss)
		if batch_idx % 500 == 0:
			ref_image = wandb.Image(TF.to_pil_image((masked_ref_images[0][3:]* 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()), caption='val_ref_image')
			masked_image = wandb.Image(TF.to_pil_image((masked_ref_images[0][0:3]* 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()), caption='val_masked_image')
			output_image = wandb.Image(TF.to_pil_image((output_images[0]* 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()), caption='val_output_image')
			output_hat_image = wandb.Image(TF.to_pil_image((output_images_hat[0]* 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()), caption='val_output_hat_image')
			self.logger.experiment.log({"val_ref_input" : [ref_image]})
			self.logger.experiment.log({"val_masked_input" : [masked_image]})
			self.logger.experiment.log({"val_output" : [output_image]})
			self.logger.experiment.log({"val_output_hat" : [output_hat_image]})

	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
		return optimizer


class Conv2dBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super().__init__()
		
		self.layers = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.InstanceNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
				nn.LeakyReLU(negative_slope=0.1, inplace=True)
		)

	def forward(self, x):
		x = self.layers(x)
		return x

class ImageEncoder(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = Conv2dBlock(6, 48, kernel_size=5, stride=2, padding=2)
		self.conv2 = Conv2dBlock(48, 96, kernel_size=4, stride=2, padding=1)
		self.conv3 = Conv2dBlock(96, 192, kernel_size=4, stride=2, padding=1)
		self.conv4 = Conv2dBlock(192, 384, kernel_size=4, stride=2, padding=1)
		self.conv5 = Conv2dBlock(384, 384, kernel_size=4, stride=2, padding=1)
		
		
	def forward(self, x):
		out1 = self.conv1(x)
		print(f'ImageEncoder, after conv1: {out1.shape}')
		out2 = self.conv2(out1)
		print(f'ImageEncoder, after conv2: {out2.shape}')
		out3 = self.conv3(out2)
		print(f'ImageEncoder, after conv3: {out3.shape}')
		out4 = self.conv4(out3)
		print(f'ImageEncoder, after conv4: {out4.shape}')
		x = self.conv5(out4)
		print(f'ImageEncoder, after conv5: {x.shape}')

		skip_connections = {
				'skip4': out1,
				'skip3': out2,
				'skip2': out3,
				'skip1': out4
		}

		return x, skip_connections


class ExplicitMem(nn.Module):
	def __init__(self, dropout=0.1):
		super().__init__()
		
		self.LipsEncoder = LipEncoder()
		
		# depending on training video, we would need to use different ExplicitMem
		self.K_nr, self.V_nr = construct_explicitmem(data_dir='/home/avocoral/Downloads/Obamaset/Obama_vid', metadata_dir='/home/avocoral/Downloads/Obamaset/Obama_meta')
		
		self.K_nr = self.K_nr.to('cuda').view(300, 60)
		self.V_nr = self.V_nr.to('cuda').view(300, 3, 112, 224)

		# self.w_q = nn.Parameter(torch.randn(60, 60))
		# self.w_k = nn.Parameter(torch.randn(60, 60))
		self.w_q = nn.Linear(in_features=60, out_features=32, bias=True)
		self.w_k = nn.Linear(in_features=60, out_features=32, bias=True)
		# self.w_o = nn.Linear(torch.randn(1, 64))
		self.dropout = nn.Dropout(dropout)
		self.d_k = 60
	
	def forward(self, query, batch_idx):
		# the input of the ExplicitMemory is tensor 25x20x3, 25 frames of landmarks3d_mouth
		# the output is tensor of shape 50x384x8x8, same as output of ImageEncoder/LipsEncoder
		# lips encoder supposed to be the equivalent of the w_v
		print(f'inside EXPLICITMEM, query.shape: {query.shape}')

		#flatten the query 60x20x3 -> 60x60
		query_flat = query.reshape(-1, 60)
		print(f'inside EXPLICITMEM, query_flat.shape: {query_flat.shape}')
		q = self.w_q(query_flat)
		print(f'q.shape: {q.shape}')
		print(f'self.K_nr.shape: {self.K_nr.shape}')
		k = self.w_k(self.K_nr)
		print(f'k.shape: {k.shape}')
		v = self.LipsEncoder(self.V_nr).view(300, -1)
		print(f'v.shape: {v.shape}')
		
		scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
		print(f'scores.shape: {scores.shape}')	
		attention_weights = torch.softmax(scores, dim=-1).reshape(60, 300)
		print(f'attention_weights.shape: {attention_weights.shape}')
		attention = torch.matmul(attention_weights, v)
		print(f'attention.shape: {attention.shape}')
		# do we need w_o?
		# do we need dropout?
		# output = attention * self.w_o
		output = self.dropout(attention)
		attention_score = torch.max(attention_weights[0])
		exp_mem_image_1st = self.V_nr[torch.argmax(attention_weights[0])]
		# if batch_idx % 500 == 0:
		# 	ref_image = wandb.Image(TF.to_pil_image((exp_mem_image_1st * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()), caption='exp mem image most compatible')
		# 	wandb_logger.experiment.log({"exp_mem[0]" : [ref_image]})
		# 	wandb_logger.experiment.log({'attention score:' : attention_score})
			
		return output, exp_mem_image_1st, attention_score


class LipEncoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Sequential(
						nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=2, bias=False),
						nn.InstanceNorm2d(48, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
						nn.LeakyReLU(negative_slope=0.1, inplace=True)
		)
		self.conv2 = nn.Sequential(
						nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1, bias=False),
						nn.InstanceNorm2d(96, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
						nn.LeakyReLU(negative_slope=0.1, inplace=True)
		)
		self.conv3 = nn.Sequential(
						nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1, bias=False),
						nn.InstanceNorm2d(192, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
						nn.LeakyReLU(negative_slope=0.1, inplace=True)
		)
		self.conv4 = nn.Sequential(
						nn.Conv2d(192, 384, kernel_size=4, stride=2, padding=1, bias=False),
						nn.InstanceNorm2d(384, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
						nn.LeakyReLU(negative_slope=0.1, inplace=True)
		)
		self.conv5 = nn.Sequential(
						nn.Conv2d(384, 384, kernel_size=4, stride=2, padding=1, bias=False),
						nn.InstanceNorm2d(384, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
						nn.LeakyReLU(negative_slope=0.1, inplace=True)
		)
		
	def forward(self, x):
		x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
		return x


class ImageDecoder(nn.Module):
	def __init__(self):
		super(ImageDecoder, self).__init__()

		self.dconv1 = ConvTranspose2dBlock(384, 384, kernel_size=4, stride=2, padding=1, bias=False)
		self.dconv2 = ConvTranspose2dBlock(768, 192, kernel_size=4, stride=2, padding=1, bias=False)
		self.dconv3 = ConvTranspose2dBlock(384, 96, kernel_size=4, stride=2, padding=1, bias=False)
		self.dconv4 = ConvTranspose2dBlock(192, 48, kernel_size=4, stride=2, padding=1, bias=False)
		self.dconv5 = ConvTranspose2dBlock(96, 48, kernel_size=4, stride=2, padding=1, bias=False)
		self.dconv6 = nn.Sequential(
				nn.Conv2d(48, 3, kernel_size=5, stride=1, padding=2),
				nn.Tanh()
		)

	def forward(self, x, skip_connections):
		print(f'ImageDecoder, input.shape: {x.shape}')
		x = self.dconv1(x)
		print(f'ImageDecoder, self.dconv1(x).shape: {x.shape}')
		x = self.dconv2(torch.cat([x, skip_connections['skip1']], dim=1))
		print(f'ImageDecoder, self.dconv2(x).shape: {x.shape}')
		x = self.dconv3(torch.cat([x, skip_connections['skip2']], dim=1))
		print(f'ImageDecoder, self.dconv3(x).shape: {x.shape}')
		x = self.dconv4(torch.cat([x, skip_connections['skip3']], dim=1))
		print(f'ImageDecoder, self.dconv4(x).shape: {x.shape}')
		x = self.dconv5(torch.cat([x, skip_connections['skip4']], dim=1))
		print(f'ImageDecoder, self.dconv5(x).shape: {x.shape}')
		x = self.dconv6(x)
		print(f'ImageDecoder, self.dconv6(x).shape: {x.shape}')
		return x


class ConvTranspose2dBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
		super(ConvTranspose2dBlock, self).__init__()
		self.layers = nn.Sequential(
				nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
				nn.InstanceNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
				nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
						nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
						nn.LeakyReLU(0.2, inplace=True),
						nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
						nn.LeakyReLU(0.2, inplace=True),
						nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
						nn.LeakyReLU(0.2, inplace=True),
						nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0),
						nn.AdaptiveAvgPool2d(1),
						nn.Sigmoid()
				)
	def forward(self, x):
		print(f'discriminator input shape: {x.shape}')
		res = self.net(x)
		res = res.squeeze()
		print(f'discriminator output shape: {res.shape}')
		return res


def discriminator_loss(d, real_images, generated_images):
	# Compute the discriminator output for real and generated images
	print(f'real_images.shape: {real_images.shape}')
	print(f'generated_images.shape: {generated_images.shape}')
	d_real = d(real_images)
	d_generated = d(generated_images)

	# Compute the adversarial loss
	adversarial_loss = -torch.mean(torch.log(d_real + 1e-8)) - torch.mean(torch.log(1 - d_generated + 1e-8))
	print(f'adversarial_loss: {adversarial_loss}')
	print(f'adversarial_loss.shape: {adversarial_loss.shape}')
	return adversarial_loss


def generator_loss(discriminator, generated_images):
	# Compute the discriminator output for generated images
	d_generated = discriminator(generated_images)

	# Compute the generator loss
	generator_loss = torch.mean(torch.log(1 - d_generated + 1e-8))
	print(f'generator_loss: {generator_loss}')
	print(f'generator_loss.shape: {generator_loss.shape}')

	return generator_loss




if __name__ == '__main__':
	# CHECK dataloaders

	# masked_ref_images, orig_images, landmarks3d = next(iter(train_dataloader))
	# transform = transforms.ToPILImage()
	# for batch_idx in range(orig_images.size(0)):
	# 	batch = orig_images[batch_idx]
	# 	for image_idx, image in enumerate(batch):
	# 		pil_image = transform(image)
	# 		image_path = f"/home/avocoral/MemFace/test_folder/batch{batch_idx}_image{image_idx}.png"
	# 		pil_image.save(image_path)
	# 		print(f"Saved image: {image_path}")	
	# 
	# for batch_idx in range(masked_ref_images.size(0)):
	# 	batch = masked_ref_images[batch_idx]
	# 	for frame_idx, frame in enumerate(batch):
	# 		frame1, frame2 = torch.split(frame, 3, dim=0)
	# 		pil_image1 = transform(frame1)
	# 		pil_image2 = transform(frame2)
	# 		image_path1 = f"/home/avocoral/MemFace/test_folder/batch{batch_idx}_frame{frame_idx}_image1.png"
	# 		image_path2 = f"/home/avocoral/MemFace/test_folder/batch{batch_idx}_frame{frame_idx}_image2.png"
	# 		pil_image1.save(image_path1)
	# 		pil_image2.save(image_path2)
	# 		print(f"Saved image: {image_path1}")	
	
	# CHECK explicit mem

	# K_nr = torch.load('/home/avocoral/Downloads/Obamaset/K_nr.pt')
	# V_nr = torch.load('/home/avocoral/Downloads/Obamaset/V_nr.pt')
	# 
	# for i, image_tensor in enumerate(V_nr):
	# 	image_path = os.path.join('/home/avocoral/MemFace/test_folder', f"image_{i}.jpg")
	# 	pil_image = torchvision.transforms.functional.to_pil_image(image_tensor)
	# 	pil_image.save(image_path)
	# 	print(f'image_{i} saved!')
	
	# check VGG loss
	
	# datamodule = NeuralRenderingDataModule()
	# datamodule.setup()
	# train_dataloader = datamodule.train_dataloader()
	# masked_ref_images, orig_images, landmarks3d = next(iter(train_dataloader))
	# orig_images = orig_images.float().to(device)
	# print(f'orig_images[0][0].shape: {orig_images[0][0].shape}')	
	# print(f'orig_images[0][0].device: {orig_images[0][0].device}')
	# vgg_loss = vgg(orig_images[0], orig_images[1], feature_layers=[2], style_layers=[0, 1, 2, 3])
	# print(f'vgg_loss: {vgg_loss}')

	torch.cuda.empty_cache()
	NRmodel = NeuralRender()
	
	datamodule = NeuralRenderingDataModule()
	datamodule.setup()
	train_dataloader = datamodule.train_dataloader()
	val_dataloader = datamodule.val_dataloader()
	
	trainer = pl.Trainer(default_root_dir='checkpoints', logger=wandb_logger, gpus=[0], accelerator="gpu", gradient_clip_val=0.5)
	trainer.fit(NRmodel, train_dataloader, val_dataloader)



	
