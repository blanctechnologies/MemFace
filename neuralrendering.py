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

from ConvLSTM_pytorch.convlstm import ConvLSTM

# Load a pre-trained VGG16 model
device='cuda'
# vgg = models.vgg16(pretrained=True).features[:23]
# vgg.to(device)
# vgg.eval()


wandb_logger = WandbLogger(name='NRmodel',project='MemFace')
pl.seed_everything(42, workers=True)


class VGGPerceptualLoss(torch.nn.Module):
		def __init__(self, resize=True):
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
		super().__init__()
		self.N = 300

		self.ImageEncoder = ImageEncoder()
		self.ExplicitMem = ExplicitMem()
		self.ConvLSTM = ConvLSTM(input_dim=384, hidden_dim=768, kernel_size=(3, 3), num_layers=1, batch_first=True)
		self.ImageDecoder = ImageDecoder()
		self.discriminator = Discriminator()
	
	def forward(self, masked_ref_images, landmarks3d):
		encoded_images, skip_connections = self.ImageEncoder(masked_ref_images.float())
		memory_images = self.ExplicitMem(landmarks3d.float())

		# memory_images: (60, 384, 4, 7) - > (60, 384, 7, 7) - shape of encoded_images
		reshaped_memory_images = memory_images.reshape(60, 384, 3, 7)
		
		print(f'encoded_images.shape: {encoded_images.shape}')
		print(f'reshaped_memory_images.shape: {reshaped_memory_images.shape}')
		interpolated_memory_images = F.interpolate(reshaped_memory_images, size=(7, 7), mode='bilinear', align_corners=False)
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

		convlstm_result = torch.cat((lstm_output[-1][0], lstm_output[-1][1]), dim=0)
		
		# and here we should reshape (bs, T, ...) -> (bs*T, ...)
		convlstm_result = convlstm_result.reshape(60, 768, 7, 7)
		print(f'convlstm_result.shape right before ImageDecoder: {convlstm_result.shape}')
		output_images_hat = self.ImageDecoder(convlstm_result, skip_connections)
		
		return output_images_hat
	
	def training_step(self, batch, batch_idx):
		masked_ref_images, output_images, landmarks3d = batch
		# reshape
		masked_ref_images = masked_ref_images.view(60, 6, 224, 224)
		output_images = output_images.view(60, 3, 224, 224)
		landmarks3d = landmarks3d.squeeze().view(60, 20, 3)

		output_images_hat = self(masked_ref_images, landmarks3d)
		device = 'cuda'	
		output_images = output_images.float().to(device)
		output_images_hat = output_images_hat.float().to(device)
		print(f'output_images.shape: {output_images.shape}')
		print(f'output_images_hat.shape: {output_images_hat.shape}')
		print(f'output_images.device: {output_images.device}')
		print(f'output_images_hat.device: {output_images_hat.device}')
		vgg_loss = vgg(output_images_hat, output_images, feature_layers=[0, 1, 2, 3], style_layers=[])
		mse_loss = torch.nn.MSELoss()
		l_rec = mse_loss(output_images, output_images_hat) + vgg_loss
		l_d_adv = discriminator_loss(self.discriminator, output_images, output_images_hat)
		l_nr_adv = generator_loss(self.discriminator, output_images_hat)
		# 20*l_rec
		loss = 1*l_rec + 1*l_d_adv + 1*l_nr_adv
		self.log("l_rec", l_rec)
		self.log("l_d_adv", l_d_adv)
		self.log("l_nr_adv", l_nr_adv)
		self.log("loss", loss)	

	def validation_step(self, batch, batch_idx):
		masked_ref_images, output_images, landmarks3d = batch
		# reshape
		masked_ref_images = masked_ref_images.view(60, 6, 224, 224)
		output_images = output_images.view(60, 3, 224, 224)
		landmarks3d = landmarks3d.squeeze().view(60, 20, 3)

		output_images_hat = self(masked_ref_images, landmarks3d)
		device = 'cuda'	
		output_images = output_images.float().to(device)
		output_images_hat = output_images_hat.float().to(device)
		print(f'output_images.shape: {output_images.shape}')
		print(f'output_images_hat.shape: {output_images_hat.shape}')
		print(f'output_images.device: {output_images.device}')
		print(f'output_images_hat.device: {output_images_hat.device}')

		vgg_loss = vgg(output_images_hat, output_images, feature_layers=[0, 1, 2, 3], style_layers=[])

		mse_loss = torch.nn.MSELoss()
		l_rec = mse_loss(output_images, output_images_hat) + vgg_loss
		l_d_adv = discriminator_loss(self.discriminator, output_images, output_images_hat)
		l_nr_adv = generator_loss(self.discriminator, output_images_hat)
		val_loss = 1*l_rec + 1*l_d_adv + 1*l_nr_adv
		self.log("val_loss", val_loss)	

	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
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
		out2 = self.conv2(out1)
		out3 = self.conv3(out2)
		out4 = self.conv4(out3)
		x = self.conv5(out4)

		skip_connections = {
				'skip1': out1,
				'skip2': out2,
				'skip3': out3,
				'skip4': out4
		}

		return x, skip_connections


class ExplicitMem(nn.Module):
	def __init__(self, dropout=0.1):
		super().__init__()
		
		self.LipsEncoder = LipEncoder()
		
		# depending on training video, we would need to use different ExplicitMem
		self.K_nr, self.V_nr = construct_explicitmem(data_dir='/home/avocoral/Downloads/Obamaset/Obama_vid', metadata_dir='/home/avocoral/Downloads/Obamaset/Obama_meta')
		
		self.K_nr = self.K_nr.to('cuda').reshape(300, 60)
		self.V_nr = self.V_nr.to('cuda').reshape(300, 3, 112, 224)

		self.w_q = nn.Parameter(torch.randn(60, 60))
		self.w_k = nn.Parameter(torch.randn(60, 60))
		# self.w_o = nn.Parameter(torch.randn(1, 64))
		self.dropout = nn.Dropout(dropout)
		self.d_k = 60
	
	def forward(self, query):
		# the input of the ExplicitMemory is tensor 25x20x3, 25 frames of landmarks3d_mouth
		# the output is tensor of shape 50x384x8x8, same as output of ImageEncoder/LipsEncoder
		# lips encoder supposed to be the equivalent of the w_v
		print(f'inside EXPLICITMEM, query.shape: {query.shape}')

		#flatten the query 60x20x3 -> 60x60
		query_flat = query.reshape(-1, 60)
		print(f'inside EXPLICITMEM, query_flat.shape: {query_flat.shape}')
		print(f'self.w_q.shape: {self.w_q.shape}')
		q = torch.matmul(query_flat, self.w_q)
		print(f'q.shape: {q.shape}')
		print(f'self.K_nr.shape: {self.K_nr.shape}')
		print(f'self.w_k.shape: {self.w_k.shape}')
		k = torch.matmul(self.K_nr, self.w_k)
		print(f'k.shape: {k.shape}')
		v = self.LipsEncoder(self.V_nr).reshape(300, -1)
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
		return output


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
						nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
						nn.Sigmoid()
				)
	def forward(self, x):
		print(f'discriminator input shape: {x.shape}')
		res = self.net(x)
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

	return adversarial_loss


def generator_loss(discriminator, generated_images):
	# Compute the discriminator output for generated images
	d_generated = discriminator(generated_images)

	# Compute the generator loss
	generator_loss = torch.mean(torch.log(1 - d_generated + 1e-8))

	return generator_loss




if __name__ == '__main__':
	torch.cuda.empty_cache()
	NRmodel = NeuralRender()
	

	datamodule = NeuralRenderingDataModule()
	datamodule.setup()

	train_dataloader = datamodule.train_dataloader()
	val_dataloader = datamodule.val_dataloader()

	trainer = pl.Trainer(default_root_dir='checkpoints', logger=wandb_logger, gpus=[0], accelerator="gpu")
	trainer.fit(NRmodel, train_dataloader, val_dataloader)




