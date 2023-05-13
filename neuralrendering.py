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

# Load a pre-trained VGG16 model
vgg = models.vgg16(pretrained=True).features[:23]
vgg.eval()

class NeuralRender(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.save_hyperparameters()

		self.N = None
		self.d_k = None
		self.d_v = None
		self.keys = None 
		self.values = None

		self.ImageEncoder = ImageEncoder()
		self.ExplicitMem = ExplicitMem()
		self.lstm = ConvLSTM()
		self.ImageDecoder = ImageDecoder()
		self.discriminator = Discriminator()
	
	def forward(self, masked_ref_images, landmarks3d):
		encoded_images = self.ImageEncoder(masked_ref_images)
		memory_images = self.ExplicitMem(landmarks3d)

		final_encoded_images = encoded_images + memory_images
		output_images_hat = self.ImageDecoder(self.lstm(final_encoded_images))
		
		return output_images_hat
	
	def training_step(self, batch, batch_idx):
		masked_ref_images, output_images, landmarks3d = batch
		output_images_hat = self(masked_ref_images)
		
		mse_loss = torch.nn.MSELoss()
		l_rec = mse_loss(output_images, output_images_hat) + mse_loss(vgg(output_images), vgg(output_images_hat))
		l_d_adv = discriminator_loss(d, output_images, output_images_hat)
		l_nr_adv = generator_loss(d, output_images_hat)
		l_total = 20*l_rec + 1*l_d_adv + 1*l_nr_adv
		self.log("loss", loss)	

	def validation_step(self, batch, batch_idx):
		masked_ref_images, output_images, landmarks3d = batch
		output_images_hat = self(masked_ref_images)
		
		mse_loss = torch.nn.MSELoss()
		l_rec = mse_loss(output_images, output_images_hat) + mse_loss(vgg(output_images), vgg(output_images_hat))
		d = self.discriminator()
		l_d_adv = discriminator_loss(d, output_images, output_images_hat)
		l_nr_adv = generator_loss(d, output_images_hat)
		l_total = 20*l_rec + 1*l_d_adv + 1*l_nr_adv
		self.log("val_loss", loss)	
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
		return optimizer


class ImageEncoder(nn.Module):
	def __init__(self):
		self.conv1 = Conv2dBlock(in_channels=6, 48, kernel_size=5, stride=2, padding=2)
		self.conv2 = Conv2dBlock(48, 96, kernel_size=4, stride=2, padding=1)
		self.conv3 = Conv2dBlock(96, 192, kernel_size=4, stride=2, padding=1)
		self.conv4 = Conv2dBlock(192, 384, kernel_size=4, stride=2, padding=1)
		self.conv5 = Conv2dBlock(384, 384, kernel_size=4, stride=2, padding=1)
		
		
	def forward(self, x):
		x = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)(x)
		return x


class ExplicitMem(nn.Module):
	def __init__(self, dropout=0.1):
		super().__init__()
		
		self.LipsEncoder = LipsEncoder()
		self.K_nr, self.V_nr = construct_explicitmem()

		self.w_q = nn.Parameter(torch.randn(60, 60))
		self.w_k = nn.Parameter(torch.randn(60, 60))
		self.w_v = nn.Parameter(torch.randn(64, 64))
		self.w_o = nn.Parameter(torch.randn(1, 64))
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, query):
		# the input of the ExplicitMemory is tensor 25x20x3, 25 frames of landmarks3d_mouth
		# the output is tensor of shape 50x384x8x8, same as output of ImageEncoder/LipsEncoder
		# lips encoder supposed to be the equivalent of the w_v
		print(f'inside EXPLICITMEM, query.shape: {query.shape}')

		#flatten the query 25x1x20x3 -> 25x60
		query_flat = query.reshape(-1, 60)
		print(f'inside EXPLICITMEM, query_flat.shape: {query_flat.shape}')
		q = torch.matmul(query, self.w_q)
		k = torch.matmul(self.keys, self.w_k)
		v = torch.matmul(self.values, self.w_v)
		
		# does it mean that instead of v = torch.matmul(self.values, self.w_v)
		# we use self.LipsEncoder(self.)

		# - mask needs to be applied as well for scoresd
		# ? is k.transpose(-2, -1) same as k.T
		
		# ? how to get mask
		# - perhaps we need to use lengths
		# if we matmul padded tensors in q with k, it will zero out
		scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
		
		# and for softmax we do need mask, the mask will set the padded elements to a very large negative value
		mask = (scores != 0).float()
		attention_weights = torch.softmax(scores, dim=-1)
		attention = torch.matmul(attention_weights, v)
		output = attention * self.w_o
		output = self.dropout(attention)



class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
	def forward(self, x):
		return self.net(input)



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


class ConvLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(ConvLSTMCell, self).__init__()
		self.hidden_size = hidden_size
		self.conv = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size=3, stride=1, padding=1, bias=True)
	
	def forward(self, input_, state):
		# unpack the previous state
		hx, cx = state
		
		# concatenate the input and hidden state along the channel dimension
		combined = torch.cat((input_, hx), dim=1)
		
		# apply the convolutional operation to the combined tensor
		gates = self.conv(combined)
		
		# split the convolutional output into separate tensors
		i, f, o, g = gates.chunk(4, dim=1)
		
		# apply the activation functions
		input_gate = torch.sigmoid(i)
		forget_gate = torch.sigmoid(f)
		output_gate = torch.sigmoid(o)
		cell_gate = torch.tanh(g)
		
		# compute the new cell and hidden state
		cy = forget_gate * cx + input_gate * cell_gate
		hy = output_gate * torch.tanh(cy)
		
		# pack the new state into a tuple
		state = (hy, cy)
		
		return hy, state


class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell(input_size, hidden_size)
    
    def forward(self, input_, state=None):
        # initialize the hidden and cell states if they are not provided
        if state is None:
            batch_size, _, height, width = input_.size()
            state = (torch.zeros(batch_size, self.cell.hidden_size, height, width, device=input_.device),
                     torch.zeros(batch_size, self.cell.hidden_size, height, width, device=input_.device))
        
        # run the input through the LSTM cell
        output, state = self.cell(input_, state)
        
        return output, state


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

	def forward(self, x1, x2, x3, x4, x5):
		x = self.dconv1(x5)
		x = torch.cat((x, x4), dim=1)
		x = self.dconv2(x)
		x = torch.cat((x, x3), dim=1)
		x = self.dconv3(x)
		x = torch.cat((x, x2), dim=1)
		x = self.dconv4(x)
		x = torch.cat((x, x1), dim=1)
		x = self.dconv5(x)
		x = self.dconv6(x)
		return x


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


def discriminator_loss(d, real_images, generated_images):
	# Compute the discriminator output for real and generated images
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
