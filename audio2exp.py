import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import pytorch_lightning as pl


pl.seed_everything(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class audio2exp(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.save_hyperparameters()


		self.encoder = encoder
		self.implicitmem = implicitmem
		self.decoder = decoder # f_dec
		self.query = None # f_enc(A), dim = T x d_k
		self.keys = None # dim = m x h_a
		self.values = None # dim = m x h_a
		self.m = 1000 # number of keys and values => output of f_enc is 1000

	def forward(self, a, mask=None, return_attn=None):
		# transformer with single head scaled dot product attn	
		scaled_dot_product()
		return

	def training_step(self, batch, batch_idx):
		return loss

	def configure_optimizers(self):
		return optimizer
	
	def scaled_dot_product(self, q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention
		
		
