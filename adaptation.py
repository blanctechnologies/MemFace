import torch
from audio2exp import Audio2Exp

audio2exp = Audio2Exp()
audio2exp.load_state_dict(torch.load('/home/avocoral/MemFace/checkpoints/audio2exp/epoch=47-step=1680.ckpt'))
audio2exp.to('cuda:0')
audio2exp.eval()
