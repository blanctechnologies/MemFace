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

device = 'cuda:0'
plugin = DDPPlugin(find_unused_parameters=True)
wandb_logger = WandbLogger(name='Audio2ExpAdaptation',project='MemFace')

audio2exp = Audio2Exp()
checkpoints = torch.load('/home/avocoral/MemFace/checkpoints/audio2exp/model-epoch=360.ckpt')
audio2exp.load_state_dict(checkpoints["state_dict"])
audio2exp.to('cuda:0')
audio2exp.eval()

# extract audio from video
def extract_audio():
    vid_filepath = '/home/avocoral/Downloads/Obamaset/Obama_vid.mp4'
    audio_filepath = '/home/avocoral/Downloads/Obamaset/Obama_vid.wav'
    cmd = f'ffmpeg -y -i {vid_filepath} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_filepath} -hide_banner -loglevel error'
    subprocess.run([cmd], shell=True)

# extract_audio()
# audio_filepath = '/home/avocoral/Downloads/Obamaset/Obama_vid.wav'

# wav2vec2 on audio
def extractAudioEncoding(audio_filepath, audio_embedding_filepath):
		processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
		model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
		sample_rate = 16000
		waveform, _ = librosa.load(audio_filepath, sr=sample_rate)
		transcript, all_logits = getAudioEncoding(waveform, processor, model)
		audio_embedding_filepath = '/home/avocoral/Downloads/Obamaset/Obama_audio/Obama.pt'
		torch.save(all_logits, audio_embedding_filepath)
		print(f'transcript: {transcript}')

# create datamodule for adaptation from first 30 seconds of Obamaset
datamodule = Audio2ExpDataModule(audio_dir='/home/avocoral/Downloads/Obamaset/Obama_audio', coeff_dir='/home/avocoral/Downloads/Obamaset/Obama_vid/Obama')
datamodule.setup('fit')

# fine-tune audio2exp on that datamodule
train_dataloader = datamodule.train_dataloader()
# val_dataloader = datamodule.val_dataloader()
# max_epochs=200,
trainer = pl.Trainer(plugins=plugin, checkpoint_callback=True, default_root_dir='adaptation_checkpoints', logger=wandb_logger, accelerator="gpu", devices=1)
trainer.fit(audio2exp, train_dataloader)

# take audio embeddings for the last 30 seconds, do inference on it and combine new exp with original pose,shape


# save new ref frames with new exp, old pose, old shape
# make a video of it
# add audio to it


