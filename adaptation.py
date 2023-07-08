import torch
from audio2exp import Audio2Exp
from preprocessing import getAudioEncoding
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import subprocess
import librosa
import Audio2ExpDataModule, Audio2ExpDataset
device = 'cuda:0'

audio2exp = Audio2Exp()
checkpoints = torch.load('/root/MemFace/MemFace/version_None/checkpoints/model-epoch=360.ckpt')
audio2exp.load_state_dict(checkpoints["state_dict"])
audio2exp.to('cuda:0')
audio2exp.eval()

# extract audio from video
def extract_audio():
    vid_filepath = '/root/Obama_vid.mp4'
    audio_filepath = '/root/Obama_vid.wav'
    cmd = f'ffmpeg -i {vid_filepath} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_filepath} -hide_banner -loglevel error'
    subprocess.run([cmd], shell=True)

audio_filepath = '/root/Obama_vid.wav'

# wav2vec2 on audio
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
sample_rate = 16000
waveform, _ = librosa.load(audio_filepath, sr=sample_rate)
transcript, all_logits = getAudioEncoding(waveform, processor, model)
audio_embedding_filepath = '/root/Obamaset/Obama_audio/Obamaset_audioencoding.pt'
torch.save(all_logits, audio_embedding_filepath)
print(f'transcript: {transcript}')

# create datamodule for adaptation from first 30 seconds of Obamaset
datamodule = Audio2ExpDataModule(stage='adaptation', audio_dir='/root/Obamaset/Obama_audio', coeff_dir='/root/Obamaset/Obama_vid')
datamodule.setpu()
train_dataloader = datamodule.train_dataloader()

# fine-tune audio2exp on that datamodule
optimizer = torch.optim.Adam(audio2exp.parameters(), lr=5e-6)
trainer = pl.Trainer(strategy = DDPStrategy(find_unused_parameters=True), checkpoint_callback=True, default_root_dir='adaptation_checkpoints', max_epochs=200, optimizer=optimizer, logger=wandb_logger, accelerator="gpu", devices=1)
trainer.fit(audio2exp, train_dataloader, val_dataloader)

# take audio embeddings for the last 30 seconds, do inference on it and combine new exp with original pose,shape


# save new ref frames with new exp, old pose, old shape
# make a video of it
# add audio to it


