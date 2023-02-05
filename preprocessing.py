import os
from pathlib import Path
from tqdm import tqdm
import subprocess
from os import listdir
import librosa
import torch
import torch.multiprocessing as mp
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import face_recognition
import face_alignment
import imageio

import sys
from pathlib import Path
from tqdm import auto


DATA_DIR = Path('/mnt/sda/AVSpeech')
AUDIO_DIR = DATA_DIR / 'audio'
AUDIO_ENCODINGS_DIR = DATA_DIR / 'audio_encodings'
TRANSCRIPT_DIR = DATA_DIR / 'transcripts'


sys.path.insert(0, '/home/avocoral/MemFace')
torch.random.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cuda"
num_workers = 6
sample_rate = 16000

def extract_audio():
	# extracts .wav audio from .mp4 files and moves all into one dir
	DATA_DIR = Path('/mnt/sda/AVSpeech/clips_unpacked_test/xaa')
	OUTPUT_DIR = '/mnt/sda/AVSpeech/audio'

	k = 0
	j = 0
	num_folders = len(listdir(DATA_DIR))
	for vid in listdir(DATA_DIR):
		if vid == '.DS_Store':
			continue
		vid_folder = os.path.join(DATA_DIR, vid)
		num_files = len(listdir(vid_folder))
		for vid_sample in listdir(vid_folder):
			if vid_sample == '.@__thumb' or vid_sample == '.DS_Store':
				continue
			vid_filepath = os.path.join(vid_folder, vid_sample)
			audio_filename = vid_sample.split('.')[0]+'.wav'
			audio_filepath = os.path.join(OUTPUT_DIR, audio_filename)
			cmd = f'ffmpeg -i {vid_filepath} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_filepath} -hide_banner -loglevel error'
			subprocess.run([cmd], shell=True)
			k += 1
			
			print(f"vid #{k}/{num_files} is ready!")
		k = 0
		print(f"folder #{j}/{num_folders} is ready!")
		j += 1


def getAudioEncodingWorker(worker_id):
		# torch.set_num_threads(4)
		print(f"worker #{worker_id} spawned")
		# read audiofile, get waveform
		processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
		model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
		num_workers = 6
		sample_rate = 16000
		audio_list = listdir(AUDIO_DIR)
		
		for i in tqdm(range(len(audio_list))):
				if i % num_workers == worker_id:
						name = audio_list[i]
						# print(f"extracting {name}")
						audio_location = AUDIO_DIR / name
						audio_encoding_location = AUDIO_ENCODINGS_DIR / (name.split('.')[0] + '.pt')
						transcript_location = TRANSCRIPT_DIR / (name.split('.')[0] + '.txt')
						try:
								waveform, _ = librosa.load(audio_location, sr=sample_rate)
						except Exception as e:
								print(f"error {e} on loading {name}")
						transcript, all_logits = getAudioEncoding(waveform, processor, model)
						# print(f'transcript: {transcription}')
						# print(f'len of logits: {len(all_logits)}')
						# audio_encoding = torch.stack((transcription, all_logits), 0)
						torch.save(all_logits, audio_encoding_location)
						with open(transcript_location, 'w') as f:
								f.write(transcript)


def getAudioEncoding(waveform, processor, model):
		# Common Voice average clip duration is 4.7 seconds, VoxCeleb duration 145.0 / 8.2 / 4.0 (max, avg, min), AVSpeech 5-10 seconds
		chunk_duration = 5 # sec
		padding_duration = 1 # sec
		chunk_len = chunk_duration*sample_rate
		number_of_chunks = len(waveform) // chunk_len + ((len(waveform) % chunk_len) != 0)
		input_padding_len = int(padding_duration*sample_rate)
		output_padding_len = model._get_feat_extract_output_lengths(input_padding_len)
		all_preds = []
		all_logits = []
		
		for start in range(number_of_chunks):
				if start == 0:
						chunk = waveform[:chunk_len+input_padding_len]
						input_values = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_values
						with torch.no_grad():
								if len(chunk) <= chunk_len:
										logits = model(input_values.to(device)).logits[0]
								else:
										logits = model(input_values.to(device)).logits[0][:-output_padding_len]
				elif len(waveform) - start*chunk_len <= chunk_len:
						chunk = waveform[start*chunk_len-input_padding_len:]
						input_values = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_values
						with torch.no_grad():
								logits = model(input_values.to(device)).logits[0][output_padding_len:]
				else:
						chunk = waveform[start*chunk_len-input_padding_len : start*chunk_len+chunk_len+input_padding_len]
						input_values = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_values
						with torch.no_grad():
								logits = model(input_values.to(device)).logits[0][output_padding_len:-output_padding_len]

				all_logits.append(logits)
				predicted_ids = torch.argmax(logits, dim=-1)
				all_preds.append(predicted_ids.cpu())
		
		transcription=processor.decode(torch.cat(all_preds))
		
		return transcription, all_logits


def getAudioEncodingSimplified(audio_location):
		processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
		model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
		sample_rate = 16000
		
		waveform, _ = librosa.load(audio_location, sr=sample_rate)
		

def extractAudioEncodings():
		torch.multiprocessing.set_start_method('spawn', force=True)
		num_workers = 6

		# Paralle the execution of a function across multiple input values
		with mp.Pool(processes=num_workers) as p:
				print(p)
				p.map(getAudioEncodingWorker, range(num_workers))


def get_files_multiface(filepath, worker_id, fa=None):
	if fa == None:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
	reader = imageio.get_reader(filepath)
	try:
		for i, frame in enumerate(reader):
			bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
			print(f"frame {i}, num of faces: {len(bboxes)}")
			if len(bboxes) > 1:
				print(f"this file {filepath} has multiple faces")
				with open(f'multiface_{worker_id}', a) as f:
					f.write(f'{filepath}')
	except IndexError:
		None

def get_files_multiface_worker(worker_id):
	num_workers = 8
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
	root_dir = '/mnt/sda/AVSpeech/clips_unpacked_test/xaa/'
	for i, folder in enumerate(os.list(root_dir)):
		if i % num_workers == worker_id and folder != '':
			folderpath = os.path.join(root_dir, folder)
			for video_file in os.list(folderpath):
				if video_file != '' and video_file != '':
					get_files_multiface(os.path.join(folderpath, video_file), worker_id = worker_id, fa=fa)

def get_files_multiface_scheduler():
	torch.multiprocessing.set_start_method('spawn', force=True)
	num_workers = 8
	with mp.Pool(processes=num_workers) as p:
			print(p)
			p.map(get_files_multiface_worker, range(num_workers))



def face_mask(root_dir):
	return None





if __name__ == '__main__':
	# extract_audio()
	# extractAudioEncodings()
	# input_video = '02uzUf1LilE_10.mp4'
	# output_folder = '/home/avocoral/MemFace/02uzUf1LilE_10.mp4'
	# extractCoeff(input_video, output_folder)
	get_files_multiface('/mnt/sda/AVSpeech/clips_unpacked_test/xaa/02uzUf1LilE/02uzUf1LilE_0.mp4')





