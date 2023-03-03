import os
from pathlib import Path
from tqdm import tqdm
import subprocess
from os import listdir
import librosa
# import torch
# import torch.multiprocessing as mp
# import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import face_recognition
import face_alignment
import imageio

import sys
from pathlib import Path
from tqdm import auto
import multiprocessing as mp

DATA_DIR = Path('/mnt/sda/AVSpeech')
AUDIO_DIR = DATA_DIR / 'audio'
AUDIO_ENCODINGS_DIR = DATA_DIR / 'audio_encodings'
TRANSCRIPT_DIR = DATA_DIR / 'transcripts'


sys.path.insert(0, '/home/avocoral/MemFace')
# torch.random.manual_seed(42)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = "cuda"
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
	# we check just by the first frame to save time.
	if fa == None:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
	reader = imageio.get_reader(filepath)
	try:
		for i, frame in enumerate(reader):
			bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
			print(f"frame {i}, num of faces: {len(bboxes)}")
			if len(bboxes) > 1:
				print(f"this file {filepath} has multiple faces")
				with open(f'/home/avocoral/MemFace/multiface_xab/multiface_xab_{worker_id}', 'a') as f:
					f.write(f'{filepath}\n')
				return True
			return False
	except IndexError:
		None

def get_files_multiface_worker(worker_id):
	num_workers = 16
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
	root_dir = '/mnt/sda/AVSpeech/clips_unpacked_test/xab/'
	num_folders = len(os.listdir(root_dir)) // num_workers
	for i, folder in enumerate(os.listdir(root_dir)):
		if i % num_workers == worker_id and folder != '.DS_Store':
			folderpath = os.path.join(root_dir, folder)
			num_files = len(os.listdir(folderpath))
			j = 0
			for video_file in os.listdir(folderpath):
				if video_file != '.@__thumb' and video_file != '.DS_Store':
					get_files_multiface(os.path.join(folderpath, video_file), worker_id = worker_id, fa=fa)
				print(f'worker {worker_id}: video #{j}/{num_files} checked, folder #{i}/{num_folders}')
				j+=1


def get_files_multiface_scheduler():
	# torch.multiprocessing.set_start_method('spawn', force=True)
	num_workers = 16
	with mp.Pool(num_workers) as p:
			print(p)
			p.map(get_files_multiface_worker, range(num_workers))


def assemble_list(dir_name):
	result_filename = '/home/avocoral/MemFace/multiface_xab.txt'
	result = ''
	total_number_of_files = 0
	result_list = []
	for filename in os.listdir(dir_name):
		with open(os.path.join(dir_name, filename)) as f:
			text = f.readline()
			filenames_noext = text.split('.mp4')
			last_occurrence = ''
			for name in filenames_noext:
				total_number_of_files += 1
				new_name = name.rsplit("/", 1)[0]
				if new_name != last_occurrence and new_name != '' and new_name != None:
					result += f'{new_name}\n'
					result_list.append(new_name)
				last_occurrence = new_name
	
	print(f'total number of multiface files: {total_number_of_files}')
	with open(result_filename, 'a') as f:
		f.write(result)
	
	return result_list

def readFPSDatacard(filepath = '/home/avocoral/super8/Stoven/fps_xaa.txt'):
		total_number_of_files = len(os.listdir('/mnt/sda/AVSpeech/clips_unpacked_test/xaa'))
		with open(filepath, 'r') as f:
			# last_line = len(f.readlines())
			not30fps = []
			for line in f.readlines():
				# print(line)
				if line != '':
					try:
						name = line.rsplit('-', 1)[0][:-1]
						fps = line.rsplit('-', 1)[1][1:]
						# print(f'name: {name}, fps: {fps}')
						if round(float(fps), 0) != 30.0:
							not30fps.append(name)
							# print('this is not 30 fps :/')
					except Exception as e:
						print(f'this is the line! -> {line}')
		print(f'making sure fps worked correctly, total not30fps len:{len(not30fps)}/{total_number_of_files}')
		return not30fps[:-1].copy()

			
def get_all_faulty_vids():
	multiface_vid = []
	with open('/home/avocoral/MemFace/multiface_xaa.txt', 'r') as f:
		for filename in f.readlines():
			multiface_vid.append(filename[:-1].rsplit('/', 1)[1])
	
	total_number_of_files = len(os.listdir('/mnt/sda/AVSpeech/clips_unpacked_test/xaa'))
	not30fps = readFPSDatacard()
	total_faulty_vids = list(dict.fromkeys(multiface_vid + not30fps))
	print(f'total faulty vid len: {len(total_faulty_vids)}/{total_number_of_files}')
	print(f'multiface_vid[0]: {multiface_vid[0]}')
	print(f'not30fps[0]: {not30fps[0]}')
	preprocessed_vids_dir = '/mnt/sda/AVSpeech/video'
	faulty_xab_dir = '/mnt/sda/AVSpeech/xaa_faulty'
	preprocessed_vids = os.listdir(preprocessed_vids_dir)
	faulty_preprocessed_vids = []
	for filename in preprocessed_vids:
		new_filename = filename.rsplit('_',1)[0]
		if new_filename in total_faulty_vids:
			faulty_preprocessed_vids.append(filename)
	
	len_faulty_preprocessed_vids = len(faulty_preprocessed_vids)
	print(f'total prep faulty vids out of all prep vids: {len_faulty_preprocessed_vids}/{len(preprocessed_vids)}')
	# move all faulty files to the faulty_xab_dir from video_extra:
	i = 0
	for filename in faulty_preprocessed_vids:
		location = os.path.join(preprocessed_vids_dir, filename)
		destination = os.path.join(faulty_xab_dir, filename)
		subprocess.run([f'mv {location} {destination}'], shell=True)
		print(f'{i}/{len(faulty_preprocessed_vids)} has been moved')
		i+=1
	
	
def get_dataset_length(preprocessed_vids_dir, orig_dir):
	total_length = 0
	i = 0
	total_len = len(os.listdir(preprocessed_vids_dir))
	for filename in os.listdir(preprocessed_vids_dir):
		orig_filename = os.path.join(orig_dir, filename.rsplit('_',1)[0], filename+'.mp4')
		length = video_length_seconds(orig_filename)
		total_length += length
		print(f'{i}/{total_len} done')
		i+=1
	print(f'total length of {preprocessed_vids_dir}: {total_length}')


def video_length_seconds(filename):
	result = subprocess.run(
		[
			"ffprobe",
			"-v",
			"error",
			"-show_entries",
			"format=duration",
			"-of",
			"default=noprint_wrappers=1:nokey=1",
			"--",
			filename,
		],
		capture_output=True,
		text=True,
	)
	try:
		return float(result.stdout)
	except ValueError:
		raise ValueError(result.stderr.rstrip("\n"))



def face_mask(root_dir):
	return None





if __name__ == '__main__':
	# extract_audio()
	# extractAudioEncodings()
	# input_video = '02uzUf1LilE_10.mp4'
	# output_folder = '/home/avocoral/MemFace/02uzUf1LilE_10.mp4'
	# extractCoeff(input_video, output_folder)
	# get_files_multiface('/mnt/sda/AVSpeech/clips_unpacked_test/xaa/02uzUf1LilE/02uzUf1LilE_0.mp4')
	# get_files_multiface_scheduler()
	# print(f"starting to assemble the list")
	# assemble_list(dir_name='/home/avocoral/MemFace/multiface_xab')
	# readFPSDatacard()
	# all_faulty_files = get_faulty_video_list()
	# file_dir = ""
	# get_faulty_existing_video_list(all_faulty_files, file_dir)
	# get_all_faulty_vids()
	preprocessed_vids_dir = '/mnt/sda/AVSpeech/video_extra'
	orig_dir = '/mnt/sda/AVSpeech/clips_unpacked_test/xab' 
	get_dataset_length(preprocessed_vids_dir, orig_dir)




