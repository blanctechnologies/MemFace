import os
from pathlib import Path
from tqdm import tqdm
import subprocess
from os import listdir


num_workers = 12

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
			cmd = f'ffmpeg -i {vid_filepath} -c copy -map 0:a {audio_filepath} -hide_banner -loglevel error'
			subprocess.run([cmd], shell=True)
			k += 1
			
			print(f"vid #{k}/{num_files} is ready!")
		k = 0
		print(f"folder #{j}/{num_folders} is ready!")
		j += 1


def extract_exp():
	# extracts exp.npy for each frame of each video


if __name__ == '__main__':
	extract_audio()
