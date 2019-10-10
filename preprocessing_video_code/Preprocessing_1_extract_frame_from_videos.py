import numpy as np
import cv2
import os
from os.path import join as pjoin
from multiprocessing.pool import ThreadPool


#%%
data_dir='C:\\Segmented_videoClips\\'
target_folder='C:\\New_video_images\\'

def exec_command(command):
	print(command)
	os.system(command)

pool = ThreadPool(processes=6)

for img_dir in os.listdir(data_dir):
	video_number=str(img_dir).split('.')[0]
	directory=target_folder+video_number
	if not os.path.exists(directory):
		os.makedirs(directory)

	video=data_dir+str(img_dir)
	# print(video)
	command='ffmpeg -i '+video+' -vf "scale=640:360,fps=10" '+directory+'\\'+video_number+'_%d.jpg'
	pool.apply_async(exec_command, (command, ))

pool.close() 
pool.join() 
	
	# os.system(command1)
    # each_dir= pjoin(data_dir, img_dir)
    # out_dir=pjoin(out, img_dir)