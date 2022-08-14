# mp4_cropped 를 포함하는 line 을 npy_full_list_txt 에서 찾기
import os
import glob
import subprocess
import pdb
from tqdm import tqdm

mp4_cropped_txt_path = "lip_success_v1.txt"
with open(mp4_cropped_txt_path, "r") as f:
    mp4_cropped_list = f.readlines()
mp4_cropped_list = [x.strip() for x in mp4_cropped_list]

npy_full_list_txt="lip_fail_file_full_path_list_v2.txt"


for dir_name in tqdm(mp4_cropped_list):   
    dir_name = dir_name.replace("/","\/")  
    #command = ("sed -i \"s/%s/""/g\" \"%s\""%(dir_name, npy_full_list_txt))
    command = ("sed -i \"/%s/d\" \"%s\""%(dir_name, npy_full_list_txt))
    subprocess.call(command, shell=True, stdout=None)
    
