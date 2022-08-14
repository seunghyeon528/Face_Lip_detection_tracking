# remainder_txt 를 포함하는 line 을 mp4_full_list_txt 에서 찾기
import os
import glob
import subprocess
import pdb
txt_path = "lip_fail_file_list.txt"
with open(txt_path, "r") as f:
    remainder_txt_list = f.readlines()
remainder_txt_list = [x.strip() for x in remainder_txt_list]

mp4_full_list_txt="66420_full_path_list.txt"
save_path = "lip_fail_file_full_path_list.txt"
with open(save_path, "w") as f:
    pass

for dir_name in remainder_txt_list:
    dir_name = dir_name.replace("/","\/")  
    
    command = ("sed -n \"/%s/p\" \"%s\""%(dir_name, mp4_full_list_txt))
    output = subprocess.check_output(command, shell=True)
    output = output.decode("utf-8")
    
    with open(save_path, "a") as f:
        f.write(output)