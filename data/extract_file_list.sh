#!/usr/bin/env bash
root_dir="/home/nas4/user/lsh/Lip_Face_ROI_labelling/out/lip_fail/lip_mp4/"
save_path="./lip_success_v1.txt"    
file_type=".mp4"

echo ${root_dir}
find "${root_dir}" -name '*'${file_type} > "${save_path}"