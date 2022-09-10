from configparser import Interpolation
import enum
import imp
from pathlib import Path
import glob 
import os
import json
import cv2
import pdb
import datetime
import logging

def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError
    
def get_logger(save_dir_path,file_name):
    log_path = 'log/{}/{}.txt'.format(save_dir_path,file_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger

def recursive_file_search(root_dir, ext):
    pathname = root_dir + "/**/*" + ext
    file_list = glob.glob(pathname, recursive=True)
    return file_list

def save_args(args):
    file_path = os.path.join(args.save_dir, 'args.txt')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file = open(file_path, "w")
    for k, v in vars(args).items():
        file.write(f"{k}:\t {v}\n")
    file.close()
    

def save_json(vid_path, save_dir, input_boxes, type):
    face_box = dict()
    if type=="face":
        face_box['Face_bounding_box']={}
        face_box['Face_bounding_box']['xtl_ytl_xbr_ybr']=input_boxes
    if type=="lip":
        face_box['Lip_bounding_box']={}
        face_box['Lip_bounding_box']['xtl_ytl_xbr_ybr']=input_boxes

    filename = os.path.basename(str(vid_path)).replace(".mp4", ".json")
    json_out_path = os.path.join(save_dir, "{}_json".format(type), filename)
    os.makedirs(os.path.dirname(json_out_path), exist_ok=True)

    with open(json_out_path, 'w', encoding='utf-8') as make_file:
        json.dump(face_box, make_file, indent="\t")


def save_mp4(vid_path, save_dir, cropped_video_frames, type, remain_path_depth):
    # filename = os.path.basename(vid_path).replace(".mp4", "")
    remain_path = os.sep.join(vid_path.rsplit("/")[-remain_path_depth:]).replace(".mp4", "")
    checkout_path = os.path.join(save_dir, "{}_mp4".format(type), remain_path)
    checkout_path = checkout_path + ".mp4"
    os.makedirs(os.path.dirname(checkout_path),exist_ok=True)

    resize_face = (224,224)
    fps = 30
    out = cv2.VideoWriter(
            checkout_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            resize_face,
        )

    for k in range(len(cropped_video_frames)):
        out.write(cropped_video_frames[k])
    out.release()


def save_image(vid_path, save_dir, cropped_video_frames, type, remain_path_depth):
    remain_path = os.sep.join(vid_path.rsplit("/")[-remain_path_depth:]).replace(".mp4", "")
    checkout_path = os.path.join(save_dir, "{}_image".format(type), remain_path)
    
    for i, frame in enumerate(cropped_video_frames):
        img_path = os.path.join(checkout_path, "{}_{}".format(str(i),str(i).zfill(5))) + ".png"
        resized_frame = cv2.resize(frame, dsize=(224,224), interpolation=cv2.INTER_LINEAR)
        os.makedirs(os.path.dirname(img_path),exist_ok=True)
        cv2.imwrite(img_path, resized_frame)