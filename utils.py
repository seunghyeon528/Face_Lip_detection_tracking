import imp
from pathlib import Path
import glob 
import os
import json
import cv2
import pdb

def recursive_file_search(root_dir, ext):
    pathname = root_dir + "/**/*" + ext
    file_list = glob.glob(pathname, recursive=True)
    return file_list


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
    dir_name = os.path.dirname(json_out_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(json_out_path, 'w', encoding='utf-8') as make_file:
        json.dump(face_box, make_file, indent="\t")

def save_mp4(vid_path, save_dir, cropped_video_frames, type):
    filename = os.path.basename(vid_path).replace(".mp4", "")
    checkout_path = os.path.join(save_dir, "{}_mp4".format(type), filename)
    checkout_path = checkout_path + ".mp4"
    
    if not os.path.exists(os.path.dirname(checkout_path)):
        os.makedirs(os.path.dirname(checkout_path))

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


