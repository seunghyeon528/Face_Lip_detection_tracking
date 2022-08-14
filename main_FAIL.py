################################################################################
#   lip ROI mp4 extract 안 된 (FAIL) 애들 
#   face_corrected_boxes / lip_corrected_boxes => 한번 xmin,xmax,ymin,ymax 대소관계 check
################################################################################

import datetime
import argparse
from modulefinder import IMPORT_NAME
from tqdm import tqdm
import pdb
import os
import statistics
import multiprocessing
from functools import partial
import matplotlib
import cv2
import numpy as np 
import yaml
import sys
from pathlib import Path
import time

from utils import recursive_file_search, find_option_type
from utils import save_mp4, save_json, get_logger
from utils_vid import load_reader, load_detector
from utils_vid import enlarge_box, get_corrected_boxes, crop_video, check_bbox_min_max

###############################################################################
##                                      ARGS
###############################################################################
parser = argparse.ArgumentParser(description='Face cropping')

## -- CONFIG
parser.add_argument('--config',         type=str,   default="./configs/NIA_LIP_FAIL_180_CV_30_v3.yaml",   help='Config YAML file')

## -- INPUT
parser.add_argument('--file_search', type = bool, default=True,help='recursive file search') # True -> use videos at root-dir, False -> use data list txt path
parser.add_argument('--root_dir', type = str, default="./input/MOBIO", help='root directory of input files')  
parser.add_argument('--data_list_txt_path', type = str, default="./data/uhd_list.txt",help='recursive file search') 

## -- OUTPUT
parser.add_argument('--save_dir', type = str, default="./bbox_rework_out/MOBIO/CV/", help='root directory of output files')  
parser.add_argument('--save_mp4', type = bool, default=True, help='save cropped mp4 or not')  
parser.add_argument('--save_json', type = bool, default=True, help='save json containing labelling points or not')  

## -- RUN ENVRIONEMNT
parser.add_argument('--multiprocessing', type = bool, default=False, help='')  
parser.add_argument('--multi_process_num', type = int, default=2, help='')  

parser.add_argument('--gpu', type = bool, default=True, help='True -> use GPU for face detector')  
parser.add_argument('--gpu_num', type = str, default='0', help='')  

parser.add_argument('--short_test', type = bool, default=False,help='True -> 30sec test') 

## -- ALGORHITHM
parser.add_argument('--shift_frame', type =int, default= 90, help='per each shift frame size, detection executed')  
parser.add_argument('--opencv_tracker_type', type = str, default="medianflow", help='one of medianflow/csrt/mosse/mil/kcf/boosting/tld/goturn') 

parser.add_argument('--lip_detect_enlarge', type =float, default=1,help='enlarge detection box before put into tracker') # LIP X 2 before tracking
parser.add_argument('--lip_enlarge', type =float, default=0,help='enlarge proportion')
parser.add_argument('--face_enlarge', type =float, default=1,help='enlarge proportion') # FACE X 2

parser.add_argument('--resize', type =bool, default=False,help='resize before put into detector / tracker') # UHD -> HD  / Detection fail 
parser.add_argument('--resize_ratio', type =float, default=0.5,help='resize proportion') # UHD -> HD  / Detection fail 

args = parser.parse_args()          




#################################################################################
##                                PROCESS LOOP
#################################################################################

def fa_DETECTION(resized_frame, fa, lip_enlarge_ratio):
    try:
        pred, probs = fa.get_landmarks(resized_frame)
        overlapped_list = []
        #################################################################
        if len(probs) >= 1:
            for prob in probs:
                overlapped_list.append(prob)
            min_index=overlapped_list.index(max(overlapped_list))
            pred=[pred[min_index]]
        pred = np.squeeze(pred)
        #################################################################
        
        ## -- FACE
        x = pred[:,0]
        y = pred[:,1]

        min_x = min(x)
        min_y = min(y)

        width = max(x) - min_x
        height = max(y) - min_y
        
        box = [int(min_x), int(min_y), int(width), int(height)]

        ## -- LIP 
        lip_x = pred[48:,0]
        lip_y = pred[48:,1]

        lip_center_x = statistics.median(lip_x)
        lip_center_y = statistics.median(lip_y)
        lip_width = max(lip_x) - min(lip_x)
        lip_height = max(lip_y) - min(lip_y)

        enlarged_lip_width = lip_width * (1+lip_enlarge_ratio)
        enlarged_lip_height = lip_height * (1+lip_enlarge_ratio)
        lip_min_x = lip_center_x - int(enlarged_lip_width/2)
        lip_min_y = lip_center_y - int(enlarged_lip_height/2)
        
        lip_box = [int(lip_min_x), int(lip_min_y), int(enlarged_lip_width), int(enlarged_lip_height)]
    except:
        box, lip_box =  None, None
    return box, lip_box


def process(args,error_logger,success_logger,vid_path):
    
    # Device Cofiguration
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_num
    device = "cuda" if args.gpu else "cpu"

    # Load Video reader / Tracker / Detector
    '''
    Libraries used ****

        Reader : skvideo 
        Tracker : OPENCV 
        Detector : face_alignment (https://github.com/1adrianb/face-alignment)
    '''
    reader, (num_frames, h, w, c) = load_reader(vid_path)
    video_shape =  (num_frames, h, w, c) 
    #face_tracker, lip_tracker = load_tracker(vid_path)
    fa = load_detector(device) 
    # pdb.set_trace()
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create,
        "goturn":cv2.TrackerGOTURN_create
    }

    try:
        start_time = time.time()

        ## --  variables
        frame_temp_list = []
        lip_temp_bboxes = []
        face_temp_bboxes = []
        
        face_label_list = []
        lip_label_list = []
        
        face_cropped_frame_list = []
        lip_cropped_frame_list = []
        
        ## -- Walk every frame
        for i, frame in tqdm(enumerate(reader.nextFrame())):
            frame_temp_list.append(frame) # RGB
            if args.resize:
                dsize_width = int (w * args.resize_ratio)
                dsize_height = int (h * args.resize_ratio)
                resized_frame = cv2.resize(frame, dsize=(dsize_width, dsize_height),interpolation=cv2.INTER_LINEAR)
            else:
                resized_frame = frame
            ####################### DETECT or TRACK #########################################################
            if i%args.shift_frame == 0: # DETECT
                face_box, lip_box = fa_DETECTION(resized_frame, fa, args.lip_detect_enlarge)
                if face_box == None: # detector fail
                    print("detection fail!") 
                    face_box = face_previous_box
                    lip_box = lip_previous_box
                else: # detector sucess
                    face_previous_box = face_box
                    lip_previous_box = lip_box
                # init tracker
                face_tracker = OPENCV_OBJECT_TRACKERS[args.opencv_tracker_type]()
                face_tracker.init(resized_frame, tuple(face_box))
                lip_tracker = OPENCV_OBJECT_TRACKERS[args.opencv_tracker_type]()
                lip_tracker.init(resized_frame, tuple(lip_box))

            else: # TRACK
                (success, box) = face_tracker.update(resized_frame)
                if success:
                    face_box = [int(x) for x in box]
                    if any(x<0 for x in face_box):
                        face_box = face_previous_box
                    else:
                        face_previous_box = face_box
                else:
                    print("track fail!")
                    print(face_previous_box)
                    face_box = face_previous_box

                (success, box) = lip_tracker.update(resized_frame)
                if success:
                    lip_box = [int(x) for x in box]
                    if any(x<0 for x in lip_box):
                        lip_box = lip_previous_box
                    else:
                        lip_previous_box = lip_box
                else:
                    print("track fail!")
                    print(lip_previous_box)
                    lip_box = lip_previous_box

            ## -- get 4 points
            face_enlarged_square_box, lip_enlarged_square_box = enlarge_box(face_box,args.face_enlarge,video_shape,args), enlarge_box(lip_box,args.lip_enlarge,video_shape,args)
            face_temp_bboxes.append(face_enlarged_square_box)
            lip_temp_bboxes.append(lip_enlarged_square_box)

            ####################### CORRECT & CROP #########################################################
            if i%args.shift_frame == 0:
                if i == 0:
                    pass
                else:
                    ## -- correct & check
                    face_corrected_boxes = get_corrected_boxes(face_temp_bboxes, args.shift_frame, video_shape)
                    face_checked_boxes = check_bbox_min_max(face_corrected_boxes,face_label_list)
                    face_label_list.extend(face_checked_boxes[:-1])
                    
                    lip_corrected_boxes = get_corrected_boxes(lip_temp_bboxes, args.shift_frame, video_shape)
                    lip_checked_boxes = check_bbox_min_max(lip_corrected_boxes,lip_label_list)
                    lip_label_list.extend(lip_checked_boxes[:-1])

                    ## -- crop
                    face_temp_cropped = crop_video(frame_temp_list,face_checked_boxes)
                    face_cropped_frame_list.extend(face_temp_cropped[:-1])
                    lip_temp_cropped = crop_video(frame_temp_list,lip_checked_boxes)
                    lip_cropped_frame_list.extend(lip_temp_cropped[:-1])
                    
                    ## -- reset temp list
                    frame_temp_list = [frame_temp_list[-1]]
                    face_temp_bboxes = [face_temp_bboxes[-1]]
                    lip_temp_bboxes = [lip_temp_bboxes[-1]]
            
            if i == 900:
                if args.short_test:
                    break
        ########################## REMAINIDER ##################################################################
        
        face_temp_cropped = crop_video(frame_temp_list, check_bbox_min_max(face_temp_bboxes,face_label_list))
        # pdb.set_trace()
        face_cropped_frame_list.extend(face_temp_cropped)
        face_label_list.extend(check_bbox_min_max(face_temp_bboxes,face_label_list))

        lip_temp_cropped = crop_video(frame_temp_list, check_bbox_min_max(lip_temp_bboxes,lip_label_list))
        lip_cropped_frame_list.extend(lip_temp_cropped)
        lip_label_list.extend(check_bbox_min_max(lip_temp_bboxes,lip_label_list))
        
        ## -- Save outpsuts
        if not args.short_test:
            assert(len(face_cropped_frame_list)==num_frames)
            assert(len(face_label_list)==num_frames)

        if args.save_mp4:
            save_mp4(vid_path=vid_path, save_dir = args.save_dir, cropped_video_frames=face_cropped_frame_list, type="face") 
            save_mp4(vid_path=vid_path, save_dir = args.save_dir, cropped_video_frames=lip_cropped_frame_list, type="lip") 
        if args.save_json:
            save_json(vid_path=vid_path, save_dir=args.save_dir, input_boxes=face_label_list, type="face")
            save_json(vid_path=vid_path, save_dir=args.save_dir, input_boxes=lip_label_list, type="lip")
        
        ## -- initialization for next vid
        face_previous_box = None
        lip_previous_box = None
        
        success_logger.info(f'====================== vid_path : {vid_path} ======================')
        success_logger.info(f'cost time : {time.time() - start_time:1.3f}')

    except Exception as e:
        error_logger.info(f'====================== vid_path : {vid_path} ======================')
        error_logger.exception(str(e))
        error_logger.info(f'{i}th frame'+"\n")



#################################################################################
##                                      MAIN
#################################################################################
def main(args):
    ## -- log
    save_dir_path = os.path.join(os.path.basename(__file__),Path(args.config).stem,datetime.datetime.now().isoformat().split('.')[0])
    error_logger = get_logger(save_dir_path, "error_log")
    success_logger = get_logger(save_dir_path, "success_log")

    ## -- load video path list
    if args.file_search:
        videos_list = recursive_file_search(args.root_dir, ".mp4")
    else:
        with open(args.data_list_txt_path,"r") as f:
            videos_list = f.readlines()
        videos_list = [x.strip() for x in videos_list]
    
    ## -- process
    if args.multiprocessing:
        pool_obj = multiprocessing.Pool(args.multi_process_num)
        func = partial(process,args,error_logger,success_logger)
        answer = pool_obj.map(func,videos_list)  
    else:
        for vid_path in tqdm(videos_list):
            process(args,error_logger,success_logger,vid_path)



if __name__ == '__main__':     
    ## -- parse YAML
    if args.config is not None:
        pdb.set_trace()
        with open(args.config, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            if k in args.__dict__:
                typ = find_option_type(k, parser)
                args.__dict__[k] = typ(v)
            else:
                sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))
    pdb.set_trace()
    main(args)