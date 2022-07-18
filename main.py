import argparse
from tqdm import tqdm
import pdb
import os
import statistics
import multiprocessing
from functools import partial

import cv2
import numpy as np 

from utils import recursive_file_search
from utils import save_mp4, save_json
from utils_vid import load_reader, load_detector
from utils_vid import enlarge_box, get_corrected_boxes, crop_video

###############################################################################
##                                      ARGS
###############################################################################
def load_args(default_config = None):
    parser = argparse.ArgumentParser(description='Face cropping')

    parser.add_argument('--root-dir', type = str, default="./sample", help='root directory of input files')  
    parser.add_argument('--save-dir', type = str, default="./out", help='root directory of output files')  

    parser.add_argument('--multiprocessing', type = bool, default=False, help='')  
    parser.add_argument('--multi-process-num', type = int, default=2, help='')  
    
    parser.add_argument('--gpu', type = bool, default=True, help='True -> use GPU for face detector')  
    parser.add_argument('--gpu-num', type = str, default='1', help='')  

    parser.add_argument('--save-mp4', type = bool, default=True, help='save cropped mp4 or not')  
    parser.add_argument('--save-json', type = bool, default=True, help='save json containing labelling points or not')  
    
    parser.add_argument('--error-txt-path', type = str, default="error_list.txt", help='')

    parser.add_argument('--shift-frame', type =int, default= 90, help='per each shift frame size, detection executed')  
    parser.add_argument('--lip-detect-enlarge', type =float, default=1,help='enlarge detection box before put into tracker') # LIP X 2 before tracking
    parser.add_argument('--lip-enlarge', type =float, default=0,help='enlarge proportion')
    parser.add_argument('--face-enlarge', type =float, default=1,help='enlarge proportion') # FACE X 2
    
    parser.add_argument('--short-test', type = bool, default=True,help='True -> 30sec test') 
    

    args = parser.parse_args()          
    return args

args = load_args()



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


def process(args,vid_path):
    
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
            ####################### DETECT or TRACK #########################################################
            if i%args.shift_frame == 0: # DETECT
                face_box, lip_box = fa_DETECTION(frame, fa, args.lip_detect_enlarge)
                if face_box == None: # detector fail
                    print("detection fail!") 
                    face_box = face_previous_box
                    lip_box = lip_previous_box
                else: # detector sucess
                    face_previous_box = face_box
                    lip_previous_box = lip_box
                    face_tracker = OPENCV_OBJECT_TRACKERS["medianflow"]()
                    face_tracker.init(frame, tuple(face_box))
                    lip_tracker = OPENCV_OBJECT_TRACKERS["medianflow"]()
                    lip_tracker.init(frame, tuple(lip_box))

            else: # TRACK
                (success, box) = face_tracker.update(frame)
                if success:
                    pdb.set_trace()
                    face_box = [int(x) for x in box]
                    face_previous_box = face_box
                else:
                    face_box = face_previous_box

                (success, box) = lip_tracker.update(frame)
                if success:
                    lip_box = [int(x) for x in box]
                    lip_previous_box = lip_box
                else:
                    lip_box = lip_previous_box

            ## -- get 4 points
            face_enlarged_square_box, lip_enlarged_square_box = enlarge_box(face_box,args.face_enlarge,video_shape), enlarge_box(lip_box,args.lip_enlarge,video_shape)
            face_temp_bboxes.append(face_enlarged_square_box)
            lip_temp_bboxes.append(lip_enlarged_square_box)

            ####################### CORRECT & CROP #########################################################
            if i%args.shift_frame == 0:
                if i == 0:
                    pass
                else:
                    ## -- correct
                    face_corrected_boxes = get_corrected_boxes(face_temp_bboxes, args.shift_frame, video_shape)
                    face_label_list.extend(face_corrected_boxes[:-1])
                    lip_corrected_boxes = get_corrected_boxes(lip_temp_bboxes, args.shift_frame, video_shape)
                    lip_label_list.extend(lip_corrected_boxes[:-1])

                    ## -- crop
                    face_temp_cropped = crop_video(frame_temp_list,face_corrected_boxes)
                    face_cropped_frame_list.extend(face_temp_cropped[:-1])
                    lip_temp_cropped = crop_video(frame_temp_list,lip_corrected_boxes)
                    lip_cropped_frame_list.extend(lip_temp_cropped[:-1])
                    
                    ## -- reset temp list
                    frame_temp_list = [frame_temp_list[-1]]
                    face_temp_bboxes = [face_temp_bboxes[-1]]
                    lip_temp_bboxes = [lip_temp_bboxes[-1]]
            
            if i == 900:
                if args.short_test:
                    break
        ########################## REMAINIDER ##################################################################
        face_temp_cropped = crop_video(frame_temp_list, face_temp_bboxes)
        face_cropped_frame_list.extend(face_temp_cropped)
        face_label_list.extend(face_temp_bboxes)

        lip_temp_cropped = crop_video(frame_temp_list, lip_temp_bboxes)
        lip_cropped_frame_list.extend(lip_temp_cropped)
        lip_label_list.extend(lip_temp_bboxes)
        
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

    except Exception as e: 
        error_path = args.error_txt_path    
        with open(error_path, "a") as f:
            f.write(vid_path + "\t" + str(e) + "\n")



#################################################################################
##                                      MAIN
#################################################################################
def main(args):

    videos_list = recursive_file_search(args.root_dir, ".mp4")
    
    if args.multiprocessing:
        pool_obj = multiprocessing.Pool(args.multi_process_num)
        func = partial(process,args)
        answer = pool_obj.map(func,videos_list)  
    else:
        for vid_path in tqdm(videos_list):
            process(args,vid_path)

if __name__ == '__main__':
    args = load_args()
    main(args)