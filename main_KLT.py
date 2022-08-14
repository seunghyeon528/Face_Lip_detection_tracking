import argparse
from tqdm import tqdm
import pdb
import os
import statistics
import multiprocessing
from functools import partial
import matplotlib
import cv2
import numpy as np 

from utils import recursive_file_search
from utils import save_mp4, save_json, get_logger
from utils_vid import load_reader, load_detector
from utils_vid import enlarge_box, get_corrected_boxes, crop_video

from KLT.getFeatures import getFeatures
from KLT.estimateAllTranslation import estimateAllTranslation
from KLT.applyGeometricTransformation import applyGeometricTransformation

###############################################################################
##                                      ARGS
###############################################################################
def load_args(default_config = None):
    parser = argparse.ArgumentParser(description='Face cropping')

    ## -- IO
    parser.add_argument('--file-search', type = bool, default=True,help='recursive file search') # True -> use videos at root-dir, False -> use data list txt path
    parser.add_argument('--root-dir', type = str, default="./input/MOBIO", help='root directory of input files')  
    parser.add_argument('--data-list-txt-path', type = str, default="./data/uhd_list.txt",help='recursive file search') 
    parser.add_argument('--save-dir', type = str, default="./bbox_rework_out/MOBIO/KLT/", help='root directory of output files')  

    parser.add_argument('--save-mp4', type = bool, default=True, help='save cropped mp4 or not')  
    parser.add_argument('--save-json', type = bool, default=False, help='save json containing labelling points or not')  

    parser.add_argument('--multiprocessing', type = bool, default=False, help='')  
    parser.add_argument('--multi-process-num', type = int, default=2, help='')  
    
    parser.add_argument('--gpu', type = bool, default=True, help='True -> use GPU for face detector')  
    parser.add_argument('--gpu-num', type = str, default='0', help='')  

    parser.add_argument('--shift-frame', type =int, default= 90, help='per each shift frame size, detection executed')  
    
    parser.add_argument('--lip-detect-enlarge', type =float, default=1,help='enlarge detection box before put into tracker') # LIP X 2 before tracking
    parser.add_argument('--lip-enlarge', type =float, default=0,help='enlarge proportion')
    parser.add_argument('--face-enlarge', type =float, default=1,help='enlarge proportion') # FACE X 2
    
    parser.add_argument('--short-test', type = bool, default=False,help='True -> 30sec test') 
    
    parser.add_argument('--resize', type =bool, default=False,help='resize before put into detector / tracker') # UHD -> HD  / Detection fail 
    parser.add_argument('--resize-ratio', type =int, default=0.5,help='resize proportion') # UHD -> HD  / Detection fail 

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


def process(args,logger, vid_path):
    
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

        ## --  variables
        frame_temp_list = []
        lip_temp_bboxes = []
        face_temp_bboxes = []
        
        face_label_list = []
        lip_label_list = []
        
        face_cropped_frame_list = []
        lip_cropped_frame_list = []
        
        resized_frame_temp_list = []

        ## -- Walk every frame
        for i, frame in tqdm(enumerate(reader.nextFrame())):
            frame_temp_list.append(frame) # RGB
            if args.resize:
                dsize_width = int (w * args.resize_ratio)
                dsize_height = int (h * args.resize_ratio)
                resized_frame = cv2.resize(frame, dsize=(dsize_width, dsize_height),interpolation=cv2.INTER_LINEAR)
            else:
                resized_frame = frame
            resized_frame_temp_list.append(resized_frame)

            ####################### DETECT or TRACK #########################################################
            if i%args.shift_frame == 0: # DETECT
                face_box, lip_box = fa_DETECTION(resized_frame, fa, args.lip_detect_enlarge)
                if face_box == None: # detector fail
                    print("detection fail!") 
                    face_box = face_previous_box
                    lip_box = lip_previous_box
                else: # detector sucess
                    #pdb.set_trace()
                    face_previous_box = face_box
                    lip_previous_box = lip_box
                    
                ## -- init KLT tracker
                [xmin,ymin,boxw,boxh] = face_box
                KLT_bbox = np.array([[[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]]).astype(float)
                startXs,startYs = getFeatures(cv2.cvtColor(resized_frame_temp_list[0],cv2.COLOR_RGB2GRAY),KLT_bbox,use_shi=False)

            else: # TRACK
                #pdb.set_trace()
                newXs, newYs = estimateAllTranslation(startXs, startYs, resized_frame_temp_list[-2], resized_frame_temp_list[-1])
                Xs, Ys ,KLT_bbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, KLT_bbox) # KLT bbox update

                # update coordinates
                startXs = Xs
                startYs = Ys

                try:
                    # update feature points as required
                    n_features_left = np.sum(Xs!=-1)
                    # print('# of Features: %d'%n_features_left)
                    if n_features_left < 15:
                        print('Generate New Features')
                        startXs,startYs = getFeatures(cv2.cvtColor(resized_frame_temp_list[-1],cv2.COLOR_RGB2GRAY),KLT_bbox)
                except:
                    print("Generate New feature error")
                    pass

                # get [xmin,ymin,boxw,boxh] format
                face_box = cv2.boundingRect(KLT_bbox[0,:,:].astype(int)) # [xmin,ymin,boxw,boxh]
                face_previous_box = face_box
            ## -- get 4 points
            face_enlarged_square_box = enlarge_box(face_box,args.face_enlarge,video_shape,args)
            face_temp_bboxes.append(face_enlarged_square_box)

            ####################### CORRECT & CROP #########################################################
            if i%args.shift_frame == 0:
                if i == 0:
                    pass
                else:
                    ## -- correct
                    face_corrected_boxes = get_corrected_boxes(face_temp_bboxes, args.shift_frame, video_shape)
                    face_label_list.extend(face_corrected_boxes[:-1])

                    ## -- crop
                    face_temp_cropped = crop_video(frame_temp_list,face_corrected_boxes)
                    face_cropped_frame_list.extend(face_temp_cropped[:-1])
                    
                    ## -- reset temp list
                    frame_temp_list = [frame_temp_list[-1]]
                    face_temp_bboxes = [face_temp_bboxes[-1]]
                    resized_frame_temp_list = [resized_frame_temp_list[-1]]
            if i == 900:
                if args.short_test:
                    break
        ########################## REMAINIDER ##################################################################
        face_temp_cropped = crop_video(frame_temp_list, face_temp_bboxes)
        # pdb.set_trace()
        face_cropped_frame_list.extend(face_temp_cropped)
        face_label_list.extend(face_temp_bboxes)
        
        ## -- Save outpsuts
        if not args.short_test:
            assert(len(face_cropped_frame_list)==num_frames)
            assert(len(face_label_list)==num_frames)

        if args.save_mp4:
            save_mp4(vid_path=vid_path, save_dir = args.save_dir, cropped_video_frames=face_cropped_frame_list, type="face") 
        if args.save_json:
            save_json(vid_path=vid_path, save_dir=args.save_dir, input_boxes=face_label_list, type="face")
        
        ## -- initialization for next vid
        face_previous_box = None
        lip_previous_box = None

    except Exception as e:
        logger.info(f'====================== vid_path : {vid_path} ======================')
        logger.exception(str(e))
        logger.info(f'{i}th frame'+"\n")



#################################################################################
##                                      MAIN
#################################################################################

def main(args):

    ## -- log
    save_dir_path = os.path.basename(__file__)
    logger = get_logger(save_dir_path)

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
        func = partial(process,args,logger)
        answer = pool_obj.map(func,videos_list)  
    else:
        for vid_path in tqdm(videos_list):
            process(args,logger,vid_path)

if __name__ == '__main__':
    args = load_args()
    main(args)