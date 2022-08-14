import skvideo.io
import cv2
import face_alignment
import pdb
import numpy as np
import pdb

def load_reader(vid_path):
    reader = skvideo.io.FFmpegReader(vid_path)
    video_shape = reader.getShape()
    (num_frames, h, w, c) = video_shape   

    return reader, video_shape

def load_detector(device):
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device,flip_input=False, face_detector='sfd') # fa_probs_threshold  = 0.95 
    return fa

def enlarge_box(box, enlarge_ratio, video_shape, args):
        (num_frames, h, w, c) = video_shape
        (xmin, ymin, width, height) = [int(v) for v in box]
        
        ## -- restore pixel range : resized -> original 
        if args.resize:
            xmin,ymin,width,height = int(xmin/args.resize_ratio), int(ymin/args.resize_ratio), int(width/args.resize_ratio), int(height/args.resize_ratio)

        ## -- get center point
        center_y = ymin + int(height/2)
        center_x = xmin + int(width/2)

        ## -- square & enlarge
        one_side = max(width, height)
        half_one_side = int(one_side * (1 + enlarge_ratio) / 2)

        ## -- check one_side length 
        y_margin = min(abs(center_y), abs(int(h)-center_y))
        x_margin = min(abs(center_x), abs(int(w)-center_x))
        margin = min(y_margin, x_margin)
        if margin < half_one_side:
            half_one_side = margin

        ymin_boundary = int(center_y - half_one_side)
        ymax_boundary = int(center_y + half_one_side)
        xmin_boundary = int(center_x - half_one_side)
        xmax_boundary = int(center_x + half_one_side)

        if ymin_boundary  < 0:
            ymin_boundary = 0
        if xmin_boundary  < 0:
            xmin_boundary = 0
        if xmax_boundary > int(w)-1:
            xmax_boundary = int(w)-1
        if ymax_boundary > int(h)-1:
            ymax_boundary = int(h)-1

        
        enlarged_square_box = [ymin_boundary, xmin_boundary, ymax_boundary, xmax_boundary] # [y:y+h, x:x+w]
        return enlarged_square_box

def get_corrected_boxes(input_boxes,shift_frame_num,video_shape):
    #pdb.set_trace()
    corrected_boxes = []
    (num_frames, h, w, c) = video_shape

    rate_of_change = (np.array(input_boxes[-1]) - np.array(input_boxes[-2]))/shift_frame_num # last detection - last tracking 
    corrected_boxes = np.int64([np.array(input_boxes[j]) + rate_of_change*j for j in range(shift_frame_num)]) # linear interpolation
    corrected_boxes = corrected_boxes.tolist()
    
    ## -- some component become negative after correction
    positive_boxes = []
    for box in corrected_boxes:
        [left, top, right, bottom] = box
        if left  < 0: 
            left = 0
        if top  < 0: 
            top = 0
        if bottom > int(w)-1:             
            bottom = int(w)-1
        if right > int(h)-1: 
            right = int(h)-1
        positive_boxes.append([left, top, right, bottom]) # [ymin,xmin,ymax,xmax]
    #pdb.set_trace() 
    positive_boxes.append(input_boxes[-1])
    return positive_boxes

def check_bbox_min_max(corrected_boxes,label_list):
    minmax_checked_boxes = []
    for i,bbox in enumerate(corrected_boxes):
        [ymin,xmin,ymax,xmax] = bbox
        if (ymin < ymax) and (xmin < xmax):
            minmax_checked_boxes.append(bbox)
        else:
            if i == 0:
                minmax_checked_boxes.append(label_list[-1])
            else:
                minmax_checked_boxes.append(minmax_checked_boxes[i-1])
    return minmax_checked_boxes

def crop_video(frame_list, corrected_boxes):
    cropped_frame_list = []
    for i, frame in enumerate(frame_list):
        [left, top, right, bottom] = corrected_boxes[i]
        cropped_img = frame[int(left):int(right), int(top):int(bottom)]
        cropped_img = cv2.resize(cropped_img, (224,224), interpolation = cv2.INTER_LINEAR)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)       
        cropped_frame_list.append(cropped_img)
    return cropped_frame_list