# Face_Lip_detection_tracking
립리딩 영상 face / lip ROI labelling 작업


## 1. Example

<img src="https://user-images.githubusercontent.com/77431192/179312614-04b450a5-ab56-4310-bed6-d2650aba0dae.gif" width="150" height="150"/> <img src="https://user-images.githubusercontent.com/77431192/179313748-81a83727-a739-4753-964f-8c4701dfd210.gif" width="150" height="150"/>

## 2. How to run
* locate videos to be processed at `./sample` (args.root_dir)
* results will be saved at `./out` (args.save_dir)
* If error occurs, file_name and error message will be written at `error_list.txt` (args.error_txt_path). 
~~~
python main.py
~~~

## 3. Output directory
* main.py detect face ROI and lip ROI at the same time.  
* `./out` directory tree is as follows. 

```bash
├── lip_mp4
   ├── sample1.mp4
   ├── sample2.mp4
   └── sample3.mp4
             '
             '  
             '
├── lip_json
   ├── sample1.json
   ├── sample2.json
   └── sample3.json
             '
             '  
             '
├── face_mp4
├── face_json

├── train.csv
``` 

## 4. Details
[Face_Lip_detection_tracking_guide](https://pollen-cardboard-eef.notion.site/Face_Lip_detection_tracking_guide-c4a59f4e3f1246b5b5c934942e7ccd42)
