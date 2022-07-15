# Face_Lip_detection_tracking
립리딩 영상 face / lip ROI labelling 작업
referring to directory https://github.com/jungwook518/Face_Lip_detection_tracking

## 1. Example

<img src="https://user-images.githubusercontent.com/77431192/179312614-04b450a5-ab56-4310-bed6-d2650aba0dae.gif" width="150" height="150"/> <img src="https://user-images.githubusercontent.com/77431192/179313748-81a83727-a739-4753-964f-8c4701dfd210.gif" width="150" height="150"/>

## 2. How to run
* locate videos to be processed at `./sample` (args.root_dir)
* results will be saved at `./out` (args.save_dir)
* if error occurs, file_name will be written at `error_list.txt` (args.error_txt_path), I bet there will be few errors!😊
~~~
python main.py
~~~

