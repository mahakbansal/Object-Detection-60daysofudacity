This is the code suggested by the YOLO object detection tutorial on https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/,
the suggested source we are following for the object detection project on #sg_wonder_vision for the Facebook Secure and Private AI Challenge.
For further technical description, please visit link. 
The above code has worked well when implemented by me. 
(yolo-images does object detection for images and yolo-videos is for videos)
In terminal , execute below commands in terminal: 
(for images)
$ python yolo.py --image images/dining_table.jpg --yolo yolo-coco  #use image file name you are using 
[INFO] loading YOLO from disk...
[INFO] YOLO took 0.362369 seconds
(for videos)
$ python yolo_video.py --input videos/car_chase_02.mp4 \   #use video file you are using 
	--output output/car_chase_02.avi --yolo yolo-coco
[INFO] loading YOLO from disk...
[INFO] 3132 total frames in video
[INFO] single frame took 0.3455 seconds
[INFO] estimated total time to finish: 1082.0806
[INFO] cleaning up...
