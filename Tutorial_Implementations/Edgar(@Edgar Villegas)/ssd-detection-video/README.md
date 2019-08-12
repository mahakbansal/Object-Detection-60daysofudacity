# ssd-detection-video

Pyimagesearch's tutorial is great, but the example
can be a little confusing to start with.  
So, I created this version to focus on being easier to follow by beginners. 
(And also for people interested in SSD).   

Differenes with original tutorial:

- Uses SSD model instead of YOLO
- Simpler code, ideal for beginners
- Has live preview instead of creating output video file 

## Usage

```sh
python ssd-video.py --input videos/dog-kid.mp4
``` 


### Sidenote

SSD implementation runs faster than Yolo, so it's feasible to show a 'live preview' (depending on your processor's speed).
However, I noticed it's less accurate.
  