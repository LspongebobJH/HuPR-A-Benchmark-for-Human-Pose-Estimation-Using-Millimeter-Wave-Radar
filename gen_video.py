# generate videos for images in visualization/

import cv2
import os
from tqdm import tqdm
List = [16, 38, 40, 41, 42,
               17, 39,  244, 245, 246, 249, 250, 251, 252, 253, 254,
               247, 248, 255, 256]
img_path = "/home/ubuntu/hupr/visualization/single_{idx}/"
vid_path = "/home/ubuntu/hupr/video/{idx}.avi"
for idx in tqdm(List):
    image_folder = img_path.format(idx=idx)
    video_name = vid_path.format(idx=idx)

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

