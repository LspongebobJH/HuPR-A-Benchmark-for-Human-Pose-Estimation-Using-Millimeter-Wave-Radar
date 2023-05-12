import sys
sys.path.append('/home/ubuntu/hupr')


from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

image_folder = "/home/ubuntu/hupr/visualization/single_{idx}/"
img_path = "/home/ubuntu/hupr/visualization/single_{which}/{frame:09d}.png"
img_gt_path = "/home/ubuntu/hupr/visualization/single_{which}/{frame:09d}_gt.png"
vid_path = "/home/ubuntu/hupr/video/2/{idx}_pred_gt.avi"
# 获取当前文件夹中所有JPG图像
List = [15, 16, 38, 40, 41, 42,
               17, 39,  244, 245, 246, 249, 250, 251, 252, 253, 254,
               247, 248, 255, 256]
# List = [15]
frame_width, frame_height = 512, 256
for idx in tqdm(List):
    # im_list = [Image.open(fn) for fn in listdir() if fn.endswith('.jpg')]
    images = [join(image_folder.format(idx=idx), img) for img in listdir(image_folder.format(idx=idx)) if '_gt' not in img]
    images_gt = [join(image_folder.format(idx=idx), img) for img in listdir(image_folder.format(idx=idx)) if '_gt' in img]
    images.sort()
    images_gt.sort()

    video_name = vid_path.format(idx=idx)
    video = cv2.VideoWriter(video_name, 0, 1, (frame_width, frame_height))

    for im, imgt in zip(images, images_gt):
        im, imgt = Image.open(im), Image.open(imgt)
        # 单幅图像尺寸
        width, height = im.size

        # 创建空白长图
        result = Image.new(im.mode, (width * 2, height))
        result.paste(im, box=(0, 0))
        result.paste(imgt, box=(width, 0))

        # PIL to OpenCV
        result = np.array(result)[:, :, ::-1]
        video.write(result)

cv2.destroyAllWindows()
video.release()





