from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
#import config
from keras_retinanet import models
from imutils import paths
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image  
import PIL  
import matplotlib.image as mpimg
import os
import argparse

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True, help='path to trained model')
ap.add_argument("-l", "--labels", required=True, help='path to class labels')
ap.add_argument("-i", "--input", required=True, help='path to input images directory')
ap.add_argument("-o", "--output", required=True, help='path to directory to store predictions')
ap.add_argument("-c", "--confidence", type=float, default=0.0, help="min probability to filter weak detections")
ap.add_argument("-b", "--backbone", required=True, help='backbone')

args = vars(ap.parse_args())

LABELS = open(args["labels"]).read().strip().split('\n')
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

model = models.load_model(args["model"], backbone_name=args['backbone'])
imagePaths = list(paths.list_images(args["input"]))

for (i, imagePath) in enumerate(imagePaths):

    print("[INFO] predicting on image {} of {}".format(i+1, imagePath))
    filename = (imagePath.split(os.path.sep)[-1]).split('.')[0]
    output_file = os.path.sep.join([args["output"], '{}.txt'.format(filename)])
    file = open(output_file, 'w')
    
    image = read_image_bgr(imagePath)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    (boxes, scores, labels) = model.predict_on_batch(image)
    boxes /= scale
    i = 0
    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):

        #i = i + 1`
        if score < args["confidence"]:
            continue
        
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(LABELS[label], score)
        draw_caption(draw, b, caption)

        #box = box.astype("int")
        width = box[2] - box[0]
        height = box[3] - box[1]
        class_label = label + 1
        row = ",".join([str(box[0]), str(box[1]), str(width), str(height), str(score), str(class_label), '-1', '-1'])
        
        file.write("{}\n".format(row))
        
        plt.savefig("result_images/{}/{}".format(filename, i))
        plt.show()

    file.close()
