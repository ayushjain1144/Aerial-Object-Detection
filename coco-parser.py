import json
import os
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-j", "--coco_json_dir", required=True, help='path to coco json file')
ap.add_argument("-d", "--ann_dir", required=True, help='directory to store converted predictions')
ap.add_argument("-g", "--grnd_dir", required=True, help='directory containing input images')

args = vars(ap.parse_args())

f = open(args['coco_json_dir'])

data = json.load(f)

data.sort(key=lambda x: x['image_id'])

os.makedirs(args['ann_dir'], exist_ok = True)

pred_list = []
grnd_list = []

for img in sorted(os.listdir(args['grnd_dir'])):
    grnd_list.append(img[:-4])

i=0
while(i<len(data)):
    img_id = data[i]['image_id']
    pred_list.append(img_id)
    with open(os.path.join(args['ann_dir'], img_id + '.txt'), "a") as fn:
        while(i<len(data) and data[i]['image_id']==img_id):
            d = data[i]
            fn.write(','.join([str(d['bbox'][0]), str(d['bbox'][1]), str(d['bbox'][2]), str(d['bbox'][3]), str(d['score']), str(d['category_id'] + 1), '-1', '-1']))
            fn.write('\n')
            i += 1

for img_id in set(grnd_list)-set(pred_list):  # add empty files for images with zero predictions
    with open(os.path.join(args['ann_dir'], img_id + '.txt'), "w") as fn:
        pass

print('number of images without any predictions is {}'.format(len(set(grnd_list)-set(pred_list))))