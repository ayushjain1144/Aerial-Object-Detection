import os 
import sys

classes = {
    0:'pedestrian',
    1:'people',
    2:'bicycle',
    3:'car',
    4:'van',
    5:'truck',
    6:'tricycle',
    7:'awning-tricycle',
    8:'bus',
    9:'motor'
}


base_dir = '/media/ayushjain1144/New Linux/aerial-object-detection/Aerial-Object-Detection/aiskyeye_dataset'
train_base_dir = os.path.join(base_dir, 'VisDrone2019-DET-train')
val_base_dir = os.path.join(base_dir, 'VisDrone2019-DET-val')
train_images = os.path.join(train_base_dir, 'images')
val_images = os.path.join(val_base_dir, 'images')

train_annotations = os.path.join(train_base_dir, 'annotations')
val_annotations = os.path.join(val_base_dir, 'annotations')

train_annotations_list = sorted(os.listdir(train_annotations))
val_annotations_list = sorted(os.listdir(val_annotations))

#old_ann = open('annotation.txt', 'r')

#new_ann = open('new_ann.txt', 'w')

if not os.path.exists(os.path.join(train_base_dir, 'annotations_mod')):
    os.makedirs(os.path.join(train_base_dir, 'annotations_mod'))


if not os.path.exists(os.path.join(val_base_dir, 'annotations_mod')):
    os.makedirs(os.path.join(val_base_dir, 'annotations_mod'))


train_annotations_mod = os.path.join(train_base_dir, 'annotations_mod')
val_annotations_mod = os.path.join(val_base_dir, 'annotations_mod')

for file_name in train_annotations_list:

    old_ann = open(os.path.join(train_annotations, file_name), 'r')
    new_ann = open(os.path.join(train_annotations_mod, file_name), 'w')

    for line in old_ann.readlines():
        #print('annotations.txt,', end='')
        img_name_base = file_name[:-3]
        img_name = train_images + '/' + img_name_base +  'jpg'
        
        
        # some lines have , at end by mistake
        if line[-2] == ',':
            line = line[:-2]
            
        try:
            x1, y1, w, h,_,o,_,_ = line.split(sep=',')

            if w == '0'  or h == '0':
                continue
        
        except Exception as e:
            print(line)
        if o == '0' or o == '11':
            continue
        else:
            obj = int(o) - 1
        x2 = int(x1) + int(w)
        y2 = int(y1) + int(h)    

        class_name=classes[obj]
        
        #print(f'{img_name},{x1},{y1},{x2},{y2},{class_name}\n')
        #sys.exit()
        new_ann.write(f'{img_name},{x1},{y1},{x2},{y2},{class_name}\n')

    new_ann.close()
    old_ann.close()


for file_name in val_annotations_list:

    old_ann = open(os.path.join(val_annotations, file_name), 'r')
    new_ann = open(os.path.join(val_annotations_mod, file_name), 'w')

    for line in old_ann.readlines():
        #print('annotations.txt,', end='')
        img_name_base = file_name[:-3]
        img_name = val_images + '/' + img_name_base +  'jpg'
        
        x1, y1, w, h,_,o,_,_ = line.split(sep=',')

        if w == '0'  or h == '0':
                continue

        if o == '0' or o == '11':
            continue
        else:
            obj = int(o) - 1
        x2 = int(x1) + int(w)
        y2 = int(y1) + int(h)    

        class_name=classes[obj]
        
        #print(f'{img_name},{x1},{y1},{x2},{y2},{class_name}\n')
        #sys.exit()
        new_ann.write(f'{img_name},{x1},{y1},{x2},{y2},{class_name}\n')

    new_ann.close()
    old_ann.close()