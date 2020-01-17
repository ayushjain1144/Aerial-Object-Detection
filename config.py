import os

base_dir = './aiskyeye_dataset'
train_base_dir = os.path.join(base_dir, 'VisDrone2019-DET-train')
val_base_dir = os.path.join(base_dir, 'VisDrone2019-DET-val')
val_test_challenge_dir = os.path.join(base_dir, 'VisDrone2019-DET-test-challenge')
val_test_dev_dir = os.path.join(base_dir, 'Visdrone2019-DET-test-dev')

train_images = os.path.join(train_base_dir, 'images')
val_images = os.path.join(val_base_dir, 'images')
val_test_challenge_images = os.path.join(val_test_challenge_dir, 'images')
val_test_dev_images = os.path.join(val_test_dev_dir, 'images')

train_annotations = os.path.join(train_base_dir, 'annotations')
val_annotations = os.path.join(val_base_dir, 'annotations')
val_test_challenge_annotations = os.path.join(val_test_challenge_dir, 'annotations')
val_test_dev_annotations = os.path.join(val_test_dev_dir, 'annotations')
