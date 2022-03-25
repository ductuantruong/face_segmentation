import albumentations as A
from os import listdir
import cv2 
from PIL import Image

train_data_dir = 'data/CelebAMaskHQ/train/train_image/'
train_label_dir = 'data/CelebAMaskHQ/train/train_mask/'
val_data_dir = 'data/CelebAMaskHQ/val/val_image/'
val_label_dir = 'data/CelebAMaskHQ/val/val_mask/'


def save_data_aug(data_dir, label_dir):
    lst_file = listdir(data_dir)
    for i, image in enumerate(lst_file):
        n_total_file = len(listdir(data_dir))
        origin_img = cv2.imread(data_dir + image)
        label_img = Image.open(label_dir + image[:-3] + 'png')
        print(label_dir + image)

        bright_origin_img = A.RandomBrightnessContrast(p=1)(image=origin_img)['image']
        cv2.imwrite(data_dir + str(n_total_file) + '.jpg', bright_origin_img)
        label_img.save(label_dir + str(n_total_file) + '.png')        
        
        clahe_origin_img = A.CLAHE(p=1)(image=origin_img)['image']
        cv2.imwrite(data_dir + str(n_total_file + 1) + '.jpg', clahe_origin_img)
        label_img.save(label_dir + str(n_total_file + 1) + '.png')   
        
        blur_origin_img = A.Blur(p=1)(image=origin_img)['image']
        cv2.imwrite(data_dir + str(n_total_file + 2) + '.jpg', blur_origin_img)
        label_img.save(label_dir + str(n_total_file + 2) + '.png') 

save_data_aug(train_data_dir, train_label_dir)
save_data_aug(val_data_dir, val_label_dir)
