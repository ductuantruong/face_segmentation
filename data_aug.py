import albumentations as A
from os import listdir
import cv2 

train_data_dir = 'data/CelebAMaskHQ/train/train_image/'
train_label_dir = 'data/CelebAMaskHQ/train/train_mask/'
val_data_dir = 'data/CelebAMaskHQ/val/val_image/'
val_label_dir = 'data/CelebAMaskHQ/val/val_mask/'


def save_data_aug(data_dir, label_dir):
    for i, image in enumerate(listdir(data_dir)):
        n_total_file = len(listdir(data_dir))
        origin_img = cv2.imread(data_dir + image)
        # origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        label_img = cv2.imread(label_dir + image[:-3] + 'png')
        print(label_dir + image)
        # label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)

        hflip_origin_img = A.HorizontalFlip(p=1)(image=origin_img)['image']
        hflip_label_img = A.HorizontalFlip(p=1)(image=label_img)['image']
        cv2.imwrite(data_dir + str(n_total_file) + '.jpg', hflip_origin_img)
        cv2.imwrite(label_dir + str(n_total_file) + '.png', hflip_label_img)
        
        vflip_origin_img = A.VerticalFlip(p=1)(image=origin_img)['image']
        vflip_label_img = A.VerticalFlip(p=1)(image=label_img)['image']
        cv2.imwrite(data_dir + str(n_total_file + 1) + '.jpg', vflip_origin_img)
        cv2.imwrite(label_dir + str(n_total_file + 1) + '.png', vflip_label_img)

        bright_origin_img = A.RandomBrightnessContrast(p=1)(image=origin_img)['image']
        cv2.imwrite(data_dir + str(n_total_file + 2) + '.jpg', bright_origin_img)
        cv2.imwrite(label_dir + str(n_total_file + 2) + '.png', label_img)        
        
        clahe_origin_img = A.CLAHE(p=1)(image=origin_img)['image']
        cv2.imwrite(data_dir + str(n_total_file + 3) + '.jpg', clahe_origin_img)
        cv2.imwrite(label_dir + str(n_total_file + 3) + '.png', label_img)   
        
        blur_origin_img = A.Blur(p=1)(image=origin_img)['image']
        cv2.imwrite(data_dir + str(n_total_file + 4) + '.jpg', blur_origin_img)
        cv2.imwrite(label_dir + str(n_total_file + 4) + '.png', label_img) 

save_data_aug(train_data_dir, train_label_dir)
save_data_aug(val_data_dir, val_label_dir)