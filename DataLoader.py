import os,cv2,torch
import numpy as np
import utils

search_files = lambda path,endwith='.tif': [os.path.join(path,f) for f in os.listdir(path) if f.endswith(endwith) ]

def _get_dir(DATA_DIR=r'D:\deep_road\tiff'):
    '''非复用'''
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'train_labels')
    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'val_labels')
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'test_labels')
    return x_train_dir,y_train_dir,x_valid_dir,y_valid_dir,x_test_dir,y_test_dir

class RoadsDataset(torch.utils.data.Dataset):


    def __init__(
            self,
            images_dir,
            masks_dir,
            Numclass = 2,  #分类数
            augmentation=None,

    ):
        self.image_paths = search_files(images_dir,endwith='.tiff')
        self.mask_paths = search_files(masks_dir,endwith='.tif')
        self.Numclass = Numclass
        self.augmentation = augmentation

    def __getitem__(self, i):

        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[i],cv2.IMREAD_GRAYSCALE)

        # one-hot-encode the mask
        mask1 = utils.OneHotEncode(mask, self.Numclass)

        # apply augmentations
        if self.augmentation:
            ImageMask = np.concatenate([image, mask1], axis=2)  #图像与Lable一同变换
            sample = self.augmentation(ImageMask)
            image2, mask2 = sample[0:image.shape[2],:,:], sample[image.shape[2]:,:,:]
            # 注意,经过augmentation,数据dtype=float64,需要转换数据类型才能够正常显示
        self.image = image
        self.image2 = image2
        self.mask2 = mask2
        return image2, mask2
    def visu(self):
        utils.visualize(Befor_Argu=self.image,
                        After_Argu=self.image2.permute(1, 2, 0).numpy().astype(np.uint8),
                        Label=utils.OneHotDecode(self.mask2.permute(1, 2, 0).numpy().astype(np.uint8)))

    def __len__(self):
        # return length of
        return len(self.image_paths)

def main():

    data_augmentation = utils.data_augmentation(ToTensor=True, HFlip=0.5, VFlip=0.5)
    x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, x_test_dir, y_test_dir = _get_dir()
    # Get train and val dataset instances
    train_dataset = RoadsDataset(
        x_train_dir, y_train_dir,Numclass = 2,
        augmentation=data_augmentation,
    )
    valid_dataset = RoadsDataset(
        x_valid_dir, y_valid_dir,Numclass = 2,
        augmentation=data_augmentation,
    )
    # 展示一个看看
    image,mask = train_dataset[0]
    train_dataset.visu()



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    return train_loader,valid_loader,train_dataset,valid_dataset
# if __name__ == '__main__':


