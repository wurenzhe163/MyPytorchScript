import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import normalize
import matplotlib.pyplot as plt


class Denormalize(object):
    '''
    return : 反标准化
    '''
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def visualize(**images):
    """
    plt展示图像
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def OneHotEncode(LabelImage,NumClass):
    '''
    Onehot Encoder  2021/03/23 by Mr.w
    -------------------------------------------------------------
    LabelImage ： Ndarry   |   NumClass ： Int
    -------------------------------------------------------------
    return： Ndarry
    '''
    one_hot_codes = np.eye(NumClass)
    try:
        one_hot_label = one_hot_codes[LabelImage]
    except IndexError:
        print('标签存在不连续值，最大值为{}--->已默认将该值进行连续化,仅适用于二分类'.format(np.max(LabelImage)))
        LabelImage[LabelImage == np.max(LabelImage)] = 1
        one_hot_label = one_hot_codes[LabelImage]
    return one_hot_label
def OneHotDecode(OneHotImage):
    '''
    OneHotDecode 2021/03/23 by Mr.w
    -------------------------------------------------------------
    OneHotImage : ndarray -->(512,512,x)
    -------------------------------------------------------------
    return : image --> (512,512)
    '''
    return np.argmax(OneHotImage,axis=-1)

def data_augmentation(ToTensor=False,Resize=None,Contrast=None,Equalize=None,HFlip=None,Invert=None,VFlip=None,
                      Rotation=None,Grayscale=None,Perspective=None,Erasing=None,Crop=None):
    '''
    DataAgumentation 2021/03/23 by Mr.w
    -------------------------------------------------------------
    ToTensor : False/True , 注意转为Tensor，通道会放在第一维
    Resize : tuple-->(500,500)
    Contrast : 0-1 -->图像被自动对比度的可能
    Equalize : 0-1 -->图像均衡可能性
    HFlip : 0-1 --> 图像水平翻转
    Invert : 0-1--> 随机翻转
    VFlip : 0-1 --> 图像垂直翻转
    Rotation : 0-360 --> 随机旋转度数范围, as : 90 , [-90,90]
    Grayscale : 0-1 --> 随机转换为灰度图像
    Perspective : 0-1 --> 随机扭曲图像
    Erasing : 0-1 --> 随机擦除
    Crop : tuple --> (500,500)
    -------------------------------------------------------------
    return : transforms.Compose(train_transform) --> 方法汇总
    '''
    #列表导入Compose
    train_transform = []
    if ToTensor == True:
        trans_totensor = transforms.ToTensor()
        train_transform.append(trans_totensor)

    if Resize != None:
        trans_Rsize = transforms.Resize(Resize)  # Resize=(500,500)
        train_transform.append(trans_Rsize)
    if Contrast != None:
        trans_Rcontrast = transforms.RandomAutocontrast(p=Contrast)
        train_transform.append(trans_Rcontrast)
    if Equalize != None:
        trans_REqualize = transforms.RandomEqualize(p=Equalize)
        train_transform.append(trans_REqualize)
    if HFlip != None:
        train_transform.append(transforms.RandomHorizontalFlip(p=HFlip))
    if Invert != None:
        train_transform.append(transforms.RandomInvert(p=Invert))
    if VFlip != None:
        train_transform.append(transforms.RandomVerticalFlip(p=VFlip))
    if Rotation != None:
        train_transform.append(transforms.RandomRotation(Rotation,expand=False,center=None,fill=0,resample=None))
    if Grayscale != None:
        train_transform.append(transforms.RandomGrayscale(p=Grayscale))
    if Perspective != None:
        train_transform.append(transforms.RandomPerspective(distortion_scale=0.5,p=Perspective,fill=0))
    if Erasing != None:
        train_transform.append(transforms.RandomErasing(p=Erasing,scale=(0.02, 0.33),ratio=(0.3, 3.3),value=0,inplace=False))
    if Crop != None:
        train_transform.append(transforms.RandomCrop(Crop,padding=None,pad_if_needed=False,fill=0,padding_mode='constant'))
    return transforms.Compose(train_transform)
