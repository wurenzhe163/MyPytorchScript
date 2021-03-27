from __future__ import print_function, division
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.utils.data
import torch
from torchstat import stat
from torchsummary import summary
import torchgeometry as tgs
from LrSchduler import PolyLR
import DataLoader

class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)   #64  256

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)   #128 128

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)  # [-1, 256, 64, 64]

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)  # [-1, 512, 32, 32]

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)  # [-1, 1024, 16, 16]

        d5 = self.Up5(e5)   # [-1, 512, 32, 32]
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)  # [-1, 512, 32, 32]

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out
class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

def DEVICE_SLECT():
    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    print(DEVICE)
    return DEVICE

model = U_Net(in_ch=3, out_ch=1)
model(torch.rand(1,3,256,256))
DEVICE=DEVICE_SLECT()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
Scheduler = PolyLR(optimizer,max_iters=500)
#损失函数也需要放到GPU中
criterion = tgs.losses.TverskyLoss(0.3,0.7).to(DEVICE)
#另外一个超参数，指定训练批次
TOTAL_EPOCHS=500
train_loader,valid_loader,train_dataset,valid_dataset = DataLoader.main()
BATCH_SIZE = 16

#记录损失函数
losses = [];
%%time
for epoch in range(TOTAL_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(DEVICE)
        labels = labels.to(DEVICE)
        #清零
        optimizer.zero_grad()
        outputs = model(images)
        #计算损失函数
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item());
        Scheduler.step()
        if (i+1) % 1 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE, loss.data.item()))


for epoch in TOTAL_EPOCHS:


    loss = criterion()
    # 向后传播
    optimizer.zero_grad()  # 注意每次迭代都需要清零
    loss.backward()

    optimizer.step()

    Scheduler.step()

    if (epoch + 1) % 1 == 0:
        print('Epoch[{}/{}], loss:{:.6f}, lr:{}'.format(epoch + 1, num_epochs, loss.item(),p['lr']))





stat(model,(3,256,256))
summary(model,(3,256,256))
model(torch.rand(1,3,28,28))






writer = SummaryWriter('logs')
image_path=r'C:\Users\SAR\Desktop\my_python_script\pytorch\images\图片3.png'
img = Image.open(image_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image('标题名',tensor_img,1,dataformats='CHW')






trans_norm = transforms.Normalize([],[])
writer.close()


