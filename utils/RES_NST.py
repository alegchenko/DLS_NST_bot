from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import copy

from utils.NST import NST
from utils.my_layers import ContentLoss, StyleLoss, Normalization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Наследуемый класс дающий возможность использовать CNN типа ResNet 50/101/152
class Res_NST(NST):
      def __init__(self, cnn):
        super(Res_NST, self).__init__()
        self.imsize = 512
        self.epochs = 700
        self.cnn = cnn

      def get_model_and_losses(self, content_img, style_img):
          cnn = self.cnn 
          model = self.SubModel(cnn, content_img, style_img)
          style_losses = model.style_losses
          content_losses = model.content_losses
          return model, style_losses, content_losses
          
          
#Вспомогательный класс формирующий частичную модель от ResNet
class SubModel(nn.Module):
        def __init__(self, cnn, content_img, style_img):
            super().__init__()

            with torch.no_grad():
              content_img = content_img.clamp_(0, 1)
              style_img = style_img.clamp_(0, 1)

            cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
            cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

            self.normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
            self.cont = content_img
            self.style = style_img
            content_losses = []
            style_losses = []
            self.conv1 = cnn.conv1
            self.bn1 = cnn.bn1
            self.relu = nn.ReLU(inplace=False)
            self.maxpool = cnn.maxpool
            self.conv2 = cnn.layer1[0].conv1
            self.bn2 = cnn.layer1[0].bn1
            self.conv3 = cnn.layer1[0].conv2
            self.bn3 = cnn.layer1[0].bn2
            self.conv4 = cnn.layer1[0].conv3
            self.bn4 = cnn.layer1[0].bn3
            self.d_conv = cnn.layer1[0].downsample[0]
            self.d_bn = cnn.layer1[0].downsample[1]
            self.conv5 = cnn.layer1[1].conv1
            self.bn5 = cnn.layer1[1].bn1
            self.conv6 = cnn.layer1[1].conv2
            self.bn6 = cnn.layer1[1].bn2
            self.conv7 = cnn.layer1[1].conv3
            self.bn7 = cnn.layer1[1].bn3
            self.conv8 = cnn.layer1[2].conv1
            self.bn8 = cnn.layer1[2].bn1
            self.conv9 = cnn.layer1[2].conv2
            self.bn9 = cnn.layer1[2].bn2
            self.conv10 = cnn.layer1[2].conv3
            self.bn10 = cnn.layer1[2].bn3
            self.conv11 = cnn.layer2[0].conv1
            self.bn11 = cnn.layer2[0].bn1
            self.conv12 = cnn.layer2[0].conv2
            self.bn12 = cnn.layer2[0].bn2
            self.conv13 = cnn.layer2[0].conv3
            
            #Расчет значений выходов слоев для изображения - эталона стиля
            x = self.normalization(style_img)
            x = self.conv1(x)

            target_feature = x.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss1 = style_loss
            style_losses.append(self.style_loss1)

            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x1 = self.conv2(x)

            target_feature = x1.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss2 = style_loss
            style_losses.append(self.style_loss2)

            x1 = self.bn2(x1)
            x1 = self.relu(x1)
            x1 = self.conv3(x1)

            target_feature = x1.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss3 = style_loss
            style_losses.append(self.style_loss3)

            x1 = self.bn3(x1)
            x1 = self.relu(x1)
            x1 = self.conv4(x1)

            target_feature = x1.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss4 = style_loss
            style_losses.append(self.style_loss4)

            x1 = self.bn4(x1)

            x = self.d_conv(x)

            target_feature = x.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss_d = style_loss
            style_losses.append(self.style_loss_d)

            x = self.d_bn(x)

            x = x + x1
            x = self.relu(x)

            x2 = x

            x = self.conv5(x)

            target_feature = x.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss5 = style_loss
            style_losses.append(self.style_loss5)

            x = self.bn5(x)
            x = self.relu(x)
            x = self.conv6(x)

            target_feature = x.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss6 = style_loss
            style_losses.append(self.style_loss6)

            x = self.bn6(x)
            x = self.relu(x)
            x = self.conv7(x)

            target_feature = x.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss7 = style_loss
            style_losses.append(self.style_loss7)

            x = self.bn7(x)+x2
            x = self.relu(x)

            x2 = x

            x = self.conv8(x)

            target_feature = x.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss8 = style_loss
            style_losses.append(self.style_loss8)

            x = self.bn8(x)
            x = self.relu(x)
            x = self.conv9(x)

            target_feature = x.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss9 = style_loss
            style_losses.append(self.style_loss9)

            x = self.bn9(x)
            x = self.relu(x)
            x = self.conv10(x)

            target_feature = x.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss10 = style_loss
            style_losses.append(self.style_loss10)


            x = self.bn10(x)+x2
            x = self.relu(x)
            x = self.conv11(x)

            target_feature = x.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss11 = style_loss
            style_losses.append(self.style_loss11)

            x = self.bn11(x)
            x = self.relu(x)
            x = self.conv12(x)

            target_feature = x.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss12 = style_loss
            style_losses.append(self.style_loss12)

            x = self.bn12(x)
            x = self.relu(x)
            x = self.conv13(x)

            target_feature = x.clone().detach()
            style_loss = StyleLoss(target_feature)
            self.style_loss13 = style_loss
            style_losses.append(self.style_loss13)


            #Расчет выходов для изображения - эталона содержания
            x = self.normalization(content_img)
            x = self.conv1(x)

            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x1 = self.conv2(x)

            x1 = self.bn2(x1)
            x1 = self.relu(x1)
            x1 = self.conv3(x1)

            x1 = self.bn3(x1)
            x1 = self.relu(x1)
            x1 = self.conv4(x1)

            x1 = self.bn4(x1)

            x = self.d_conv(x)
            x = self.d_bn(x)

            x = x + x1
            x = self.relu(x)

            x2 = x

            x = self.conv5(x)

            x = self.bn5(x)
            x = self.relu(x)
            x = self.conv6(x)

            x = self.bn6(x)
            x = self.relu(x)
            x = self.conv7(x)

            x = self.bn7(x)+x2
            x = self.relu(x)

            x2 = x

            x = self.conv8(x)

            x = self.bn8(x)
            x = self.relu(x)
            x = self.conv9(x)

            x = self.bn9(x)
            x = self.relu(x)
            x = self.conv10(x)


            x = self.bn10(x)+x2
            x = self.relu(x)
            x = self.conv11(x)

            target = x.clone().detach()
            content_loss = ContentLoss(target)
            self.cont_loss1 = content_loss
            content_losses.append(self.cont_loss1)       

            x = self.bn11(x)
            x = self.relu(x)
            x = self.conv12(x)

            #Сохраним все loss
            self.content_losses = content_losses
            self.style_losses = style_losses

        # Forward
        def forward(self, x):
            x = self.normalization(x)
            x = self.conv1(x)

            x = self.style_loss1(x)

            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x1 = self.conv2(x)

            x1 = self.style_loss2(x1)

            x1 = self.bn2(x1)
            x1 = self.relu(x1)
            x1 = self.conv3(x1)

            x1 = self.style_loss3(x1)

            x1 = self.bn3(x1)
            x1 = self.relu(x1)
            x1 = self.conv4(x1)

            x1 = self.style_loss4(x1)

            x1 = self.bn4(x1)

            x = self.d_conv(x)

            x = self.style_loss_d(x)

            x = self.d_bn(x)

            x = x + x1
            x = self.relu(x)

            x2 = x 

            x = self.conv5(x)

            x = self.style_loss5(x)

            x = self.bn5(x)
            x = self.relu(x)
            x = self.conv6(x)
            x = self.style_loss6(x)

            x = self.bn6(x)
            x = self.relu(x)
            x = self.conv7(x)
            x = self.style_loss7(x)

            x = self.bn7(x)+x2
            x = self.relu(x)

            x2 = x

            x = self.conv8(x)
            x = self.style_loss8(x)
            
            x = self.bn8(x)
            x = self.relu(x)
            x = self.conv9(x)
            x = self.style_loss9(x)

            x = self.bn9(x)
            x = self.relu(x)
            x = self.conv10(x)
            x = self.style_loss10(x)

            x = self.bn10(x)+x2
            x = self.relu(x)
            x = self.conv11(x)
            x = self.style_loss11(x)
            x = self.cont_loss1(x)

            x = self.bn11(x)
            x = self.relu(x)
            x = self.conv12(x)
            x = self.style_loss12(x)

            x = self.bn12(x)
            x = self.relu(x)
            x = self.conv13(x)
            x = self.style_loss13(x)

            return x



