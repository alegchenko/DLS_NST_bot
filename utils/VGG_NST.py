from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from PIL import Image
from io import BytesIO

import torchvision.transforms as transforms
import torchvision.models as models

import copy

from utils.NST import NST

from utils.my_layers import ContentLoss, StyleLoss, Normalization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Наследуемый класс добавляющий возможность работы с прямолинейными CNN, например VGG16 / VGG19
class VGG_NST(NST):
      def __init__(self, cnn):
        super(VGG_NST, self).__init__()
        self.imsize = 512
        self.epochs = 350
        self.cnn = cnn

      def get_model_and_losses(self, content_img, style_img):
          # normalization module
          cnn = self.cnn
          normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
          normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
          normalization = Normalization(normalization_mean, normalization_std).to(device)
          content_layers = ['conv_4']
          style_layers = ['conv_1','conv_2','conv_3','conv_4', 'conv_5']
# Оптимальная конфигурация
          # just in order to have an iterable access to or list of content/syle
          # losses
          content_losses = []
          style_losses = []

          # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
          # to put in modules that are supposed to be activated sequentially
          model = nn.Sequential(normalization)
#Пробегаем по слоям vgg копируя их и делая вставки для возможности получения промежуточных активаций и расчета лоссов на них
          i = 0  # increment every time we see a conv
          for layer in cnn.children():
              if isinstance(layer, nn.Conv2d):
                  i += 1
                  name = 'conv_{}'.format(i)
              elif isinstance(layer, nn.ReLU):
                  name = 'relu_{}'.format(i)
                  # The in-place version doesn't play very nicely with the ContentLoss
                  # and StyleLoss we insert below. So we replace with out-of-place
                  # ones here.
                  layer = nn.ReLU(inplace=False)
              elif isinstance(layer, nn.MaxPool2d):
                  name = 'pool_{}'.format(i)
              elif isinstance(layer, nn.BatchNorm2d):
                  name = 'bn_{}'.format(i)
              else:
                  raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

              model.add_module(name, layer)

              if name in content_layers:
                  # add content loss:
                  target = model(content_img).detach()
                  content_loss = ContentLoss(target)
                  model.add_module("content_loss_{}".format(i), content_loss)
                  content_losses.append(content_loss)

              if name in style_layers:
                  # add style loss:
                  target_feature = model(style_img).detach()
                  style_loss = StyleLoss(target_feature)
                  model.add_module("style_loss_{}".format(i), style_loss)
                  style_losses.append(style_loss)

          # now we trim off the layers after the last content and style losses
          for i in range(len(model) - 1, -1, -1):
              if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                  break

          model = model[:(i + 1)]

          return model, style_losses, content_losses


