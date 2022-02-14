from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import telebot
from telebot import types

import torchvision.transforms as transforms
import torchvision.models as models

import copy

from utils.NST import NST
from utils.RES_NST import SubModel
from utils.my_layers import ContentLoss, StyleLoss, Normalization
import sys
from PIL import Image
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Full_bot(NST):
      def __init__(self, vgg, res):
        super(Full_bot, self).__init__()
        self.imsize = 512
        self.epochs = 350 
        self.res = res 
        self.vgg = vgg
        self.is_vgg = True #Флаг определяющий выбранную модель

#Метод для вызова готового скрипта был наиболее простой интеграцией готовой модели SRGAN
      def super_res_run(self, path):
          status = os.system("python3 ./eval.py -c '' -i configs/srgan.json resources/pretrained/srgan.pth photos/img.jpg")
          return status

#Метод строящий необходимую часть модели с добавленными слоями выходов лосов
      def get_model_and_losses(self, content_img, style_img):
          #Проверим какая из моделей выбрана
          if self.is_vgg:
              #Сценарий для VGG
              self.epochs = 350
              cnn = self.vgg
              normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
              normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
              normalization = Normalization(normalization_mean, normalization_std).to(device)
              content_layers = ['conv_4']
              style_layers = ['conv_1','conv_2','conv_3','conv_4', 'conv_5']
#Оптимальная конфигурация 
              # just in order to have an iterable access to or list of content/syle
              # losses
              content_losses = []
              style_losses = []

              # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
              # to put in modules that are supposed to be activated sequentially
              model = nn.Sequential(normalization)
#Цикл пробегающий по слоя модели и создающий вставки выходов лосов при обнаружение Conv 
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

          #Если выбрана модель ResNet
          else:
              self.epochs = 700
              cnn = self.res 
              model = SubModel(cnn, content_img, style_img) #SubModel мой класс в котором прописаны пути вычислений по слоям взятым из resnet
              style_losses = model.style_losses
              content_losses = model.content_losses

          return model, style_losses, content_losses

      def start_bot(self, token):
          bot = telebot.TeleBot(token);

      #Описание сценариев для бота

          @bot.message_handler(commands=["start"])
          def start_message(message):
              markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
              btn1 = types.KeyboardButton('Перенос стиля')
              btn2 = types.KeyboardButton('Повышение разрешения')
              markup.add(btn1, btn2)
              bot.send_message(message.chat.id,'Здравствуйте! Какая из опций вас интересует?', reply_markup=markup)


          @bot.message_handler(commands=["help"])
          def help_message(message):
	            bot.send_message(message.chat.id,'Данный бот предоставляет опции переноса стиля между изображениями, а также 4ех кратного увеличения разрешения фотографий. Для выбора интересующей опции нажмите в меню start, для получения информации об используемых моделях, в меню нажмите info')

             
          @bot.message_handler(commands=["info"])
          def info_message(message):
              bot.send_message(message.chat.id,'Для Переноса стиля используется алгоритм Гатиса https://arxiv.org/abs/1508.06576, более подробно с реализацией можно ознакомиться в источнике https://pytorch.org/tutorials/advanced/neural_style_tutorial.html, доступны реализации c VGG19 и ResNet50. Модель повышения разрешения использует SRGAN https://arxiv.org/abs/1609.04802, код основан на репозитории https://github.com/mseitzer/srgan')

          

          @bot.message_handler(content_types=['text'])
          def start_nst(message):
              if message.text == '/start':
                  start_message(message)

              elif message.text == '/help':
                  help_message(message)

              elif message.text == '/info':
                  info_message(message)

              elif message.text is None:
                  bot.send_message(message.from_user.id, "Неверный тип сообщения");
                  start_nst(message)
              else:

                  if message.text == 'Перенос стиля':
                      bot.send_message(message.from_user.id, "Вы обратились к  NST боту для переноса стиля. Вы можете использовать одну из двух моделей: базовую - VGG или эксперементальную ResNet. С какой продолжим?");
                      markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                      btn1 = types.KeyboardButton('VGG')
                      btn2 = types.KeyboardButton('ResNet')
                      markup.add(btn1, btn2)
                      bot.send_message(message.from_user.id, "Какую модель будем использовать?", reply_markup=markup);
                      bot.register_next_step_handler(message, chose_model);
                  elif message.text == 'Повышение разрешения':
                      bot.send_message(message.from_user.id, "Отправьте изображение для обработки");
                      bot.register_next_step_handler(message, super_res);

          def chose_style_source(message):
                      markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                      btn1 = types.KeyboardButton('Свой стиль')
                      btn2 = types.KeyboardButton('Готовые стили')
                      markup.add(btn1, btn2)
                      bot.send_message(message.from_user.id, "Вы можете загрузить собственный образец стиля или же выбрать из каталога, где возьмём стиль?", reply_markup=markup);
                      bot.register_next_step_handler(message, chose);

          def chose_model(message):
            if message.text == '/start':
                start_message(message)

            elif message.text == '/help':
                help_message(message)

            elif message.text == '/info':
                info_message(message)

            elif message.text is None:
                bot.send_message(message.from_user.id, "Неверный тип сообщения");
                bot.register_next_step_handler(message, chose_model);
            else:
                if message.text == 'VGG':
                      bot.send_message(message.from_user.id, "Выбрана модель на основе VGG19")
                      self.is_vgg = True
                      chose_style_source(message)
                elif message.text == 'ResNet':
                      bot.send_message(message.from_user.id, "Выбрана модель на основе ResNet50")
                      self.is_vgg = False
                      chose_style_source(message)
                else:
                      markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                      btn1 = types.KeyboardButton('VGG')
                      btn2 = types.KeyboardButton('ResNet')
                      markup.add(btn1, btn2)
                      bot.send_message(message.from_user.id, "Неверное название модели, попробуйте еше раз", reply_markup=markup);
                      bot.register_next_step_handler(message, chose_model);

          def chose(message):
              if message.text == '/start':
                  start_message(message)

              elif message.text == '/help':
                  help_message(message)

              elif message.text == '/info':
                  info_message(message)

              elif message.text is None:
                  bot.send_message(message.from_user.id, "Неверный тип сообщения");
                  chose_style_source(message)
              else:

                  if message.text == 'Готовые стили':
                      
                      img = open('Styles.png', 'rb')
                      bot.send_photo(message.from_user.id, photo=img);
                      markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                      btn1 = types.KeyboardButton('1')
                      btn2 = types.KeyboardButton('2')
                      btn3 = types.KeyboardButton('3')
                      btn4 = types.KeyboardButton('4')
                      btn5 = types.KeyboardButton('5')
                      btn6 = types.KeyboardButton('6')
                      btn7 = types.KeyboardButton('Back')
                      markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7)
                      bot.send_message(message.from_user.id, "Отправьте номер выбранного стиля",reply_markup=markup);
                      bot.register_next_step_handler(message, def_styles);
                  elif message.text == 'Свой стиль':
                      bot.send_message(message.from_user.id, "Отправьте эталон стиля");
                      bot.register_next_step_handler(message, own_style);
                  else:
                      bot.send_message(message.from_user.id, "Неизвестная команда, еше раз ознакомьтесь с подсказками");
                      chose_style_source(message)



          def def_styles(message):
              if message.text == '/start':
                  start_message(message)

              elif message.text == '/help':
                  help_message(message)

              elif message.text == '/info':
                  info_message(message)

              elif message.text is None:
                  markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                  btn1 = types.KeyboardButton('1')
                  btn2 = types.KeyboardButton('2')
                  btn3 = types.KeyboardButton('3')
                  btn4 = types.KeyboardButton('4')
                  btn5 = types.KeyboardButton('5')
                  btn6 = types.KeyboardButton('6')
                  btn7 = types.KeyboardButton('Back')
                  markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7)
                  bot.send_message(message.from_user.id, "Неверный тип сообщения,повторите ввод");
                  bot.register_next_step_handler(message, def_styles);
                  
              else:

                  if message.text == '1':
                      self.style_img = self.image_loader('стиль1.jpg')
                      bot.send_message(message.from_user.id, "Загрузите изображение для трансформации");
                      bot.register_next_step_handler(message, trans_def_styles);

                  elif message.text == '2':
                      self.style_img = self.image_loader('стиль2.jpg')
                      bot.send_message(message.from_user.id, "Загрузите изображение для трансформации");
                      bot.register_next_step_handler(message, trans_def_styles);

                  elif message.text == '3':
                      self.style_img = self.image_loader('стиль3.jpg')
                      bot.send_message(message.from_user.id, "Загрузите изображение для трансформации");
                      bot.register_next_step_handler(message, trans_def_styles);

                  elif message.text == '4':
                      self.style_img = self.image_loader('стиль4.jpg')
                      bot.send_message(message.from_user.id, "Загрузите изображение для трансформации");
                      bot.register_next_step_handler(message, trans_def_styles);
                  
                  elif message.text == '5':
                      self.style_img = self.image_loader('стиль5.jpg')
                      bot.send_message(message.from_user.id, "Загрузите изображение для трансформации");
                      bot.register_next_step_handler(message, trans_def_styles);

                  elif message.text == '6':
                      self.style_img = self.image_loader('стиль6.jpg')
                      bot.send_message(message.from_user.id, "Загрузите изображение для трансформации");
                      bot.register_next_step_handler(message, trans_def_styles);

                  elif message.text == 'Back':
                      bot.register_next_step_handler(message, start_nst,reply_markup=markup);

                  else:
                      markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                      btn1 = types.KeyboardButton('1')
                      btn2 = types.KeyboardButton('2')
                      btn3 = types.KeyboardButton('3')
                      btn4 = types.KeyboardButton('4')
                      btn5 = types.KeyboardButton('5')
                      btn6 = types.KeyboardButton('6')
                      btn7 = types.KeyboardButton('Back')
                      markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7)
                      bot.send_message(message.from_user.id, "Ошибка, стиля с данным номером нет в каталоге, повторите ввод");
                      bot.register_next_step_handler(message, def_styles);

          def how_to_continue(message):
              if message.text == '/start':
                  start_message(message)

              elif message.text == '/help':
                  help_message(message)

              elif message.text == '/info':
                  info_message(message)

              elif message.text is None:
                  markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                  btn1 = types.KeyboardButton('Новый стиль')
                  btn2 = types.KeyboardButton('Готовые стили')
                  btn3 = types.KeyboardButton('Продолжить с тем же стилем')
                  markup.add(btn1, btn2, btn3)
                  bot.send_message(message.from_user.id, "Неверный тип сообщения, повторите ввод",reply_markup=markup);
                  bot.register_next_step_handler(message, how_to_continue);
              else:
                  if message.text == 'Новый стиль':
                      bot.send_message(message.from_user.id, "Отправьте эталон стиля");
                      bot.register_next_step_handler(message, own_style);
                  elif message.text == 'Продолжить с тем же стилем':
                      bot.send_message(message.from_user.id, "Отправьте изображение для трансформации");                  
                      bot.register_next_step_handler(message, trans_own_style);

                  elif message.text == 'Готовые стили':
                      markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                      btn1 = types.KeyboardButton('1')
                      btn2 = types.KeyboardButton('2')
                      btn3 = types.KeyboardButton('3')
                      btn4 = types.KeyboardButton('4')
                      btn5 = types.KeyboardButton('5')
                      btn6 = types.KeyboardButton('6')
                      btn7 = types.KeyboardButton('Back')
                      markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7)
                      bot.send_message(message.from_user.id, "Выбирите стиль",reply_markup=markup);
                      img = open('Styles.png', 'rb')
                      bot.send_photo(message.from_user.id, photo=img);                  
                      bot.register_next_step_handler(message, def_styles);

          def after_res(message):
              if message.text == '/start':
                  start_message(message)

              elif message.text == '/help':
                  help_message(message)

              elif message.text == '/info':
                  info_message(message)

              elif message.text is None:
                  bot.send_message(message.from_user.id, "Неверный тип сообщения");
                  bot.register_next_step_handler(message, how_to_continue);

          @bot.message_handler(content_types=['photo'])
          def super_res(message):
              if message.text == '/start':
                  start_message(message)

              elif message.text == '/help':
                  help_message(message)

              elif message.text == '/info':
                  info_message(message)

              elif message.photo is None:
                  bot.send_message(message.from_user.id, "Сообщение должно быть изображением, повторите попытку")
                  bot.register_next_step_handler(message, own_style);
              else:
                  file_info = bot.get_file(message.photo[-1].file_id)
                  downloaded_file = bot.download_file(file_info.file_path)
                  path = 'photos/img.jpg'
                  with open(path, 'wb') as new_file:
                  # записываем данные в файл
                      new_file.write(downloaded_file)
                  bot.send_message(message.from_user.id, "Начинаем обработку, это может занять некоторое время")
                  status = self.super_res_run(path)
                  if status == 0:
                      img = open('photos/img_pred.png', 'rb')
                      bot.send_photo(message.from_user.id, photo=img);
                      markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                      btn1 = types.KeyboardButton('Повторим')
                      btn2 = types.KeyboardButton('Вначало')
                      markup.add(btn1, btn2)
                      bot.send_message(message.from_user.id, "Чей займемся теперь?",reply_markup=markup)
                      bot.register_next_step_handler(message, after_res);
                  else:
                      bot.send_photo(message.from_user.id, photo=img);
                      markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                      btn1 = types.KeyboardButton('Повторим')
                      btn2 = types.KeyboardButton('Вначало')
                      markup.add(btn1, btn2)
                      bot.send_message(message.from_user.id, "Сбой обработки",reply_markup=markup)
                      bot.register_next_step_handler(message, after_res);
                  



          def own_style(message):
              if message.text == '/start':
                  start_message(message)

              elif message.text == '/help':
                  help_message(message)

              elif message.text == '/info':
                  info_message(message)

              elif message.photo is None:
                  bot.send_message(message.from_user.id, "Сообщение должно быть изображением, повторите попытку")
                  bot.register_next_step_handler(message, own_style);
              else:
                  file_info = bot.get_file(message.photo[-1].file_id)
                  downloaded_file = bot.download_file(file_info.file_path)
                  self.style_img = self.bytes_loader(downloaded_file)
                  bot.send_message(message.from_user.id, "Стиль принят, отправьте изображение для трансформации");
                  bot.register_next_step_handler(message, trans_own_style);

          def trans_own_style(message):
              if message.text == '/start':
                  start_message(message)

              elif message.text == '/help':
                  help_message(message)

              elif message.text == '/info':
                  info_message(message)
              elif message.photo is None:
                  bot.send_message(message.from_user.id, "Сообщение должно быть изображением, повторите попытку")
                  bot.register_next_step_handler(message, trans_own_style);
              else:
                  file_info = bot.get_file(message.photo[-1].file_id)
                  downloaded_file = bot.download_file(file_info.file_path)
                  self.content_img = self.bytes_loader(downloaded_file)
                  unloader = transforms.ToPILImage()
                  bot.send_message(message.from_user.id, "Начинаем перенос стиля, это может занять некоторое время");
                  out = self.run_style_transfer(self.content_img, self.style_img,num_steps = self.epochs)
                  image = out.cpu().clone()  
                  image = image.squeeze(0)      
                  image = unloader(image)
                  image.save("res.png")
                  img = open('res.png', 'rb')
                  bot.send_photo(message.from_user.id, photo=img);
                  markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                  btn1 = types.KeyboardButton('Новый стиль')
                  btn2 = types.KeyboardButton('Готовые стили')
                  btn3 = types.KeyboardButton('Продолжить с тем же стилем')
                  markup.add(btn1, btn2, btn3)
                  bot.send_message(message.from_user.id, "Для продолжения с тем же стилем отправьте 'Продолжить с тем же стилем', для загрузки нового стиля 'Новый стиль', для просмотра имеющихся стилей 'Готовые стили'",reply_markup=markup);
                  bot.register_next_step_handler(message, how_to_continue);



          def trans_def_styles(message):
              if message.text == '/start':
                  start_message(message)

              elif message.text == '/help':
                  help_message(message)

              elif message.text == '/info':
                  info_message(message)

              elif message.photo is None:
                  bot.send_message(message.from_user.id, "Сообщение должно быть изображением, повторите попытку")
                  bot.register_next_step_handler(message, trans_def_styles);
              else:
                  file_info = bot.get_file(message.photo[-1].file_id)
                  downloaded_file = bot.download_file(file_info.file_path)
                  self.content_img = self.bytes_loader(downloaded_file)
                  unloader = transforms.ToPILImage()
                  bot.send_message(message.from_user.id, "Начинаем перенос стиля, это может занять некоторое время");
                  out = self.run_style_transfer(self.content_img, self.style_img, num_steps = self.epochs)
                  image = out.cpu().clone()  
                  image = image.squeeze(0)      
                  image = unloader(image)
                  image.save("res.png")
                  img = open('res.png', 'rb')
                  bot.send_photo(message.from_user.id, photo=img);
                  markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                  btn1 = types.KeyboardButton('Новый стиль')
                  btn2 = types.KeyboardButton('Готовые стили')
                  btn3 = types.KeyboardButton('Продолжить с тем же стилем')
                  markup.add(btn1, btn2, btn3)
                  bot.send_message(message.from_user.id, "Для продолжения с тем же стилем отправьте 'Продолжить с тем же стилем', для загрузки своего стиля 'Новый стиль', для просмотра имеющихся стилей 'Готовые стили'",reply_markup=markup);
                  bot.register_next_step_handler(message, how_to_continue);



          #Запускаем хостинг бота
          bot.polling(none_stop=True, interval=0,timeout = 123)
