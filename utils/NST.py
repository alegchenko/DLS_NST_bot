from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

import sys
from PIL import Image
from io import BytesIO
import telebot
from telebot import types



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NST():
      def __init__(self):
        super(NST, self).__init__()
        self.imsize = 512
      

      def get_input_optimizer(self, input_img):
          # this line to show that input is a parameter that requires a gradient
          optimizer = optim.LBFGS([input_img],lr = 0.4)
          return optimizer

      def image_loader(self,image_name):
            loader = transforms.Compose([transforms.Resize(self.imsize),
            transforms.CenterCrop(self.imsize),
              # scale imported image
            transforms.ToTensor()]) 


            image = Image.open(image_name)
            # fake batch dimension required to fit network's input dimensions
            image = loader(image).unsqueeze(0)
            return image.to(device, torch.float)

      def bytes_loader(self,data):
            loader = transforms.Compose([transforms.Resize(self.imsize),
            transforms.CenterCrop(self.imsize),
              # scale imported image
            transforms.ToTensor()]) 

            stream = BytesIO(data)
            image = Image.open(stream).convert("RGB")
            stream.close()
            # fake batch dimension required to fit network's input dimensions
            image = loader(image).unsqueeze(0)
            return image.to(device, torch.float)



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
                  bot.register_next_step_handler(message, how_to_continue);
              else:

                  if message.text == 'Перенос стиля':
                      bot.send_message(message.from_user.id, "Вы обратились к  NST боту для переноса стиля. Вы можете использовать готовые стили (для просмотра стилей отправьте def_styles), также вы можете попробовать свой стиль, для этого отправьте own_style");
                      markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                      btn1 = types.KeyboardButton('Свой стиль')
                      btn2 = types.KeyboardButton('Готовые стили')
                      markup.add(btn1, btn2)
                      bot.send_message(message.from_user.id, "Будем использовать свой стиль или готовые?", reply_markup=markup);
                      bot.register_next_step_handler(message, chose);
              

         
          def chose(message):
              if message.text == '/start':
                  start_message(message)

              elif message.text == '/help':
                  help_message(message)

              elif message.text == '/info':
                  info_message(message)

              elif message.text is None:
                  bot.send_message(message.from_user.id, "Неверный тип сообщения");
                  bot.register_next_step_handler(message, how_to_continue);
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
                      bot.send_message(message.from_user.id, "Здравствуйте! вы обратились к  NST боту для переноса стиля. Вы можете использовать готовые стили (для просмотра стилей отправьте def_styles), также вы можете попробовать свой стиль, для этого отправьте own_style");
                      bot.register_next_step_handler(message, chose);



          def def_styles(message):
              if message.text == '/start':
                  start_message(message)

              elif message.text == '/help':
                  help_message(message)

              elif message.text == '/info':
                  info_message(message)

              elif message.text is None:
                  bot.send_message(message.from_user.id, "Неверный тип сообщения");
                  bot.register_next_step_handler(message, how_to_continue);
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
                  bot.send_message(message.from_user.id, "Неверный тип сообщения");
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

          @bot.message_handler(content_types=['photo'])
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
                  bot.send_message(message.from_user.id, "Для продолжения с тем же стилем отправьте continue, для загрузки нового стиля new_style, для просмотра имеющихся стилей def_styles",reply_markup=markup);
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
          bot.polling(none_stop=True, interval=0)
           

      #Общий метод для процесса переноса стиля, для разных CNN состоит в разлинчых методах возвращающих модель и лосы
      def run_style_transfer(self,content_img, style_img, num_steps= 300,
                       style_weight=1000000, content_weight=1):
          model, style_losses, content_losses = self.get_model_and_losses(content_img,style_img)

          input_img = content_img.clone()
          with torch.no_grad():
              input_img.clamp_(0, 1)  

          input_img.requires_grad_(True)
          model.requires_grad_(False)
          best_img = [input_img.clone()] #Запоминание изображения с лучшими показателями
          best_losses = []
#Шедулер был добавлен в связи с нестабильностью лосов в некоторых случаях и применяется при скачках лоса 
          optimizer = self.get_input_optimizer(input_img)
          scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

          print('Optimizing..')
          run = [0]
          hist = []
          while run[0] <= num_steps:

              def closure():
                  # correct the values of updated input image
                  with torch.no_grad():
                      input_img.clamp_(0, 1)
#Расчет лоссов на фичи cnn и оптимизации пикселей изображения для уменьшения суммарной ошбки
                  optimizer.zero_grad()
                  model(input_img)
                  style_score = 0
                  content_score = 0

                  for sl in style_losses:
                      style_score += sl.loss
                  for cl in content_losses:
                      content_score += cl.loss

                  style_score *= style_weight
                  content_score *= content_weight

                  loss = style_score + content_score
                  loss.backward()
                  hist.append(loss.clone().detach())
                  if(loss>torch.min(torch.tensor(hist[-4:]))*1.5):
                    scheduler.step()
                  if(run[0]>51):
                    if(loss<=torch.min(torch.tensor(hist[50:-1]))):
                      best_img[0] = input_img.clone()
                      best_losses.append([style_score,content_score])
                      

                  run[0] += 1
                  if run[0] % 50 == 0:
                      print("run {}:".format(run))
                      print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                          style_score.item(), content_score.item()))
                      print()

                  return style_score + content_score

              optimizer.step(closure)

          with torch.no_grad():
              input_img.clamp_(0, 1)

          return best_img[0]
