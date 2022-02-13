
from utils.RES_NST import  Res_NST #класс умеющий делать NST с помошью resnet 50/101/152
from utils.VGG_NST import VGG_NST #класс умеющий делать NST с помошью vgg 16/19
from utils.Full_bot import Full_bot #класс содержаший функционал прошлых моделей и + super resolution
import torch

#Назначим вычисления на cpu
device = "cpu"

#Загрузим веса моделей
vgg = torch.load("weights/vgg")
res = torch.load("weights/ResNet")
bot = Full_bot(vgg,res)

while True:
	try:
		bot.start_bot("5110316882:AAEXjriZPBJBP5xUqOYWoAm3XhU3H08ceYk")
	except:
		continue


