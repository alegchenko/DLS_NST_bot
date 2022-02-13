
from utils.RES_NST import  Res_NST
from utils.VGG_NST import VGG_NST
from utils.Full_bot import Full_bot
import torch

#Назначим вычисления на cpu
device = "cpu"

#Загрузим веса моделей
vgg = torch.load("weights/vgg")
res = torch.load("weights/ResNet")
bot = Full_bot(vgg,res)
bot.start_bot("5110316882:AAEXjriZPBJBP5xUqOYWoAm3XhU3H08ceYk")
while True:
	try:
		bot.start_bot("5110316882:AAEXjriZPBJBP5xUqOYWoAm3XhU3H08ceYk")
	except:
		continue


