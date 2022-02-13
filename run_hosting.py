
from utils.RES_NST import  Res_NST #класс умеющий делать NST с помошью resnet 50/101/152
from utils.VGG_NST import VGG_NST #класс умеющий делать NST с помошью vgg 16/19
from utils.Full_bot import Full_bot #класс содержаший функционал прошлых моделей и + super resolution
import torch

#Назначим вычисления на cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Загрузим веса моделей
vgg = torch.load("weights/vgg").to(device).eval()
res = torch.load("weights/ResNet").to(device).eval()
bot = Full_bot(vgg,res)
bot.start_bot("5110316882:AAEXjriZPBJBP5xUqOYWoAm3XhU3H08ceYk")
while True:
	try:
		bot.start_bot("5110316882:AAEXjriZPBJBP5xUqOYWoAm3XhU3H08ceYk")
	except:
		continue


