
from utils.RES_NST import  Res_NST #класс умеющий делать NST с помошью resnet 50/101/152
from utils.VGG_NST import VGG_NST #класс умеющий делать NST с помошью vgg 16/19
from utils.Full_bot import Full_bot #класс содержаший функционал прошлых моделей и + super resolution
import torch

#Основная моя работа импортированные выше классы, большое кол-во других файлов относиться к готовой реализации SRGAN

#Назначим вычисления на gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Загрузим веса моделей
vgg = torch.load("weights/vgg").to(device).eval()
res = torch.load("weights/ResNet").to(device).eval()
bot = Full_bot(vgg,res)

#В моей реализации хостинг организованн ввиде метода для класса NST, но при этом классическое использование класса также возможно

#Цикл используется чтобы мог перезапуститься в случаи сбоев
while True:
	try:
		bot.start_bot("5110316882:AAEXjriZPBJBP5xUqOYWoAm3XhU3H08ceYk")
	except:
		continue


