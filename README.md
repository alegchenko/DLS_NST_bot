# DLS_NST_bot
DLS telegram-bots final project

Проект telegram-bots Вариант 1 (основная часть перенос стиля)

(комментарий: при обращение к боту в случаи не ответа попробуйте отправить одну из команд еше раз, при долгом простои бот не всегда реагирует на первое сообщение или же в случаи ошибки(напрмиер переполнения озу при super res перезапуститься и будет ждать новой команды)

Запуск хостинга осуществляется с помощью run_hosting.py

Отчет лежит в файлах репозитория в pdf формате (последние изминение было связано с этим)

Ссылка на отчёт (Аналог пдфки в репозитории)

https://docs.google.com/document/d/1DTQ6eirtPgJPjRkUd0xkAQULq22poC5l/edit?usp=sharing&ouid=113146755203817294600&rtpof=true&sd=true

Итог работы: 

1)Воспроизведена модель Леона Гатиса на основе VGG

2)Построена модель на основе ResNEt дающая отличающиеся результаты

С примерами полученными при переносе стилей можно ознакомиться в папке Exampels

С примерами получчеными при увеличение разрешения можно ознакомиться в Super_res_exampels

3)Написаный собственные классы для работы с данными моделями имеющие как методы для прямой обработки так и для хостинга в телеграм (классы my_layers, Full_bot и NST с его наслдениками, другие классы были взяты как готовая реализация доп опции super resolution и не содержат моих комментариев и работы)

4)Протестирована и интегрирована модель SRGAN (Нежелательно использовать большие изображения, из-за крайне большего расходу озу / памяти gpu) проверенный размер для хостинга 100x100

Комментарий про используемый сервер для хостинга, модель размешена на доступном мне по ssh ключу удаленному unix серверу. Для хостинга бота мной был:

1)установлен отсутствующий python3

2)установлено virtualenv. 

3)Создано виртуальное окружение python 3.6

4)установлены в нем необходимые библиотеки (pytorch, torchvision, PyTelegramBotAPI версии 4.4, telebot, PIL)

5)через scp загружена папка со всем необходимым

6)модель запущена на фоновую рабоу через nohup python3 run_hossting.py &
