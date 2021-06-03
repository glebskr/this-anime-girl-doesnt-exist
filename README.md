# AI powered Anime girls generator
# Артамонов Сергей
# Скроба Глеб
# ИДБ-18-09

#### dataset/data/ - должен содержать датасет с изображениями
#### results/checkpoints/model-[64]-[3]/ - должен хранить чекпоинты генератора и дискриминатора
#### для генерации изображений файл test.py ( geb_type - тип генерации( 'random_gen' и 'fix_hair_eye' ), hair и  eyes для выбора цвета)
####    #hair = ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 'pink', 'blue', 'black', 'brown', 'blonde']
####    #eyes = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 'brown', 'red', 'blue']

#### Запуск:
  +  `cd frontend && yarn && yarn build`
  +  `cd .. && pip install -r requirements.txt`
  +  `python app.py`
  
  
Натренированная модель:
https://drive.google.com/file/d/114Gn4zntp2nn_RlJsB9sER8Wy-6sPdwm/view?usp=sharing
