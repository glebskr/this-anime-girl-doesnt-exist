# AI powered Anime girls generator
<img src="https://media.giphy.com/media/dkaqOQtSlv2IJcux6q/giphy.gif" width="462" height="480" frameBorder="0" class="giphy-embed" allowFullScreen />


#### dataset/data/ - должен содержать датасет с изображениями
#### results/checkpoints/model-[64]-[3]/ - должен хранить чекпоинты генератора и дискриминатора
#### для обучения модели файл train.py
####    `hair = ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 'pink', 'blue', 'black', 'brown', 'blonde']`
####    `eyes = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 'brown', 'red', 'blue']`

#### Запуск:
  +  `cd frontend && yarn && yarn build`
  +  `cd .. && pip install -r requirements.txt`
  +  `python app.py`
  
  
Генератор и дискриминатор(чекпоинты):
https://drive.google.com/drive/folders/1VoJH1z-evTEorUpJAeopTxTh_77lBO_4?usp=sharing

Датасет(с изображениями):
https://drive.google.com/drive/folders/1qDwQCLvVCjBNBAKnQTPEqKMLdEIxZNjf?usp=sharing
