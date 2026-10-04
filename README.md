
# Анализатор эмоций на python

Десктоп программа заточенная на анализ эмоций с помощью модели DeepFace.Опирающейся на базовые 7 эмоций по системе американского психолога Пола Экмана.

___

## Установка зависимостей
Версия python3.12+



### Отдельная установка pytorch
Установка с поддержкой видеокарты(GPU) 

```bash
pip install pytorch
```
Облегченная установка без GPU для процессора

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
### Установка остальных зависимостей
Из файла зависимостей 
```bash
pip install -r requirements.txt
```
Напрямую
```bash
pip install PyQt6 opencv-python ultralytics deepface
```
___
При первом запуске файла main.py DeepFace загрузит веса модели.
___
Окно программы:
<img width="1917" height="1077" alt="image" src="https://github.com/user-attachments/assets/991a27d9-0ecb-462e-9c91-e893ff20584f" />