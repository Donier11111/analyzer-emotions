
# Анализатор эмоций на python

Десктоп программа заточенная на анализ эмоций с помощью модели DeepFace.Опирающейся на базовые 7 эмоций по системе американского психолога Пола Экмана.

___

## Требования
- Python 3.12–3.13, Linux (камеры открываются через V4L2)
- Файл весов `yolov8n-face-lindevs.pt` (https://github.com/lindevs/yolov8-face) положить рядом с `main.py`

## Установка
PyTorch для процессора (без видеокарты):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
или с поддержкой GPU:
```bash
pip install torch torchvision
```
Остальные зависимости:
```bash
pip install -r requirements.txt
```
или напрямую:
```bash
pip install PyQt6 opencv-python ultralytics deepface cv2-enumerate-cameras
```
Вместе с DeepFace автоматически установится TensorFlow (это большой пакет).
При первом запуске DeepFace сам скачает веса модели эмоций.