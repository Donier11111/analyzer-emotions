import sys
import cv2
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QComboBox, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO
from deepface import DeepFace

STYLESHEET = """
QWidget {
    background-color: #1a1515; 
    color: #e6dbdb; /* Кремовый текст */
    font-family: 'JetBrainsMonoNF-Regular', 'Google Sans Flex', sans-serif; /* Твой системный шрифт */
    font-size: 14px;
}

QLabel#VideoScreen {
    background-color: #110e0e; 
    border: 2px dashed #423535;
    border-radius: 12px;
    font-size: 18px;
    color: #8c7a7a;
}

QPushButton {
    background-color: #e59d9d; 
    color: #1a1515; 
    border: none;
    border-radius: 8px;
    padding: 10px 15px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #f4b8b8; 
}

QPushButton:pressed {
    background-color: #c98585; 
    margin-top: 2px;
}

QPushButton:disabled {
    background-color: #3b2f2f; 
    color: #736262;
}

QComboBox {
    background-color: #2b2323;
    border: 1px solid #423535;
    border-radius: 8px;
    padding: 8px 15px;
}

QComboBox:disabled {
    background-color: #1a1515;
    color: #736262;
}
"""

#Видео поток и анализ кадров
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    # Дополнительный сигнал для отправки сообщений в интерфейс (например, если видео закончилось)
    status_signal = pyqtSignal(str)

    def __init__(self, video_source):
        super().__init__()
        self.video_source = video_source 
        self.is_running = True
        self.save_path = None
        self.video_writer = None

    def run(self):
        # Загружаем модель для поиска лиц только при старте потока
        model = YOLO("yolov8n-face-lindevs.pt")
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            self.status_signal.emit("Ошибка: Не удалось открыть видео источник.")
            return
            
        #Переменные для реализации оптимизации анализа
        frame_counter = 0
        skip_frames = 10
        last_data = []

        #Определение положения лица и его эмоций
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                self.status_signal.emit("Видео завершено.")
                break

            # 1. Оптимизация размера для уменьшения нагрузки на компьютер
            h, w = frame.shape[:2]
            new_w = 640
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h))

            # 2. Логика нейросетей
            if frame_counter % skip_frames == 0: #Оптимизациия анализа путем пропуска 10 кадров
                results = model(frame, stream=True, verbose=False)
                last_data = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0]) #Получение координат лица моделью YOLO 
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(new_w, x2), min(new_h, y2)
                        
                        face_crop = frame[y1:y2, x1:x2] #Выделение лица для дальнейшего анализа
                        emotion_text = "Unknown" #Начальная эмоция
                        
                        if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                            try:
                                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB) # Смена палитры из-за особенности CV
                                df_res = DeepFace.analyze(rgb_crop, actions=['emotion'], 
                                                          enforce_detection=False, #Определение эмоции моделью Deepface
                                                          detector_backend='skip', silent=True)
                                
                                if isinstance(df_res, list): #Выбор доминирующей эмоции из всех определенных 
                                    emotion_text = df_res[0]['dominant_emotion'] 
                                else:
                                    emotion_text = df_res['dominant_emotion']
                            except Exception:
                                pass # Игнорируем мелкие ошибки при смазанном лице
                        
                        last_data.append((x1, y1, x2, y2, emotion_text))
            
            # 3. Отрисовка
            for (x1, y1, x2, y2, emotion) in last_data:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 4. Запись видео (если указан путь сохранения)
            if self.save_path:
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # Создаем файл по пути, который выбрал пользователь
                    self.video_writer = cv2.VideoWriter(self.save_path, fourcc, 15.0, (new_w, new_h))
                self.video_writer.write(frame)

            # 5.Вывод видео в окне программы
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, new_w, new_h, rgb_image.strides[0], QImage.Format.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)
            
            frame_counter += 1

        # Очистка ресурсов при остановке
        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()

    def start_recording(self, path):
        self.save_path = path

    def stop_recording(self):
        self.save_path = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def stop(self):
        self.is_running = False
        self.wait() # Ждем корректного завершения потока


# --- 2. ГЛАВНЫЙ ИНТЕРФЕЙС (GUI) ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализатор эмоций")
        self.resize(750, 650) 
        
        self.thread = None # Переменная для хранения потока
        self.selected_file_path = None # Путь к загруженному видео
        
        self.init_ui()

    def init_ui(self):
        # Главный вертикальный слой
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Экран для видео
        self.image_label = QLabel("Ожидание запуска...")
        self.image_label.setObjectName("VideoScreen") 
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.image_label)

        # ПАНЕЛЬ НАСТРОЕК 
        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(10)
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["📷 Веб-камера", "📁 Видеофайл"])
        self.source_combo.currentIndexChanged.connect(self.toggle_source_mode)
        
        self.btn_select_file = QPushButton("Выбрать файл")
        self.btn_select_file.setEnabled(False) 
        self.btn_select_file.clicked.connect(self.select_input_file)
        
        settings_label = QLabel("Источник видео:")
        settings_label.setStyleSheet("font-weight: bold;")
        
        settings_layout.addWidget(settings_label)
        settings_layout.addWidget(self.source_combo)
        settings_layout.addWidget(self.btn_select_file)
        settings_layout.addStretch() 
        main_layout.addLayout(settings_layout)

        # ПАНЕЛЬ УПРАВЛЕНИЯ
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        
        self.btn_start = QPushButton("▶ Запустить")
        self.btn_start.clicked.connect(self.start_stream)
        
        self.btn_stop = QPushButton("⏹ Остановить")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_stream)
        
        self.btn_record = QPushButton("⏺ Начать запись")
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self.toggle_record)
        
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addWidget(self.btn_record)
        main_layout.addLayout(controls_layout)

        self.setLayout(main_layout)

    # Логика переключения Веб-камера или Видеофайл
    def toggle_source_mode(self):
        if self.source_combo.currentText() == "📁 Видеофайл":
            self.btn_select_file.setEnabled(True)
        else: #т.к система CV считывает 0 - как веб-камеру нам достаточно базовым значением поставить 0 а менять в зависимости от пользователя 
            self.btn_select_file.setEnabled(False)
            self.selected_file_path = None
            self.image_label.setText("Ожидание запуска...")

    # Выбор файла для анализа
    def select_input_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видеофайл", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_name:
            self.selected_file_path = file_name
            self.image_label.setText(f"Выбран файл:\n{file_name.split('/')[-1]}")

    # Запуск обработки
    def start_stream(self):
        source = 0 # По умолчанию веб-камера
        if self.source_combo.currentText() == "📁 Видеофайл":
            if not self.selected_file_path:
                self.image_label.setText("Сначала выберите файл!")
                return
            source = self.selected_file_path

        # Создаем и запускаем поток
        self.thread = VideoThread(source)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.handle_status)
        self.thread.start()

        # Состояние кнопок
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.source_combo.setEnabled(False)
        
        self.image_label.setStyleSheet("border: 2px solid #a8c793; border-radius: 12px; background-color: #000;")

    # Рекакция на паузу
    def stop_stream(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
            
        self.image_label.clear()
        self.image_label.setText("Видео остановлено")
        self.image_label.setStyleSheet("")
        
        self.btn_record.setText("⏺ Начать запись")
        self.btn_record.setStyleSheet("")
        
        # Возвращаем кнопки в исходное состояние
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.source_combo.setEnabled(True)

    # Логика сохранения видео
    def toggle_record(self):
        if self.thread is None:
            return

        if self.btn_record.text() == "⏺ Начать запись":
            # Вызываем диалоговое окно для сохранения
            save_path, _ = QFileDialog.getSaveFileName(self, "Сохранить видео как...", "output.mp4", "Video Files (*.mp4)")
            if save_path:
                self.thread.start_recording(save_path)
                self.btn_record.setText("⏸ Остановить запись")
                self.btn_record.setStyleSheet("background-color: #ed7979; color: #1a1515;")
            else:
                self.thread.stop_recording()
                self.btn_record.setText("⏺ Начать запись")
                self.btn_record.setStyleSheet("")

    def update_image(self, qt_image):
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def handle_status(self, msg):
        # Если видео кончилось, сбрасываем интерфейс
        if msg == "Видео завершено.":
            self.stop_stream()
        self.image_label.setText(msg)

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    app.setStyleSheet(STYLESHEET)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())