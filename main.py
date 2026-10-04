import sys
import time
import cv2
from cv2_enumerate_cameras import enumerate_cameras
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QComboBox, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO
from deepface import DeepFace

# Словарь для перевода эмоций с английского на русский
emotion_translation = {
    "angry": "Злой",
    "disgust": "Отвращение",
    "fear": "Страх",
    "happy": "Счастливый",
    "sad": "Грустный",
    "surprise": "Удивление",
    "neutral": "Нейтральный"
}

# Стили для оформления окна приложения
STYLESHEET = (
    "QWidget { background-color: #1a1515; color: #e6dbdb; font-family: 'JetBrainsMonoNF-Regular', 'Google Sans Flex', sans-serif; font-size: 14px; }\n"
    "QLabel#VideoScreen { background-color: #110e0e; border: 2px dashed #423535; border-radius: 12px; font-size: 18px; color: #8c7a7a; }\n"
    "QPushButton { background-color: #e59d9d; color: #1a1515; border: none; border-radius: 8px; padding: 10px 15px; font-weight: bold; }\n"
    "QPushButton:hover { background-color: #f4b8b8; }\n"
    "QPushButton:pressed { background-color: #c98585; margin-top: 2px; }\n"
    "QPushButton:disabled { background-color: #3b2f2f; color: #736262; }\n"
    "QComboBox { background-color: #2b2323; border: 1px solid #423535; border-radius: 8px; padding: 8px 15px; }\n"
    "QComboBox:disabled { background-color: #1a1515; color: #736262; }"
)

# Видео поток и анализ кадров
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    # Дополнительный сигнал для отправки сообщений в интерфейс (например, если видео закончилось)
    status_signal = pyqtSignal(str)
    # Сигнал для передачи пути сохраненного файла после завершения экспорта
    export_finished_signal = pyqtSignal(str)

    def __init__(self, video_source, export_path=None):
        super().__init__()
        self.video_source = video_source 
        self.export_path = export_path
        self.is_running = True
        self.save_path = export_path
        self.video_writer = None

    def run(self):
        # Загружаем модель для поиска лиц только при старте потока
        model = YOLO("yolov8n-face-lindevs.pt")
        
        # Для Linux V4L2 камер
        if isinstance(self.video_source, int) or (isinstance(self.video_source, str) and self.video_source.startswith("/dev/video")):
            cap = cv2.VideoCapture(self.video_source, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            self.status_signal.emit("Ошибка: Не удалось открыть видео источник.")
            return

        # Получаем FPS исходного файла или ставим базовый
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 60:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
        # Переменные для реализации оптимизации анализа
        frame_counter = 0
        skip_frames = 10
        last_data = []

        # Определение положения лица и его эмоций
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Оптимизация размера для уменьшения нагрузки на компьютер
            h, w = frame.shape[:2]
            new_w = 640
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h))

            # 2. Логика нейросетей
            if frame_counter % skip_frames == 0: # Оптимизациия анализа путем пропуска 10 кадров
                results = model(frame, stream=True, verbose=False)
                last_data = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0]) # Получение координат лица моделью YOLO 
                        
                        # Вычисляем 30% от ширины и высоты найденного лица
                        margin_x = int((x2 - x1) * 0.3)
                        margin_y = int((y2 - y1) * 0.3)
                        
                        # Расширяем рамку во все стороны, не выходя за края кадра
                        crop_x1 = max(0, x1 - margin_x)
                        crop_y1 = max(0, y1 - margin_y)
                        crop_x2 = min(new_w, x2 + margin_x)
                        crop_y2 = min(new_h, y2 + margin_y)
                        
                        # Вырезаем расширенную область
                        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2] 
                        
                        emotion_text = "Unknown" # Начальная эмоция
                        translated_emotion = "Неопределено"
                        
                        if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                            try:
                                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB) # Смена палитры из-за особенности CV
                                df_res = DeepFace.analyze(rgb_crop, actions=['emotion'], 
                                                          enforce_detection=False, # Определение эмоции моделью Deepface
                                                          detector_backend='skip', silent=True)
                                
                                if isinstance(df_res, list): # Выбор доминирующей эмоции из всех определенных 
                                    emotion_text = df_res[0]['dominant_emotion'] 
                                else:
                                    emotion_text = df_res['dominant_emotion']
                                translated_emotion = emotion_translation.get(emotion_text, "Неопределено")
                            except Exception:
                                pass # Игнорируем мелкие ошибки при смазанном лице
                                
                        last_data.append((x1, y1, x2, y2, translated_emotion))
            
            # 3. Отрисовка
            for (x1, y1, x2, y2, emotion) in last_data:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 4. Запись видео (если указан путь сохранения)
            if self.save_path:
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # Создаем файл по пути, который выбрал пользователь
                    self.video_writer = cv2.VideoWriter(self.save_path, fourcc, fps, (new_w, new_h))
                self.video_writer.write(frame)

            # 5. Вывод видео в окне программы
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, new_w, new_h, rgb_image.strides[0], QImage.Format.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)

            # Подсчет прогресса при обработке и экспорте
            if self.export_path and total_frames > 0:
                percent = int((frame_counter / total_frames) * 100)
                self.status_signal.emit(f"Обработка и сохранение: {percent}% ({frame_counter}/{total_frames})")
            
            frame_counter += 1

        # Очистка ресурсов при остановке
        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()

        if self.export_path and self.is_running:
            self.export_finished_signal.emit(self.export_path)
        else:
            self.status_signal.emit("Видео завершено.")

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


# Поток для плавного воспроизведения сохраненного видео
class PlaybackThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.is_running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 60:
            fps = 25.0
        delay = 1.0 / fps

        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            new_w = 640
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h))

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, new_w, new_h, rgb_image.strides[0], QImage.Format.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)
            time.sleep(delay)

        cap.release()
        self.status_signal.emit("Воспроизведение завершено.")

    def stop(self):
        self.is_running = False
        self.wait()


# Главный интерфейс программы
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализатор эмоций")
        self.resize(750, 650) 
        self.camera_mapping = {} 
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

        # Статусная строка для вывода информации и процентов
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #a8c793; font-weight: bold;")
        main_layout.addWidget(self.status_label)

        # Панель настроек
        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(10)
        
        self.source_combo = QComboBox()
        self.load_available_cameras()
        self.source_combo.currentIndexChanged.connect(self.toggle_source_mode)
        
        # Добавляем кнопку обновления
        self.btn_refresh = QPushButton("🔄 Обновить камеры")
        self.btn_refresh.clicked.connect(self.refresh_cameras)
        
        self.btn_select_file = QPushButton("Выбрать файл")
        self.btn_select_file.setEnabled(False) 
        self.btn_select_file.clicked.connect(self.select_input_file)
        
        settings_label = QLabel("Источник видео:")
        settings_label.setStyleSheet("font-weight: bold;")
        
        settings_layout.addWidget(settings_label)
        settings_layout.addWidget(self.btn_refresh) # Вставляем в слой
        settings_layout.addWidget(self.source_combo)
        settings_layout.addWidget(self.btn_select_file)
        settings_layout.addStretch()
        main_layout.addLayout(settings_layout)

        # Панель управления
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        
        self.btn_start = QPushButton("▶ Запустить")
        self.btn_start.clicked.connect(self.start_stream)

        # Кнопка для обработки и скачивания медиафайла
        self.btn_process_save = QPushButton("💾 Обработать и скачать")
        self.btn_process_save.clicked.connect(self.start_process_and_save)
        self.btn_process_save.setVisible(False)
        
        self.btn_stop = QPushButton("⏹ Остановить")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_stream)
        
        self.btn_record = QPushButton("⏺ Начать запись")
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self.toggle_record)
        
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_process_save)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addWidget(self.btn_record)
        main_layout.addLayout(controls_layout)

        self.setLayout(main_layout)

    # Сканирует систему через cv2_enumerate_cameras без отсечения индексов
    def load_available_cameras(self):
        self.source_combo.clear()
        self.camera_mapping.clear()
        
        added_paths = set()
        
        try:
            cameras = enumerate_cameras()
            for cam in cameras:
                # Используем системный путь устройства (напр. /dev/video0) или нормализованное имя
                device_key = getattr(cam, 'path', str(cam.index))
                if device_key not in added_paths:
                    display_name = f"📷 {cam.name} [{cam.index}]"
                    self.source_combo.addItem(display_name)
                    # Сохраняем системный путь, если доступен, либо числовой индекс
                    self.camera_mapping[display_name] = getattr(cam, 'path', cam.index)
                    added_paths.add(device_key)
        except Exception as e:
            print(f"Ошибка при поиске камер: {e}")

        # Если устройств нет совсем, ставим стандартный fallback
        if not self.camera_mapping:
            display_name = "📷 Стандартная веб-камера (0)"
            self.source_combo.addItem(display_name)
            self.camera_mapping[display_name] = 0

        # Всегда добавляем медиафайл в конец списка
        self.source_combo.addItem("📽️Медиафайл")

    # Обновляет список доступных устройств по кнопке
    def refresh_cameras(self):
        self.load_available_cameras()
        self.toggle_source_mode()

    # Логика переключения Веб-камера или Видеофайл
    def toggle_source_mode(self):
        is_media = (self.source_combo.currentText() == "📽️Медиафайл")
        self.btn_select_file.setEnabled(is_media)
        
        # Для медиафайла скрываем кнопку ручной записи и открываем кнопку готовой обработки
        self.btn_record.setVisible(not is_media)
        self.btn_process_save.setVisible(is_media)
        
        if is_media:
            has_file = bool(self.selected_file_path)
            self.btn_start.setEnabled(has_file)
            self.btn_process_save.setEnabled(has_file)
            if not has_file:
                self.image_label.setText("Выберите медиафайл для начала работы")
        else: # т.к система CV считывает 0 - как веб-камеру нам достаточно базовым значением поставить 0 а менять в зависимости от пользователя 
            self.btn_start.setEnabled(True)
            self.selected_file_path = None
            self.image_label.setText("Ожидание запуска...")
        self.status_label.setText("")

    # Выбор файла для анализа
    def select_input_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видеофайл", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_name:
            self.selected_file_path = file_name
            self.image_label.setText(f"Выбран файл:\n{file_name.split('/')[-1]}")
            self.btn_start.setEnabled(True)
            self.btn_process_save.setEnabled(True)

    # Запуск обработки на лету без сохранения
    def start_stream(self):
        current_selection = self.source_combo.currentText()
        if current_selection == "📽️Медиафайл":
            if not self.selected_file_path:
                self.image_label.setText("Сначала выберите файл!")
                return
            source = self.selected_file_path
        else:
            source = self.camera_mapping.get(current_selection, 0)
        
        # Создаем и запускаем поток
        self.thread = VideoThread(source)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.handle_status)
        self.thread.start()

        self.set_ui_state_running()

    # Запуск полной обработки с сохранением файла и последующим плавным показом
    def start_process_and_save(self):
        if not self.selected_file_path:
            self.image_label.setText("Сначала выберите файл!")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Сохранить видео как...", "processed_video.mp4", "Video Files (*.mp4)")
        if not save_path:
            return

        self.thread = VideoThread(self.selected_file_path, export_path=save_path)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.handle_status)
        self.thread.export_finished_signal.connect(self.on_export_finished)
        self.thread.start()

        self.set_ui_state_running()

    # Запуск воспроизведения уже готового видео
    def on_export_finished(self, saved_video_path):
        self.status_label.setText("Готово! Запуск плавного воспроизведения...")
        self.thread = PlaybackThread(saved_video_path)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.handle_status)
        self.thread.start()

    # Блокировка и разблокировка кнопок во время работы
    def set_ui_state_running(self):
        self.btn_start.setEnabled(False)
        self.btn_process_save.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.source_combo.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self.btn_select_file.setEnabled(False)
        self.image_label.setStyleSheet("border: 2px solid #a8c793; border-radius: 12px; background-color: #000;")

    # Реакция на паузу
    def stop_stream(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
            
        self.image_label.clear()
        self.image_label.setText("Видео остановлено")
        self.image_label.setStyleSheet("")
        self.status_label.setText("")
        
        self.btn_record.setText("⏺ Начать запись")
        self.btn_record.setStyleSheet("")
        
        is_media = (self.source_combo.currentText() == "📽️Медиафайл")
        has_file = bool(self.selected_file_path)
        
        # Возвращаем кнопки в исходное состояние
        self.btn_start.setEnabled(not is_media or has_file)
        self.btn_process_save.setEnabled(is_media and has_file)
        self.btn_stop.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.source_combo.setEnabled(True)
        self.btn_refresh.setEnabled(True)
        self.btn_select_file.setEnabled(is_media)

    # Логика сохранения видео с веб-камеры
    def toggle_record(self):
        if self.thread is None:
            return

        if self.btn_record.text() == "⏺ Начать запись":
            # Вызываем диалоговое окно для сохранения
            save_path, _ = QFileDialog.getSaveFileName(self, "Сохранить видео как...", "webcam_record.mp4", "Video Files (*.mp4)")
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
        self.status_label.setText(msg)
        # Если видео кончилось, сбрасываем интерфейс
        if msg in ("Видео завершено.", "Воспроизведение завершено."):
            self.stop_stream()

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    app.setStyleSheet(STYLESHEET)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())