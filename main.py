# 系统和基础库导入
import logging
import sys
import os
import time
import threading
import configparser
from logging.handlers import RotatingFileHandler
from datetime import datetime

# 图像处理相关库导入
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 深度学习相关库导入
import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# UI相关库导入
import pygame
import qdarkstyle
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QImage, QLinearGradient, QColor, QBrush
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from myui import Ui_MainWindow

# 新增人脸识别库导入
from face_recognition import FaceRecognition

class SecuritySystem:
    def __init__(self):
        # 设置日志
        self.setup_logging()
        
        # 初始化配置
        self.config = self._load_config()
        
        # 系统状态
        self.is_running = False
        self.is_detecting = False
        self.is_alerting = False
        self.is_recording = False
        self.alert_enabled = self.config.getboolean('Settings', 'alert_or_not')
        
        # 阈值设置
        self.face_conf_threshold = float(self.config.get('Settings', 'conf_threshold'))
        self.fire_conf_threshold = float(self.config.get('Settings', 'conf_threshold_fire'))
        self.alert_duration = float(self.config.get('Settings', 'alerting_time'))
        
        # 初始化人脸识别系统
        self.face_recognition = FaceRecognition('face_database.pkl', threshold=self.face_conf_threshold)
        
        # 音频初始化
        pygame.mixer.init()
        
        # 录制和截图相关
        self.output_dir = os.path.join(os.getcwd(), "security_records")
        self.snapshot_dir = self.output_dir  # 默认与录制目录相同
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_writer = None
        self.last_snapshot_time = 0
        self.snapshot_cooldown = 20  # 截图冷却时间（秒）
        
        # 初始化YOLO模型
        self.fire_model = self._init_fire_model()
        
    def setup_logging(self):
        """设置日志系统"""
        # 创建logs目录
        os.makedirs('logs', exist_ok=True)
        
        # 设置日志文件名（包含日期）
        log_file = f'logs/security_system_{datetime.now().strftime("%Y%m%d")}.log'
        
        # 配置日志处理器
        handler = RotatingFileHandler(
            log_file,
            maxBytes=1024*1024,  # 1MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # 配置根日志记录器
        self.logger = logging.getLogger('SecuritySystem')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        
    def log_event(self, message, level='info'):
        """记录事件到日志"""
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)

    def _load_config(self):
        """加载配置文件，如果不存在则创建默认配置"""
        config = configparser.ConfigParser()
        try:
            config.read('./settings.ini', encoding='utf-8')
        except Exception as e:
            logging.error(f"配置文件读取失败: {str(e)}")
            config['Settings'] = {
                'alert_or_not': 'False',
                'alerting_time': '5.0',
                'conf_threshold': '0.5',
                'conf_threshold_fire': '0.5',
                'mask_processingForm': 'True'
            }
            with open('./settings.ini', 'w', encoding='utf-8') as f:
                config.write(f)
            logging.info("已创建默认配置文件")
        return config

    def _init_fire_model(self):
        """初始化YOLO火灾检测模型"""
        try:
            model = YOLO('model/best.pt')
            model.conf = self.fire_conf_threshold  # 设置置信度阈值
            return model
        except Exception as e:
            self.log_event(f"加载YOLO模型失败: {str(e)}", 'error')
            return None

    def _detect_fire(self, frame):
        """使用YOLO模型进行火灾检测"""
        if self.fire_model is None:
            return False
            
        try:
            # 进行预测
            results = self.fire_model(frame, verbose=False)
            
            # 检查是否检测到火灾或烟雾
            for result in results:
                for cls, conf in zip(result.boxes.cls, result.boxes.conf):
                    # cls 0是火灾，1是烟雾
                    if conf > self.fire_conf_threshold:
                        return True
            
            return False
            
        except Exception as e:
            self.log_event(f"火灾检测出错: {str(e)}", 'error')
            return False

    def update_config(self, key, value):
        """更新配置文件"""
        self.config.set('Settings', key, str(value))
        with open('settings.ini', 'w') as f:
            self.config.write(f)

    def process_frame(self, frame):
        """处理单帧图像"""
        if frame is None:
            self.log_event("无法获取图像", 'warning')
            return None, "无法获取图像"

        # 调整图像大小
        frame = cv2.resize(frame, (500, 400))
        status_info = []

        if not self.is_detecting:
            return frame, "等待开始检测"

        # 重置警报指示灯状态
        self.reset_indicators()

        # 火灾检测
        has_fire = self._detect_fire(frame)
        if has_fire:
            self.log_event("检测到火灾或烟雾！", 'warning')
            status_info.append("警告：检测到火灾或烟雾！")
            self._trigger_fire_alarm()
            # 设置火灾指示灯
            ui.fire_indicator.setStyleSheet("""
                QLabel {
                    background-color: #FF0000;
                    color: white;
                    border: 2px solid #E8F5E9;
                    border-radius: 15px;
                    padding: 10px;
                    min-width: 100px;
                    text-align: center;
                }
            """)
        else:
            self.log_event("未检测到火灾或烟雾")
            status_info.append("未检测到火灾或烟雾")

        # 人脸检测和识别
        face_results = self.face_recognition.process_frame(frame)
        
        if not face_results:
            self.log_event("未检测到人脸")
            status_info.append("未检测到人脸")
        else:
            for result in face_results:
                if result['user'] == "unknown":
                    msg = f"检测到陌生人（置信度：{result['confidence']:.2f}）"
                    self.log_event(msg, 'warning')
                    status_info.append(msg)
                    self._trigger_stranger_alarm()
                    self._take_snapshot(frame, face_results)
                    # 设置陌生人指示灯
                    ui.stranger_indicator.setStyleSheet("""
                        QLabel {
                            background-color: #FF0000;
                            color: white;
                            border: 2px solid #E8F5E9;
                            border-radius: 15px;
                            padding: 10px;
                            min-width: 100px;
                            text-align: center;
                        }
                    """)
                else:
                    msg = f"检测到已知人员：{result['user']}（置信度：{result['confidence']:.2f}）"
                    self.log_event(msg)
                    status_info.append(msg)

        # 绘制结果
        processed_frame = self.face_recognition.draw_results(frame, face_results)
        return processed_frame, "\n".join(status_info)

    def _take_snapshot(self, frame, face_results):
        """拍摄陌生人照片（带冷却时间）"""
        current_time = time.time()
        if current_time - self.last_snapshot_time >= self.snapshot_cooldown:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            for idx, result in enumerate(face_results):
                if result['user'] == "unknown":
                    # 获取人脸框
                    box = result['box']
                    h, w = frame.shape[:2]
                    
                    # 扩展裁剪范围（扩展30%）
                    face_width = box[2] - box[0]
                    face_height = box[3] - box[1]
                    padding_x = int(face_width * 0.3)
                    padding_y = int(face_height * 0.3)
                    
                    # 确保扩展后的坐标在图像范围内
                    x1 = max(0, box[0] - padding_x)
                    y1 = max(0, box[1] - padding_y)
                    x2 = min(w, box[2] + padding_x)
                    y2 = min(h, box[3] + padding_y)
                    
                    # 裁剪人脸
                    face_img = frame[y1:y2, x1:x2]
                    
                    # 保存图片
                    filename = os.path.join(self.snapshot_dir, f"stranger_{timestamp}_{idx+1}.jpg")
                    cv2.imwrite(filename, face_img)
                    self.log_event(f"已保存陌生人照片：{filename}")
            
            self.last_snapshot_time = current_time

    def _trigger_fire_alarm(self):
        """触发火灾警报"""
        if not self.is_alerting and self.alert_enabled:
            self.is_alerting = True
            pygame.mixer.music.load('./fire_alarm.mp3')
            pygame.mixer.music.play()
            threading.Thread(target=self._play_alarm, args=('fire',)).start()

    def _trigger_stranger_alarm(self):
        """触发陌生人警报"""
        if not self.is_alerting and self.alert_enabled:
            self.is_alerting = True
            pygame.mixer.music.load('./stranger_alarm.mp3')
            pygame.mixer.music.play()
            threading.Thread(target=self._play_alarm, args=('stranger',)).start()

    def _play_alarm(self, alarm_type):
        """播放警报"""
        start_time = time.time()
        while self.alert_enabled and self.is_alerting:
            if time.time() - start_time >= self.alert_duration:
                break
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
            time.sleep(1)
        pygame.mixer.music.stop()
        self.is_alerting = False

    def start_recording(self):
        """开始录制视频"""
        if not self.is_recording:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
            self.video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                20.0,  # FPS
                (500, 400)  # 视频尺寸
            )
            self.is_recording = True
            ui.status_text.append("开始录制视频")
            ui.record_button.setText("停止录制")

    def stop_recording(self):
        """停止录制视频"""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            ui.status_text.append("视频录制已停止")
            ui.record_button.setText("开始录制")

    def start_camera(self):
        """启动摄像头和检测"""
        try:
            # 尝试打开摄像头
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                ui.status_text.append("默认摄像头打开失败，尝试其他摄像头...")
                cap = cv2.VideoCapture(1)

            if not cap.isOpened():
                ui.status_text.append("错误：无法打开任何摄像头")
                return

            ui.status_text.append("摄像头已成功打开")
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    ui.status_text.append("警告：无法读取摄像头画面")
                    continue

                try:
                    # 处理帧
                    processed_frame, status_info = self.process_frame(frame)
                    if processed_frame is not None:
                        # 转换为Qt图像并显示
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_frame.shape
                        qt_image = QImage(rgb_frame.data, w, h, w * ch, QImage.Format_RGB888)
                        ui.screen.setPixmap(QPixmap.fromImage(qt_image))
                        
                        # 更新状态信息
                        ui.status_text.append(status_info)
                        
                        # 如果正在录制，保存帧
                        if self.is_recording and self.video_writer is not None:
                            self.video_writer.write(processed_frame)

                    # 检查退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                except cv2.error as e:
                    ui.status_text.append(f"图像处理错误: {str(e)}")
                    continue

        except Exception as e:
            ui.status_text.append(f"发生错误: {str(e)}")
            
        finally:
            self.is_running = False
            if self.is_recording:
                self.stop_recording()
            if 'cap' in locals() and cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            ui.screen.setPixmap(QPixmap('./start_screen.jpg'))
            ui.screen.setScaledContents(True)
            ui.status_text.append("摄像头已关闭")

    def reset_indicators(self):
        """重置所有指示灯状态"""
        normal_style = """
            QLabel {
                background-color: #37474F;
                color: #E8F5E9;
                border: 2px solid #E8F5E9;
                border-radius: 15px;
                padding: 10px;
                min-width: 100px;
                text-align: center;
            }
        """
        ui.fire_indicator.setStyleSheet(normal_style)
        ui.smoke_indicator.setStyleSheet(normal_style)
        ui.stranger_indicator.setStyleSheet(normal_style)

# 创建系统实例
security_system = SecuritySystem()

def toggle_camera():
    """切换摄像头状态"""
    if security_system.is_running:
        security_system.is_running = False
        ui.camera_button.setText("打开摄像头")
        security_system.log_event("正在关闭摄像头...")
    else:
        security_system.is_running = True
        ui.camera_button.setText("关闭摄像头")
        security_system.log_event("正在打开摄像头...")
        threading.Thread(target=security_system.start_camera).start()

def toggle_detection():
    """切换检测状态"""
    security_system.is_detecting = not security_system.is_detecting
    ui.detect_button.setText("停止检测" if security_system.is_detecting else "开始检测")
    security_system.log_event("开始检测" if security_system.is_detecting else "已停止检测")

def update_face_threshold():
    """更新人脸检测阈值"""
    try:
        threshold = float(ui.face_threshold.text())
        if threshold < 0.1:
            threshold = 0.1
            ui.face_threshold.setText(str(threshold))
            show_warning("输入阈值不能低于0.1")
        elif threshold > 0.9:
            threshold = 0.9
            ui.face_threshold.setText(str(threshold))
            show_warning("输入阈值不能高于0.9")
        elif threshold < 0.8:
            show_warning("推荐为0.8-0.9，阈值过高容易漏报，过低容易误报")
        
        security_system.face_conf_threshold = threshold
        security_system.update_config('conf_threshold', threshold)
        
    except ValueError:
        show_warning("无法将输入转换为数字")
        ui.face_threshold.setText(str(security_system.face_conf_threshold))

def update_fire_threshold():
    """更新火灾检测阈值"""
    try:
        threshold = float(ui.fire_threshold.text())
        if threshold < 0.1:
            threshold = 0.1
            ui.fire_threshold.setText(str(threshold))
            show_warning("输入阈值不能低于0.1")
        elif threshold > 0.9:
            threshold = 0.9
            ui.fire_threshold.setText(str(threshold))
            show_warning("输入阈值不能高于0.9")
        elif not (0.6 <= threshold <= 0.7):
            show_warning("推荐为0.6-0.7，阈值过高容易漏报，过低容易误报")
        
        security_system.fire_conf_threshold = threshold
        security_system.update_config('conf_threshold_fire', threshold)
        
    except ValueError:
        show_warning("无法将输入转换为数字")
        ui.fire_threshold.setText(str(security_system.fire_conf_threshold))

def toggle_alert():
    """切换警报状态"""
    security_system.alert_enabled = not security_system.alert_enabled
    security_system.update_config('alert_or_not', security_system.alert_enabled)
    ui.alert_ignore.setText("屏蔽警报" if security_system.alert_enabled else "开启警报")

def toggle_recording():
    """切换录制状态"""
    if security_system.is_recording:
        security_system.stop_recording()
    else:
        security_system.start_recording()

def show_warning(message):
    """显示警告对话框"""
    msg_box = QMessageBox()
    msg_box.setWindowTitle("警告")
    msg_box.setText(message)
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.exec_()

def initialize_ui():
    """初始化UI界面"""
    window.setWindowTitle('智能家庭安防系统')
    ui.screen.setPixmap(QPixmap('./start_screen.jpg'))
    ui.screen.setScaledContents(True)
    
    # 设置按钮样式
    ui.camera_button.setStyleSheet('''QPushButton{background:#E8F5E9; color:#263238;}QPushButton:hover{background:#C8E6C9;}''')
    ui.detect_button.setStyleSheet('''QPushButton{background:#E8F5E9; color:#263238;}QPushButton:hover{background:#C8E6C9;}''')
    
    # 添加录制按钮
    ui.record_button = QtWidgets.QPushButton(ui.button_widget)
    ui.record_button.setMinimumSize(120, 60)
    ui.record_button.setText("开始录制")
    ui.record_button.setStyleSheet('''QPushButton{background:#E8F5E9; color:#263238;}QPushButton:hover{background:#C8E6C9;}''')
    ui.button_layout.addWidget(ui.record_button)
    
    # 设置目录路径
    ui.record_dir_path.setText(security_system.output_dir)
    ui.snapshot_dir_path.setText(security_system.output_dir)
    
    # 设置滑块初始值
    ui.fire_threshold.setValue(int(security_system.fire_conf_threshold * 100))
    ui.face_threshold.setValue(int(security_system.face_conf_threshold * 100))
    
    # 绑定事件
    ui.camera_button.clicked.connect(toggle_camera)
    ui.detect_button.clicked.connect(toggle_detection)
    ui.alert_ignore.clicked.connect(toggle_alert)
    ui.record_button.clicked.connect(toggle_recording)
    
    # 绑定新的事件
    ui.record_dir_button.clicked.connect(lambda: select_directory('record'))
    ui.snapshot_dir_button.clicked.connect(lambda: select_directory('snapshot'))
    ui.fire_threshold.valueChanged.connect(update_fire_threshold_from_slider)
    ui.face_threshold.valueChanged.connect(update_face_threshold_from_slider)

def select_directory(dir_type):
    """选择目录"""
    directory = QFileDialog.getExistingDirectory(window, "选择目录", "")
    if directory:
        if dir_type == 'record':
            ui.record_dir_path.setText(directory)
            security_system.output_dir = directory
        else:
            ui.snapshot_dir_path.setText(directory)
            security_system.snapshot_dir = directory
        
        # 确保目录存在
        os.makedirs(directory, exist_ok=True)

def update_fire_threshold_from_slider(value):
    """从滑块更新火灾检测阈值"""
    threshold = value / 100.0
    ui.fire_value_label.setText(f"{threshold:.2f}")
    security_system.fire_conf_threshold = threshold
    security_system.update_config('conf_threshold_fire', threshold)
    
    if not (0.5 <= threshold <= 0.9):
        show_warning("推荐为0.6-0.7，阈值过高容易漏报，过低容易误报")

def update_face_threshold_from_slider(value):
    """从滑块更新人脸识别阈值"""
    threshold = value / 100.0
    ui.face_value_label.setText(f"{threshold:.2f}")
    security_system.face_conf_threshold = threshold
    security_system.update_config('conf_threshold', threshold)
    
    if threshold < 0.6:
        show_warning("推荐为0.6-0.9，阈值过高容易漏报，过低容易误报")

if __name__ == "__main__":
    app = QApplication([])
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    # setup stylesheet
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyside2())
    
    initialize_ui()
    window.show()
    
    sys.exit(app.exec_())