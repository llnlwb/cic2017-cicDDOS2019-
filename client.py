import sys
import os
import configparser
import sqlite3
import hashlib
from datetime import datetime, timedelta
import logging
import requests
import json
import time
import pandas as pd
import numpy as np
from collections import deque
import threading
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QMessageBox, QDialog, QTableWidget, QTableWidgetItem, QMainWindow, QGridLayout,
    QFrame, QComboBox, QSpinBox, QProgressBar, QDoubleSpinBox, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from traffic_capture import TrafficCapture

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = 'client_config.ini'

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class UserAuth:
    def __init__(self, db_path='threat_logs.db'):
        """初始化数据库连接并创建表"""
        self.db_path = db_path
        if not os.path.exists(db_path):
            logger.info(f"创建新的数据库文件：{db_path}")
        try:
            self.conn = sqlite3.connect(db_path)
            logger.info(f"成功连接到数据库：{db_path}")
            self.create_tables()
        except Exception as e:
            logger.error(f"数据库连接失败：{str(e)}")
            raise

    def create_tables(self):
        """创建用户表和威胁日志表结构"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                locked INTEGER DEFAULT 0,
                failed_attempts INTEGER DEFAULT 0,
                last_failed_attempt TEXT,
                lock_expiry TEXT,
                created_at TEXT NOT NULL,
                password_reset_token TEXT,
                reset_token_expiry TEXT
            )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS threat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                attack_type TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                src_ip TEXT,
                dst_ip TEXT,
                src_port INTEGER,
                dst_port INTEGER,
                threat_score REAL,
                flow_id TEXT
            )''')
            self.conn.commit()
            logger.info("用户表和威胁日志表创建或已存在")
        except Exception as e:
            logger.error(f"创建表失败：{str(e)}")
            raise

    def authenticate(self, username, password):
        """用户认证"""
        cursor = self.conn.cursor()
        try:
            cursor.execute('''SELECT password_hash, locked, lock_expiry, failed_attempts 
                           FROM users 
                           WHERE username=?''', (username,))
            result = cursor.fetchone()
            if not result:
                logger.warning(f"用户不存在：{username}")
                return False, "用户不存在"
            stored_hash, locked, lock_expiry, attempts = result
            if locked and lock_expiry and datetime.now() < datetime.fromisoformat(lock_expiry):
                remaining = (datetime.fromisoformat(lock_expiry) - datetime.now()).seconds // 60
                logger.warning(f"账户已锁定：{username}，剩余时间：{remaining}分钟")
                return False, f"账户已锁定，剩余时间：{remaining}分钟"
            input_hash = hashlib.sha256(password.encode()).hexdigest()
            if input_hash != stored_hash:
                new_attempts = attempts + 1
                lock_time = datetime.now().isoformat() if new_attempts >= 3 else None
                lock_exp = (datetime.now() + timedelta(minutes=5)).isoformat() if new_attempts >= 3 else None
                cursor.execute('''UPDATE users SET 
                                failed_attempts=?, 
                                last_failed_attempt=?, 
                                locked=?, 
                                lock_expiry=? 
                                WHERE username=?''',
                               (new_attempts, lock_time, new_attempts >= 3, lock_exp, username))
                self.conn.commit()
                logger.warning(f"密码错误：{username}，剩余尝试次数：{3 - new_attempts}")
                if new_attempts >= 3:
                    return False, "密码错误次数过多，账户已锁定5分钟"
                return False, f"密码错误，剩余尝试次数：{3 - new_attempts}"
            cursor.execute('''UPDATE users SET 
                            failed_attempts=0, 
                            locked=0, 
                            lock_expiry=NULL
                            WHERE username=?''', (username,))
            self.conn.commit()
            logger.info(f"用户认证成功：{username}")
            return True, "认证成功"
        except Exception as e:
            logger.error(f"登录失败：{str(e)}")
            return False, f"登录失败：{str(e)}"

    def create_user(self, username, password, role='user'):
        """创建新用户"""
        if not self.validate_password_complexity(password):
            logger.warning(f"密码复杂度不足：{username}")
            return False, "密码必须包含大小写字母、数字和特殊字符，且长度至少8位"
        cursor = self.conn.cursor()
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            created_at = datetime.now().isoformat()
            cursor.execute('''INSERT INTO users 
                            (username, password_hash, role, created_at)
                            VALUES (?, ?, ?, ?)''',
                           (username, password_hash, role, created_at))
            self.conn.commit()
            logger.info(f"用户注册成功：{username}")
            return True, "用户注册成功"
        except sqlite3.IntegrityError:
            logger.warning(f"用户名已存在：{username}")
            return False, "用户名已存在"
        except Exception as e:
            logger.error(f"数据库错误：{str(e)}")
            return False, f"数据库错误：{str(e)}"

    def validate_password_complexity(self, password):
        """验证密码复杂度"""
        if len(password) < 8:
            return False
        if not any(c.isupper() for c in password):
            return False
        if not any(c.islower() for c in password):
            return False
        if not any(c.isdigit() for c in password):
            return False
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?~' for c in password):
            return False
        return True

class LoginDialog(QDialog):
    def __init__(self, auth_system):
        super().__init__()
        self.auth_system = auth_system
        self.initUI()

    def initUI(self):
        self.setWindowTitle('用户登录')
        self.setFixedSize(400, 300)
        self.setStyleSheet(""" 
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #e6e6e6);
                border-radius: 15px;
            }
            QLabel {
                font-size: 14px;
                color: #333;
                font-family: 'Arial';
            }
            QLineEdit {
                padding: 10px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 8px;
                background: #f9f9f9;
            }
            QLineEdit:focus {
                border: 1px solid #4a90e2;
                background: #ffffff;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4a90e2, stop:1 #357abd);
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #357abd, stop:1 #4a90e2);
            }
        """)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        title_label = QLabel("网络安全系统登录")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        self.lbl_username = QLabel('用户名:')
        self.txt_username = QLineEdit()
        self.txt_username.setPlaceholderText("请输入用户名")
        self.lbl_password = QLabel('密码:')
        self.txt_password = QLineEdit()
        self.txt_password.setEchoMode(QLineEdit.Password)
        self.txt_password.setPlaceholderText("请输入密码")
        self.btn_login = QPushButton('登录')
        self.btn_login.clicked.connect(self.authenticate_user)

        layout.addWidget(self.lbl_username)
        layout.addWidget(self.txt_username)
        layout.addWidget(self.lbl_password)
        layout.addWidget(self.txt_password)
        layout.addWidget(self.btn_login)

        self.setLayout(layout)

    def authenticate_user(self):
        username = self.txt_username.text().strip()
        password = self.txt_password.text().strip()
        if not username or not password:
            QMessageBox.warning(self, '输入错误', '用户名和密码不能为空')
            return
        success, message = self.auth_system.authenticate(username, password)
        if success:
            QMessageBox.information(self, '登录成功', '认证成功，正在跳转主界面...')
            self.accept()
        else:
            QMessageBox.critical(self, '登录失败', message)

class RegisterDialog(QDialog):
    def __init__(self, auth_system):
        super().__init__()
        self.auth_system = auth_system
        self.initUI()

    def initUI(self):
        self.setWindowTitle('用户注册')
        self.setFixedSize(400, 350)
        self.setStyleSheet(""" 
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #e6e6e6);
                border-radius: 15px;
            }
            QLabel {
                font-size: 14px;
                color: #333;
                font-family: 'Arial';
            }
            QLineEdit {
                padding: 10px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 8px;
                background: #f9f9f9;
            }
            QLineEdit:focus {
                border: 1px solid #4a90e2;
                background: #ffffff;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4a90e2, stop:1 #357abd);
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #357abd, stop:1 #4a90e2);
            }
        """)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        title_label = QLabel("用户注册")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        self.lbl_username = QLabel('用户名:')
        self.txt_username = QLineEdit()
        self.txt_username.setPlaceholderText("请输入用户名")
        self.lbl_password = QLabel('密码:')
        self.txt_password = QLineEdit()
        self.txt_password.setEchoMode(QLineEdit.Password)
        self.txt_password.setPlaceholderText("请输入密码")
        self.lbl_confirm = QLabel('确认密码:')
        self.txt_confirm = QLineEdit()
        self.txt_confirm.setEchoMode(QLineEdit.Password)
        self.txt_confirm.setPlaceholderText("请再次输入密码")
        self.btn_register = QPushButton('注册')
        self.btn_register.clicked.connect(self.register_user)

        layout.addWidget(self.lbl_username)
        layout.addWidget(self.txt_username)
        layout.addWidget(self.lbl_password)
        layout.addWidget(self.txt_password)
        layout.addWidget(self.lbl_confirm)
        layout.addWidget(self.txt_confirm)
        layout.addWidget(self.btn_register)

        self.setLayout(layout)

    def register_user(self):
        username = self.txt_username.text().strip()
        password = self.txt_password.text().strip()
        confirm = self.txt_confirm.text().strip()
        if not username or not password:
            QMessageBox.warning(self, '输入错误', '用户名和密码不能为空')
            return
        if password != confirm:
            QMessageBox.warning(self, '输入错误', '两次输入的密码不一致')
            return
        success, message = self.auth_system.create_user(username, password)
        if success:
            QMessageBox.information(self, '注册成功', message)
            self.accept()
        else:
            QMessageBox.critical(self, '注册失败', message)

class MainDialog(QDialog):
    def __init__(self, auth_system):
        super().__init__()
        self.auth_system = auth_system
        self.initUI()

    def initUI(self):
        self.setWindowTitle('欢迎使用网络安全系统')
        self.setFixedSize(400, 300)
        self.setStyleSheet(""" 
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #e6e6e6);
                border-radius: 15px;
            }
            QLabel {
                font-size: 18px;
                color: #333;
                font-family: 'Arial';
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4a90e2, stop:1 #357abd);
                color: white;
                font-size: 14px;
                padding: 12px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #357abd, stop:1 #4a90e2);
            }
        """)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        title_label = QLabel("网络安全态势评估系统")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        self.btn_login = QPushButton('登录')
        self.btn_login.clicked.connect(self.open_login)
        layout.addWidget(self.btn_login)

        self.btn_register = QPushButton('注册')
        self.btn_register.clicked.connect(self.open_register)
        layout.addWidget(self.btn_register)

        self.setLayout(layout)

    def open_login(self):
        login_dialog = LoginDialog(self.auth_system)
        if login_dialog.exec() == QDialog.Accepted:
            self.accept()

    def open_register(self):
        register_dialog = RegisterDialog(self.auth_system)
        if register_dialog.exec() == QDialog.Accepted:
            QMessageBox.information(self, "注册成功", "注册成功，请登录！")

class RecordDetailDialog(QDialog):
    def __init__(self, record, parent=None):
        super().__init__(parent)
        self.setWindowTitle("威胁记录详情")
        self.setFixedSize(500, 400)
        self.setStyleSheet(""" 
            QDialog {
                background: #ffffff;
                border-radius: 10px;
            }
            QLabel {
                font-size: 14px;
                color: #2d3436;
                font-family: 'Arial';
            }
            QPushButton {
                background: #4a90e2;
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background: #357abd;
            }
        """)
        layout = QVBoxLayout()
        layout.setSpacing(10)

        for key, value in record.items():
            label = QLabel(f"{key.replace('_', ' ').title()}: {value}")
            layout.addWidget(label)

        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

class RiskEvaluator(QThread):
    update_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    save_record_signal = pyqtSignal(dict)
    packet_batch_signal = pyqtSignal(pd.DataFrame)

    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        if not self.config:
            logger.error("配置文件加载失败，请检查配置文件内容。")
            raise ValueError("配置文件加载失败，请检查配置文件内容。")
        if "window_interval" not in self.config or "dashboard_update_interval" not in self.config:
            logger.error("配置文件中缺少 'window_interval' 或 'dashboard_update_interval' 配置项")
            raise ValueError("配置文件中缺少 'window_interval' 或 'dashboard_update_interval' 配置项")
        self.window = deque(maxlen=self.config["window_interval"])
        self.risk_scores = deque(maxlen=100)
        self.running = False
        self.mode = 'normal'
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.variance_threshold = 0.01
        self.top_k_features = 50
        self.feature_cols = [
            'Bwd Packet Length Std', 'Packet Length Std', 'Fwd IAT Std', 'Bwd Packet Length Mean',
            'Avg Bwd Segment Size', 'Bwd Packet Length Max', 'Packet Length Variance', 'Min Packet Length',
            'Fwd Packet Length Min', 'Idle Mean', 'Idle Max', 'Idle Min', 'Fwd IAT Max', 'Flow IAT Max',
            'Max Packet Length', 'Avg Fwd Segment Size', 'Fwd Packet Length Mean', 'Flow Packets/s',
            'Packet Length Mean', 'Fwd Packets/s', 'Average Packet Size', 'Flow IAT Std', 'Destination Port',
            'PSH Flag Count', 'Down/Up Ratio', 'Flow Bytes/s', 'Flow Duration', 'Fwd IAT Total', 'Protocol',
            'Bwd Packet Length Min', 'Source Port', 'ACK Flag Count', 'FIN Flag Count', 'Bwd IAT Std',
            'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Max', 'Fwd Packet Length Max', 'Subflow Fwd Bytes',
            'Total Length of Fwd Packets', 'Init_Win_bytes_forward', 'URG Flag Count', 'Bwd IAT Total',
            'Fwd Packet Length Std', 'Fwd PSH Flags', 'Bwd Packets/s', 'Init_Win_bytes_backward',
            'SYN Flag Count', 'Bwd IAT Mean', 'min_seg_size_forward'
        ]
        self.attack_risk_scores = {
            "BENIGN": 0,
            "DoS": 6,
            "DrDoS": 6,
            "PortScan": 8,
            "Bot": 8,
            "WebDDoS": 6,
            "Heartbleed": 12,
            "Syn": 8,
            "UDPLag": 10
        }
        self.use_kafka = self.config.get("use_kafka", "False").lower() == "true" and self.mode == 'normal'
        self.kafka_producer = None
        if self.use_kafka and self.mode == 'normal':
            try:
                from kafka import KafkaProducer
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.config["kafka_broker"],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                logger.info("Kafka 生产者初始化成功")
            except Exception as e:
                logger.error(f"Kafka 生产者初始化失败: {e}")
                self.use_kafka = False
                self.kafka_producer = None
        self.test_data = None
        self.test_index = 0
        self.traffic_capture = TrafficCapture(self.config)
        self.traffic_capture.set_log_signal(self.log_signal)
        self.traffic_capture.packet_batch_signal.connect(self.process_packet_batch_slot)

    def load_config(self):
        config = configparser.ConfigParser()
        try:
            if not os.path.exists(CONFIG_PATH):
                logger.error(f"配置文件 {CONFIG_PATH} 不存在，请检查路径。")
                return {}
            config.read(CONFIG_PATH)
            conf_dict = {
                "server_url": config.get("SERVER", "url", fallback="http://127.0.0.1:5000"),
                "high_threshold": float(config.get("RISK", "high_risk_threshold", fallback="0.8")),
                "low_threshold": float(config.get("RISK", "low_risk_threshold", fallback="0.5")),
                "window_interval": int(config.get("RISK", "window_interval", fallback="30")),
                "dashboard_update_interval": int(config.get("RISK", "dashboard_update_interval", fallback="10")),
                "kafka_broker": config.get("KAFKA", "broker_address", fallback="localhost:9092"),
                "kafka_topic": config.get("KAFKA", "topic", fallback="network_traffic"),
                "db_host": config.get("DATABASE", "host", fallback="localhost"),
                "db_user": config.get("DATABASE", "user", fallback="root"),
                "db_password": config.get("DATABASE", "password", fallback="123456"),
                "db_name": config.get("DATABASE", "name", fallback="network_security"),
                "use_kafka": config.get("KAFKA", "use_kafka", fallback="False"),
                "test_data_path": config.get("DATA", "test_data_path", fallback="train_data.csv"),
                "interface": config.get("TRAFFIC_CAPTURE", "interface", fallback="eth0"),
                "capture_interval": int(config.get("TRAFFIC_CAPTURE", "capture_interval", fallback="10"))
            }
            logger.info(f"加载的配置文件内容：{conf_dict}")
            return conf_dict
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}

    def clean_data(self, df):
        """数据清洗"""
        zero_cols = df.columns[(df == 0).all()]
        if zero_cols.any():
            df = df.drop(zero_cols, axis=1)
            logger.info(f"已删除全零列: {list(zero_cols)}")
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif df[col].dtype == 'object':
                df[col] = df[col].fillna('unknown')
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.ffill().bfill()
        return df

    def select_discriminative_features(self, df):
        """特征选择，保留 src_ip, dst_ip, src_port, dst_port 和 Flow ID_str"""
        numeric_cols = df.select_dtypes(include=np.number).columns
        variances = df[numeric_cols].var()
        low_variance_cols = variances[variances < self.variance_threshold].index
        if low_variance_cols.any():
            df = df.drop(columns=low_variance_cols)
            logger.info(f"已删除低方差列（方差<{self.variance_threshold}）：{list(low_variance_cols)}")
            numeric_cols = df.select_dtypes(include=np.number).columns
        
        selected_cols = []
        if len(numeric_cols) > 0 and 'Label' in df.columns:
            try:
                selector = SelectKBest(score_func=f_classif, k=min(self.top_k_features, len(numeric_cols)))
                selector.fit(df[numeric_cols], df['Label'])
                scores = pd.Series(selector.scores_, index=numeric_cols)
                selected_cols = scores.sort_values(ascending=False).index[:self.top_k_features].tolist()
                logger.info(f"选择的top特征（基于ANOVA F值）：{selected_cols}")
            except Exception as e:
                logger.warning(f"SelectKBest 特征选择失败：{str(e)}，使用所有数值列")
                selected_cols = numeric_cols.tolist()
        else:
            selected_cols = numeric_cols.tolist()
        
        # 强制保留非数值列
        preserved_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'Flow ID_str']
        for col in preserved_cols:
            if col in df.columns and col not in selected_cols:
                selected_cols.append(col)
        
        valid_selected_cols = [col for col in selected_cols if col in df.columns]
        if len(valid_selected_cols) < len(selected_cols):
            missing_cols = [col for col in selected_cols if col not in df.columns]
            logger.warning(f"以下特征列在 DataFrame 中缺失，已移除：{missing_cols}")
        
        logger.debug(f"最终选择的特征列：{valid_selected_cols}")
        return df[valid_selected_cols]

    def process_features(self, df):
        """特征处理"""
        if 'Protocol' in df.columns:
            df['Protocol'] = self.label_encoder.fit_transform(df['Protocol'].astype(str))
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df

    def process_packet_batch_slot(self, df):
        """处理捕获的数据包批次"""
        if df is not None and not df.empty:
            logger.info(f"[数据处理 - 开始] 处理捕获的数据包批次，记录数: {len(df)}")
            try:
                processed_df = self.clean_data(df)
                processed_df = self.select_discriminative_features(processed_df)
                processed_df = self.process_features(processed_df)
                self.send_batch_to_server(processed_df)
                logger.info("[数据处理 - 完成] 数据包批次处理完成")
            except Exception as e:
                logger.error(f"[数据处理 - 错误] 数据包批次处理失败：{str(e)}")
                self.log_signal.emit(f"错误: 数据包批次处理失败: {str(e)}")
        else:
            logger.warning("[数据处理 - 警告] 接收到空数据包批次")

    def process_packet(self, packet):
        """处理单个数据包，兼容测试模式"""
        try:
            features = {
                "src_ip": str(packet.get("src_ip", "")),
                "dst_ip": str(packet.get("dst_ip", "")),
                "src_port": int(packet.get("src_port", 0)),
                "dst_port": int(packet.get("dst_port", 0)),
                "protocol": str(packet.get("protocol", "")),
                "packet_length": float(packet.get("packet_length", 0)),
                "duration": float(packet.get("duration", 0)),
                "bytes_sent": float(packet.get("bytes_sent", 0)),
                "bytes_received": float(packet.get("bytes_received", 0)),
                "timestamp": datetime.now().timestamp(),
                "Flow ID_str": packet.get("Flow ID_str", "Unknown")
            }
            for col in self.feature_cols:
                if col in packet:
                    try:
                        value = packet[col]
                        if isinstance(value, str):
                            value = float(value) if value.strip() else 0.0
                        features[col] = float(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"特征 {col} 转换失败，设置为 0.0: {e}")
                        features[col] = 0.0
                else:
                    features[col] = 0.0
            features = {k: v for k, v in features.items() if k != 'Label'}
            logger.debug(f"处理的数据包特征：{features}")
            if self.mode == 'normal' and self.use_kafka and self.kafka_producer:
                self.kafka_producer.send(self.config["kafka_topic"], features)
                self.kafka_producer.flush()
            return features
        except Exception as e:
            logger.error(f"数据包处理失败: {str(e)}")
            return None

    def send_to_server(self, features, retries=3, timeout=5):
        """发送特征到服务器或本地推理"""
        try:
            from local_inference import LocalInference
            local_inf = LocalInference(
                model_path="best_student_model.pth",
                scaler_path="scaler.pkl"
            )
            attack_type, confidence = local_inf.infer(features)
            if attack_type and confidence:
                logger.info(f"本地推理成功: Attack Type={attack_type}, Confidence={confidence:.4f}")
                return {"attack_type": attack_type, "confidence": confidence}
            else:
                logger.warning("本地推理失败，尝试服务器推理")
        except Exception as e:
            logger.error(f"本地推理失败: {str(e)}")

        for attempt in range(retries):
            try:
                model_features = {k: features[k] for k in self.feature_cols}
                model_features["Flow ID_str"] = features.get("Flow ID_str", "Unknown")
                logger.debug(f"发送到服务器的特征：{model_features}")
                response = requests.post(
                    f"{self.config['server_url']}/predict",
                    json={"features": [model_features]},
                    timeout=timeout
                )
                logger.info(f"推理服务响应状态码：{response.status_code}, 内容：{response.text}")
                if response.status_code == 200:
                    results = response.json()
                    logger.debug(f"推理结果：{results}")
                    return results[0] if results else None
                else:
                    logger.error(f"推理服务返回错误: {response.status_code}, 响应内容: {response.text}")
            except Exception as e:
                logger.error(f"推理服务调用失败 (尝试 {attempt + 1}/{retries}): {str(e)}")
                time.sleep(1)
        return None

    def send_batch_to_server(self, df):
        """批量发送数据到服务器"""
        if df.empty:
            logger.warning("[批量发送 - 警告] 数据帧为空，跳过发送")
            return
        try:
            features_list = df.to_dict('records')
            model_features_list = []
            for row in features_list:
                model_features = {}
                for col in self.feature_cols:
                    model_features[col] = row.get(col, 0.0)
                model_features["Flow ID_str"] = row.get("Flow ID_str", f"{row.get('src_port', '0')}-{row.get('dst_port', '0')}")
                model_features["src_ip"] = row.get("src_ip", "")
                model_features["dst_ip"] = row.get("dst_ip", "")
                model_features["src_port"] = row.get("src_port", 0)
                model_features["dst_port"] = row.get("dst_port", 0)
                model_features_list.append(model_features)
                logger.debug(f"构造的 features: {model_features}")
            if self.mode == 'normal' and self.use_kafka and self.kafka_producer:
                for features in model_features_list:
                    self.kafka_producer.send(self.config["kafka_topic"], features)
                self.kafka_producer.flush()
            response = requests.post(
                f"{self.config['server_url']}/predict",
                json={"features": model_features_list},
                timeout=10
            )
            if response.status_code == 200:
                results = response.json()
                for features, result in zip(features_list, results):
                    risk_level, threat_score = self.classify_risk(
                        result["attack_type"], result["confidence"]
                    )
                    record = {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": result["attack_type"],
                        "risk": risk_level,
                        "threat_score": threat_score,
                        "src_ip": features.get("src_ip", ""),
                        "dst_ip": features.get("dst_ip", ""),
                        "src_port": features.get("src_port", 0),
                        "dst_port": features.get("dst_port", 0),
                        "flow_id": features.get("Flow ID_str", "Unknown")
                    }
                    logger.debug(f"生成的记录: {record}")
                    self.save_record_signal.emit(record)
                    self.window.append(record)
                    self.risk_scores.append(threat_score)
                    log_msg = (f"{record['time']} - Flow ID: {record['flow_id']}, "
                              f"攻击类型: {record['type']}, "
                              f"风险等级: {record['risk']}, 威胁分数: {record['threat_score']:.2f}")
                    self.log_signal.emit(log_msg)
            else:
                logger.error(f"批量推理失败: {response.status_code}, 响应内容: {response.text}")
        except Exception as e:
            logger.error(f"批量发送到服务器失败: {str(e)}")
            self.log_signal.emit(f"错误: 批量发送到服务器失败: {str(e)}")

    def classify_risk(self, attack_type, confidence):
        base_score = self.attack_risk_scores.get(attack_type, 0)
        if attack_type == "BENIGN":
            return "SAFE", 0.0
        if confidence > self.config["high_threshold"]:
            final_score = base_score * 0.75
            return "HIGH_RISK", final_score
        elif confidence > self.config["low_threshold"]:
            final_score = base_score * 0.3
            return "LOW_RISK", final_score
        else:
            final_score = 0.0
            return "SAFE", final_score

    def evaluate_window(self, time_window=None):
        """评估最近 time_window 秒的数据，或整个 window"""
        attack_counts = {attack: 0 for attack in self.attack_risk_scores.keys()}
        total_score = 0
        if time_window:
            cutoff_time = datetime.now().timestamp() - time_window
            recent_records = [
                r for r in self.window 
                if datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S").timestamp() >= cutoff_time
            ]
        else:
            recent_records = list(self.window)
        
        logger.debug(f"[RiskEvaluator - evaluate_window] 最近记录数: {len(recent_records)}")
        non_safe = [r for r in recent_records if r["risk"] != "SAFE"]
        high_count = sum(1 for r in non_safe if r["risk"] == "HIGH_RISK")
        low_count = sum(1 for r in non_safe if r["risk"] == "LOW_RISK")
        total = len(non_safe)
        risk_ratio = total / self.config["window_interval"] if self.config["window_interval"] > 0 else 0

        for record in recent_records:
            attack_counts[record["type"]] += 1
            total_score += record.get("threat_score", 0)

        logger.debug(f"[RiskEvaluator - evaluate_window] 攻击类型统计: {attack_counts}")
        status = "GREEN"
        if total_score > 60:
            status = "RED"
        elif total_score >= 30:
            status = "YELLOW"

        recent_records = recent_records[-10:]

        return {
            "status": status,
            "high_risk": high_count,
            "low_risk": low_count,
            "total_events": total,
            "risk_ratio": round(risk_ratio, 2),
            "risk_score": round(total_score, 2),
            "recent_records": recent_records,
            "attack_counts": attack_counts
        }

    def load_test_data(self):
        try:
            file_path = self.config.get("test_data_path", "test_data.csv")
            if not os.path.exists(file_path):
                logger.error(f"测试数据文件不存在: {file_path}")
                self.log_signal.emit(f"错误: 测试数据文件不存在: {file_path}")
                return False
            self.test_data = pd.read_csv(file_path)
            if len(self.test_data) < 2:
                logger.error("测试数据文件至少需要2行数据")
                self.log_signal.emit("错误: 测试数据文件至少需要2行数据")
                return False
            self.test_data.columns = [col.strip() for col in self.test_data.columns]
            # 确保测试数据包含必要的字段，并填充更有意义的默认值
            required_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "Flow ID_str"]
            ports = [443, 80, 5000]  # 目标端口列表
            for col in required_cols:
                if col not in self.test_data.columns:
                    logger.warning(f"测试数据缺少字段: {col}，将填充默认值")
                    if col == "src_ip":
                        self.test_data[col] = pd.Series(["192.168.1." + str(i % 255) for i in range(len(self.test_data))])
                    elif col == "dst_ip":
                        self.test_data[col] = pd.Series(["172.26.162.173" for _ in range(len(self.test_data))])
                    elif col == "src_port":
                        self.test_data[col] = pd.Series([10000 + i for i in range(len(self.test_data))])
                    elif col == "dst_port":
                        self.test_data[col] = pd.Series([ports[i % 3] for i in range(len(self.test_data))])
                    elif col == "Flow ID_str":
                        self.test_data[col] = pd.Series([f"flow_{i}" for i in range(len(self.test_data))])
            # 更新现有列以确保目标IP和端口符合要求
            self.test_data['dst_ip'] = "172.26.162.173"
            self.test_data['dst_port'] = pd.Series([ports[i % 3] for i in range(len(self.test_data))])
            for col in self.feature_cols:
                if col not in self.test_data.columns:
                    logger.warning(f"测试数据缺少特征列: {col}，将填充为0.0")
                    self.test_data[col] = 0.0
                else:
                    self.test_data[col] = pd.to_numeric(self.test_data[col], errors='coerce').fillna(0.0)
            self.test_index = 0
            logger.info(f"成功加载测试数据，共{len(self.test_data)}行")
            return True
        except Exception as e:
            logger.error(f"加载测试数据失败: {str(e)}")
            self.log_signal.emit(f"错误: 加载测试数据失败: {str(e)}")
            return False

    def process_test_batch(self, start_idx, batch_size):
        end_idx = min(start_idx + batch_size, len(self.test_data))
        batch_df = self.test_data.iloc[start_idx:end_idx]
        if not batch_df.empty:
            self.send_batch_to_server(batch_df)
        return end_idx, []

    def run(self):
        self.running = True
        last_update = time.time()
        if self.mode == 'test':
            if not self.load_test_data():
                logger.error("测试模式初始化失败，无法加载测试数据")
                self.running = False
                return
            logger.info("测试模式已启动，从CSV文件读取数据")
            batch_size = 10
            while self.running:
                try:
                    start_idx = self.test_index
                    self.test_index, batch_results = self.process_test_batch(start_idx, batch_size)
                    if time.time() - last_update >= self.config["dashboard_update_interval"]:
                        stats = self.evaluate_window(time_window=self.config["dashboard_update_interval"])
                        stats["risk_trend"] = list(self.risk_scores)[-50:]
                        self.update_signal.emit(stats)
                        last_update = time.time()
                    if self.test_index >= len(self.test_data):
                        self.test_index = 0
                    if self.running:
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"测试模式处理错误: {str(e)}")
                    self.log_signal.emit(f"错误: 测试模式处理错误: {str(e)}")
                    time.sleep(1)
        else:
            logger.info("正常模式已启动，启动流量捕获")
            self.traffic_capture.capture_traffic()
            while self.running:
                if time.time() - last_update >= self.config["dashboard_update_interval"]:
                    stats = self.evaluate_window(time_window=self.config["dashboard_update_interval"])
                    stats["risk_trend"] = list(self.risk_scores)[-50:]
                    self.update_signal.emit(stats)
                    last_update = time.time()
                time.sleep(1)

    def stop(self):
        self.running = False
        self.wait()
        self.traffic_capture.stop()
        if self.kafka_producer:
            self.kafka_producer.close()
            self.kafka_producer = None
        self.window.clear()
        self.risk_scores.clear()

    def set_mode(self, mode):
        if self.running:
            logger.warning("请先停止监控再切换模式")
            return False
        if mode not in ['normal', 'test']:
            logger.error(f"无效的模式: {mode}")
            return False
        self.mode = mode
        logger.info(f"模式已设置为: {mode}")
        self.use_kafka = self.config.get("use_kafka", "False").lower() == "true" and self.mode == 'normal'
        if not self.use_kafka and self.kafka_producer:
            self.kafka_producer.close()
            self.kafka_producer = None
        return True

    def load_sample_data(self):
        return []

    def update_thresholds(self, high_threshold, low_threshold):
        self.config["high_threshold"] = high_threshold
        self.config["low_threshold"] = low_threshold
        logger.info(f"更新风险阈值: 高风险={high_threshold}, 低风险={low_threshold}")

class DashboardUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("网络安全态势评估系统")
        self.setGeometry(100, 100, 1600, 1200)
        self.auth_db = UserAuth()

        self.evaluator = RiskEvaluator()
        self.evaluator.update_signal.connect(self.update_dashboard)
        self.evaluator.log_signal.connect(self.update_log)
        self.evaluator.save_record_signal.connect(self.save_to_database)
        self.init_ui()
        self.start_button.setEnabled(True)
        self.mode = 'normal'

    def init_ui(self):
        self.setStyleSheet(""" 
            QMainWindow {
                background: #f5f6fa;
                font-family: 'Arial';
            }
            QLabel {
                color: #2d3436;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4a90e2, stop:1 #357abd);
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #357abd, stop:1 #4a90e2);
            }
            QPushButton:disabled {
                background: #b0b0b0;
                color: #d0d0b0;
            }
            QTableWidget {
                background: #ffffff;
                border: 1px solid #dfe4ea;
                border-radius: 8px;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 10px;
            }
            QHeaderView::section {
                background: #f1f3f5;
                padding: 10px;
                border: none;
                font-size: 13px;
            }
            QFrame#card {
                background: #ffffff;
                border: 1px solid #dfe4ea;
                border-radius: 12px;
                margin: 10px;
                padding: 15px;
            }
            QComboBox {
                padding: 8px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 8px;
                background: #f9f9f9;
            }
            QComboBox:hover {
                border: 1px solid #4a90e2;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 8px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 8px;
                background: #f9f9f9;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #4a90e2;
            }
            QProgressBar {
                border: 1px solid #dfe4ea;
                border-radius: 5px;
                text-align: center;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background: #4a90e2;
                border-radius: 5px;
            }
        """)

        main_widget = QWidget()
        main_layout = QGridLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("网络安全态势评估系统")
        title_label.setFont(QFont("Arial", 26, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label, 0, 0, 1, 3)

        status_card = QFrame()
        status_card.setObjectName("card")
        status_layout = QVBoxLayout()
        status_layout.setSpacing(15)
        self.status_label = QLabel("安全状态：未知")
        self.status_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.risk_score_label = QLabel("风险分数：0.00")
        self.risk_score_label.setFont(QFont("Arial", 18))
        self.running_status_label = QLabel("运行状态：已停止")
        self.running_status_label.setFont(QFont("Arial", 16))
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.risk_score_label)
        status_layout.addWidget(self.running_status_label)
        status_card.setLayout(status_layout)
        main_layout.addWidget(status_card, 1, 0, 1, 1)

        stats_card = QFrame()
        stats_card.setObjectName("card")
        stats_layout = QGridLayout()
        stats_layout.setSpacing(15)
        self.high_risk_label = QLabel("高风险事件：0")
        self.low_risk_label = QLabel("低风险事件：0")
        self.total_events_label = QLabel("总风险事件：0")
        self.risk_ratio_label = QLabel("风险比例：0.0%")
        self.risk_ratio_bar = QProgressBar()
        self.risk_ratio_bar.setMaximum(100)
        self.risk_ratio_bar.setValue(0)
        for i, label in enumerate([self.high_risk_label, self.low_risk_label, self.total_events_label, self.risk_ratio_label]):
            label.setFont(QFont("Arial", 16))
            stats_layout.addWidget(label, i // 2, i % 2)
        stats_layout.addWidget(self.risk_ratio_bar, 2, 0, 1, 2)
        stats_card.setLayout(stats_layout)
        main_layout.addWidget(stats_card, 1, 1, 1, 1)

        trend_card = QFrame()
        trend_card.setObjectName("card")
        trend_layout = QVBoxLayout()
        trend_layout.addWidget(QLabel("风险分数趋势（最近50次）"))
        self.trend_figure = plt.Figure(figsize=(8, 4))
        self.trend_canvas = FigureCanvas(self.trend_figure)
        self.trend_canvas.setMinimumHeight(300)
        trend_layout.addWidget(self.trend_canvas)
        trend_card.setLayout(trend_layout)
        main_layout.addWidget(trend_card, 1, 2, 1, 1)

        chart_card = QFrame()
        chart_card.setObjectName("card")
        chart_layout = QVBoxLayout()
        chart_layout.addWidget(QLabel("最近10秒攻击类型统计"))
        self.figure = plt.Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)
        chart_layout.addWidget(self.canvas)
        chart_card.setLayout(chart_layout)
        main_layout.addWidget(chart_card, 2, 0, 1, 3)

        control_card = QFrame()
        control_card.setObjectName("card")
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["实时监控模式", "测试模式"])
        self.mode_combo.currentTextChanged.connect(self.set_mode)

        self.start_button = QPushButton("开始监控")
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button = QPushButton("停止监控")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.refresh_button = QPushButton("刷新数据")
        self.refresh_button.clicked.connect(self.refresh_data)
        self.update_model_button = QPushButton("更新模型")
        self.update_model_button.clicked.connect(self.update_model)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("高风险阈值："))
        self.high_threshold_spin = QDoubleSpinBox()
        self.high_threshold_spin.setRange(0.0, 1.0)
        self.high_threshold_spin.setSingleStep(0.1)
        self.high_threshold_spin.setValue(self.evaluator.config["high_threshold"])
        threshold_layout.addWidget(self.high_threshold_spin)
        threshold_layout.addWidget(QLabel("低风险阈值："))
        self.low_threshold_spin = QDoubleSpinBox()
        self.low_threshold_spin.setRange(0.0, 1.0)
        self.low_threshold_spin.setSingleStep(0.1)
        self.low_threshold_spin.setValue(self.evaluator.config["low_threshold"])
        threshold_layout.addWidget(self.low_threshold_spin)
        apply_threshold_button = QPushButton("应用阈值")
        apply_threshold_button.clicked.connect(self.apply_thresholds)
        threshold_layout.addWidget(apply_threshold_button)

        for widget in [self.mode_combo, self.start_button, self.stop_button, self.refresh_button, self.update_model_button]:
            control_layout.addWidget(widget)
        control_layout.addLayout(threshold_layout)
        control_layout.addStretch()
        control_card.setLayout(control_layout)
        main_layout.addWidget(control_card, 3, 0, 1, 3)

        log_card = QFrame()
        log_card.setObjectName("card")
        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("最近威胁日志（双击查看详情）"))
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(9)
        self.log_table.setHorizontalHeaderLabels([
            "时间", "流ID", "攻击类型", "风险等级", "威胁分数",
            "源IP", "目标IP", "源端口", "目标端口"
        ])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log_table.verticalHeader().setDefaultSectionSize(40)
        self.log_table.setStyleSheet(""" 
            QTableWidget {
                background: #ffffff;
                border: 1px solid #dfe4ea;
                border-radius: 8px;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 10px;
            }
            QHeaderView::section {
                background: #f1f3f5;
                padding: 10px;
                border: none;
                font-size: 13px;
            }
        """)
        self.log_table.setSortingEnabled(True)
        self.log_table.doubleClicked.connect(self.show_record_detail)
        log_layout.addWidget(self.log_table)
        log_card.setLayout(log_layout)
        main_layout.addWidget(log_card, 4, 0, 2, 3)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.reset_dashboard()

    def reset_dashboard(self):
        self.status_label.setText("安全状态：未知")
        self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2d3436; background-color: #ffffff; padding: 10px; border-radius: 8px;")
        self.risk_score_label.setText("风险分数：0.00")
        self.running_status_label.setText("运行状态：已停止")
        self.high_risk_label.setText("高风险事件：0")
        self.low_risk_label.setText("低风险事件：0")
        self.total_events_label.setText("总风险事件：0")
        self.risk_ratio_label.setText("风险比例：0.0%")
        self.risk_ratio_bar.setValue(0)
        self.trend_figure.clear()
        self.trend_canvas.draw()
        self.figure.clear()
        self.canvas.draw()
        self.log_table.setRowCount(0)

    def set_mode(self, mode_text):
        mode = 'normal' if mode_text == "实时监控模式" else 'test'
        if self.evaluator.set_mode(mode):
            self.mode = mode
            self.start_button.setEnabled(True)
            self.reset_dashboard()
            QMessageBox.information(self, "模式切换", f"已切换至{mode.upper()}模式")
        else:
            QMessageBox.warning(self, "模式切换失败", "请检查模式或数据文件")

    def start_monitoring(self):
        if not self.evaluator.isRunning():
            self.evaluator.start()
            self.running_status_label.setText("运行状态：正在运行")
            QMessageBox.information(self, "提示", "监控已启动")

    def stop_monitoring(self):
        if self.evaluator.isRunning():
            self.evaluator.stop()
            self.running_status_label.setText("运行状态：已停止")
            self.reset_dashboard()
            QMessageBox.information(self, "提示", "监控已停止")

    def refresh_data(self):
        if self.evaluator.isRunning():
            stats = self.evaluator.evaluate_window(time_window=self.evaluator.config["dashboard_update_interval"])
            stats["risk_trend"] = list(self.evaluator.risk_scores)[-50:]
            self.evaluator.update_signal.emit(stats)
            QMessageBox.information(self, "提示", "数据已刷新")
        else:
            QMessageBox.warning(self, "提示", "请先启动监控")

    def apply_thresholds(self):
        high_threshold = self.high_threshold_spin.value()
        low_threshold = self.low_threshold_spin.value()
        if low_threshold >= high_threshold:
            QMessageBox.warning(self, "输入错误", "低风险阈值必须小于高风险阈值")
            return
        self.evaluator.update_thresholds(high_threshold, low_threshold)
        QMessageBox.information(self, "成功", "风险阈值已更新")

    def update_model(self):
        try:
            from model_updater import ModelUpdater
            updater = ModelUpdater(flask_url=self.evaluator.config["server_url"])
            if updater.update_model():
                QMessageBox.information(self, "成功", "模型更新成功")
            else:
                QMessageBox.critical(self, "失败", "模型更新失败")
        except Exception as e:
            logger.error(f"模型更新失败: {str(e)}")
            QMessageBox.critical(self, "失败", f"模型更新失败: {str(e)}")

    def show_record_detail(self, index):
        row = index.row()
        record = {
            "time": self.log_table.item(row, 0).text() if self.log_table.item(row, 0) else "N/A",
            "flow_id": self.log_table.item(row, 1).text() if self.log_table.item(row, 1) else "N/A",
            "attack_type": self.log_table.item(row, 2).text() if self.log_table.item(row, 2) else "N/A",
            "risk_level": self.log_table.item(row, 3).text() if self.log_table.item(row, 3) else "N/A",
            "threat_score": self.log_table.item(row, 4).text() if self.log_table.item(row, 4) else "N/A",
            "source_ip": self.log_table.item(row, 5).text() if self.log_table.item(row, 5) else "N/A",
            "destination_ip": self.log_table.item(row, 6).text() if self.log_table.item(row, 6) else "N/A",
            "source_port": self.log_table.item(row, 7).text() if self.log_table.item(row, 7) else "N/A",
            "destination_port": self.log_table.item(row, 8).text() if self.log_table.item(row, 8) else "N/A"
        }
        dialog = RecordDetailDialog(record, self)
        dialog.exec()

    def update_log(self, message):
        """处理日志信号并显示到日志表格"""
        logger.debug(f"接收到日志消息：{message}")
        print(message)

    def update_dashboard(self, stats):
        logger.info(f"更新仪表盘，接收到的统计数据：{stats}")
        self.status_label.setText(f"安全状态：{stats['status']}")
        self.risk_score_label.setText(f"风险分数：{stats['risk_score']:.2f}")
        color_map = {
            "GREEN": {"color": "#27ae60", "bg": "#e8f5e9"},
            "YELLOW": {"color": "#f39c12", "bg": "#fff9e6"},
            "RED": {"color": "#e74c3c", "bg": "#ffebee"}
        }
        status_style = f"font-size: 24px; font-weight: bold; color: {color_map[stats['status']]['color']}; background-color: {color_map[stats['status']]['bg']}; padding: 10px; border-radius: 8px;"
        self.status_label.setStyleSheet(status_style)
        self.high_risk_label.setText(f"高风险事件：{stats['high_risk']}")
        self.low_risk_label.setText(f"低风险事件：{stats['low_risk']}")
        self.total_events_label.setText(f"总风险事件：{stats['total_events']}")
        self.risk_ratio_label.setText(f"风险比例：{stats['risk_ratio']*100:.1f}%")
        self.risk_ratio_bar.setValue(int(stats['risk_ratio'] * 100))
        self.update_bar_chart(stats['attack_counts'])
        self.update_trend_chart(stats.get('risk_trend', []))
        self.update_log_table(stats['recent_records'])

    def update_bar_chart(self, attack_counts):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        attacks = list(attack_counts.keys())
        counts = list(attack_counts.values())
        logger.debug(f"[DashboardUI - update_bar_chart] 攻击类型: {attacks}, 计数: {counts}")
        if sum(counts) == 0:
            ax.text(0.5, 0.5, '暂无攻击数据', horizontalalignment='center', verticalalignment='center', fontsize=14, transform=ax.transAxes)
            ax.set_xlabel('攻击类型', fontsize=12)
            ax.set_ylabel('出现次数', fontsize=12)
            ax.set_title('最近10秒攻击类型统计', fontsize=14)
        else:
            ax.bar(attacks, counts, color='#4a90e2')
            ax.set_xlabel('攻击类型', fontsize=12)
            ax.set_ylabel('出现次数', fontsize=12)
            ax.set_title('最近10秒攻击类型统计', fontsize=14)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        self.figure.tight_layout()
        self.canvas.draw()

    def update_trend_chart(self, risk_trend):
        self.trend_figure.clear()
        ax = self.trend_figure.add_subplot(111)
        if not risk_trend:
            ax.text(0.5, 0.5, '暂无风险趋势数据', horizontalalignment='center', verticalalignment='center', fontsize=14, transform=ax.transAxes)
        else:
            ax.plot(risk_trend, marker='o', color='#4a90e2')
            ax.set_xlabel('时间', fontsize=12)
            ax.set_ylabel('风险分数', fontsize=12)
            ax.set_title('风险分数趋势', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
        self.trend_figure.tight_layout()
        self.trend_canvas.draw()

    def update_log_table(self, recent_records):
        logger.debug(f"更新日志表格，recent_records: {recent_records}")
        self.log_table.setRowCount(len(recent_records))
        for row, record in enumerate(recent_records):
            self.log_table.setItem(row, 0, QTableWidgetItem(record.get('time', 'N/A')))
            self.log_table.setItem(row, 1, QTableWidgetItem(record.get('flow_id', 'N/A')))
            self.log_table.setItem(row, 2, QTableWidgetItem(record.get('type', 'N/A')))
            self.log_table.setItem(row, 3, QTableWidgetItem(record.get('risk', 'N/A')))
            self.log_table.setItem(row, 4, QTableWidgetItem(f"{record.get('threat_score', 0.0):.2f}"))
            self.log_table.setItem(row, 5, QTableWidgetItem(str(record.get('src_ip', ''))))
            self.log_table.setItem(row, 6, QTableWidgetItem(str(record.get('dst_ip', ''))))
            self.log_table.setItem(row, 7, QTableWidgetItem(str(record.get('src_port', 0))))
            self.log_table.setItem(row, 8, QTableWidgetItem(str(record.get('dst_port', 0))))

    def save_to_database(self, record):
        try:
            cursor = self.auth_db.cursor()
            cursor.execute('''INSERT INTO threat_logs (
                timestamp, attack_type, risk_level, src_ip, dst_ip, 
                src_port, dst_port, threat_score, flow_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (
                               record['time'],
                               record['type'],
                               record['risk'],
                               record['src_ip'],
                               record['dst_ip'],
                               record['src_port'],
                               record['dst_port'],
                               record['threat_score'],
                               record['flow_id']
                           ))
            self.auth_db.commit()
            logger.info(f"威胁记录保存到数据库: {record}")
        except Exception as e:
            logger.error(f"保存威胁记录到数据库失败: {str(e)}")
            self.evaluator.log_signal.emit(f"错误: 保存威胁记录到数据库失败: {str(e)}")

    def closeEvent(self, event):
        self.evaluator.stop()
        self.auth_db.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    auth_system = UserAuth()
    main_dialog = MainDialog(auth_system)
    if main_dialog.exec() == QDialog.Accepted:
        dashboard = DashboardUI()
        dashboard.show()
        sys.exit(app.exec_())