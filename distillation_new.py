import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
import psutil
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ================== 配置参数 ==================
# 数据路径配置
RAW_TRAIN_PATH = r"D:\bysj\server\passdata\train_data.csv"  # 训练数据路径
RAW_TEST_PATH = r"D:\bysj\server\passdata\test_data.csv"   # 测试数据路径
SAVE_DIR = r"D:\毕业设计\模型蒸馏"                        # 模型保存目录
TEACHER_MODEL_PATH = r"D:\毕业设计\模拟训练\best_model.pth"  # 教师模型路径

# 数据读取范围
DATA_CONFIG = {
    'train': {'start': 500000, 'end': 600000},  # 训练数据行范围
    'test': {'start': 50000, 'end': 65000}      # 测试数据行范围
}

# 模型训练参数
MODEL_CONFIG = {
    'num_classes': 9,        # 9大类
    'seq_length': 50,        # 50维特征
    'batch_size': 64,
    'epochs': 200,
    'init_lr': 1e-4,
    'weight_decay': 1e-5,
    'log_interval': 50,
    'val_interval': 2000,
    'test_interval': 1,
    'threshold': 0.5
}

# 蒸馏参数
DISTILL_CONFIG = {
    'temperature': 3.0,       # 软标签温度系数
    'alpha': 0.5,             # 蒸馏损失权重
    'teacher_d_model': 128,   # 教师模型维度
    'student_d_model': 64,    # 学生模型维度
    'student_layers': 2       # 学生模型Transformer层数
}

# 训练控制参数
MAX_TRAIN_HOURS = 10
PATIENCE = 5
TARGET_ACC = 0.98
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ================== 数据加载模块 ==================
class CustomDataLoader:
    @staticmethod
    def load_data(file_path, start, end):
        """加载CSV数据，读取前50列作为特征，第51列作为标签，并过滤unknown标签"""
        try:
            # 读取指定行范围
            df = pd.read_csv(file_path, skiprows=range(1, start+1), nrows=end-start)
            print(f"\n[加载] 文件: {os.path.basename(file_path)} | 原始样本数: {len(df):,}")
            
            # 检查列数是否足够（至少51列：50特征+1标签）
            if df.shape[1] < 51:
                raise ValueError(f"数据文件列数不足: 预期至少51列（50特征+1标签），实际{df.shape[1]}列")
            
            # 查找或指定Label列
            if 'Label' not in df.columns:
                # 假设第51列是标签（列索引50）
                if df.shape[1] < 51:
                    raise ValueError("数据文件中无法找到Label列，且列数不足51")
                df['Label'] = df.iloc[:, 50]
            
            # 过滤unknown标签（值为99）
            initial_count = len(df)
            df = df[df['Label'] != 99]
            filtered_count = len(df)
            print(f"[标签过滤] 移除unknown标签（99）: {initial_count - filtered_count:,} 条")
            print("过滤后标签分布:", df['Label'].value_counts().to_dict())
            
            # 提取特征和标签
            X = df.iloc[:, :50].values.astype(np.float32)
            y = df['Label'].values.astype(np.int64)
            
            # 确保特征维度为50
            if X.shape[1] != MODEL_CONFIG['seq_length']:
                raise ValueError(f"特征维度错误: 预期 {MODEL_CONFIG['seq_length']}，实际 {X.shape[1]}")
            
            # 确保标签在0-8范围内
            if y.min() < 0 or y.max() > 8:
                raise ValueError(f"标签越界: 预期0-8，实际范围[{y.min()},{y.max()}]")
            
            print(f"[特征选择] 保留特征数: {X.shape[1]}")
            return X, y
        except Exception as e:
            print(f"\033[31m数据加载失败: {str(e)}\033[0m")
            exit(1)

# ================== 教师模型 ==================
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = DISTILL_CONFIG['teacher_d_model']
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.GELU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=8, dim_feedforward=512,
                dropout=0.1, batch_first=True, activation='gelu'
            ),
            num_layers=6
        )
        self.output = nn.Sequential(
            nn.Linear(MODEL_CONFIG['seq_length'] * self.d_model, 512),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, MODEL_CONFIG['num_classes'])
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)
        return self.output(x)

# ================== 学生模型 ==================
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = DISTILL_CONFIG['student_d_model']
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.GELU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=4, dim_feedforward=256,
                dropout=0.1, batch_first=True, activation='gelu'
            ),
            num_layers=DISTILL_CONFIG['student_layers']
        )
        self.output = nn.Sequential(
            nn.Linear(MODEL_CONFIG['seq_length'] * self.d_model, 256),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, MODEL_CONFIG['num_classes'])
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)
        return self.output(x)

# ================== 蒸馏损失函数 ==================
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.hard_criterion = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        hard_loss = self.hard_criterion(student_logits, targets.squeeze().long())
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits.detach() / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        return total_loss, hard_loss, soft_loss

# ================== 评估指标 ==================
def calculate_accuracy(y_logits, y_true):
    y_pred = torch.argmax(y_logits, dim=1)
    correct = (y_pred == y_true).float()
    return torch.mean(correct).item()

# ================== 模型大小计算 ==================
def get_model_size(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    return (param_size + buffer_size) / 1024**2

# ================== 推理速度测试 ==================
def measure_inference_time(model, input_tensor, num_iterations=100):
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # 预热
            _ = model(input_tensor)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    avg_time = (time.time() - start_time) / num_iterations * 1000
    return avg_time

# ================== 蒸馏训练流程 ==================
def main():
    process = psutil.Process(os.getpid())
    training_state = {
        'best_acc': 0.0,
        'no_improve_epochs': 0,
        'start_time': time.time(),
        'total_timeout': MAX_TRAIN_HOURS * 3600,
        'best_model_state': None
    }

    # 加载数据
    print("\n\033[34m=== 数据加载阶段 ===")
    X_train, y_train = CustomDataLoader.load_data(
        RAW_TRAIN_PATH,
        DATA_CONFIG['train']['start'],
        DATA_CONFIG['train']['end']
    )
    X_test, y_test = CustomDataLoader.load_data(
        RAW_TEST_PATH,
        DATA_CONFIG['test']['start'],
        DATA_CONFIG['test']['end']
    )
    
    # 拆分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, shuffle=True)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 转换为张量
    X_train = torch.FloatTensor(X_train).reshape(-1, MODEL_CONFIG['seq_length'], 1)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val).reshape(-1, MODEL_CONFIG['seq_length'], 1)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test).reshape(-1, MODEL_CONFIG['seq_length'], 1)
    y_test = torch.LongTensor(y_test)
    
    # 创建DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=4
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=MODEL_CONFIG['batch_size']*2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=2
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=MODEL_CONFIG['batch_size']*2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=2
    )

    # 加载教师模型
    print("\n\033[34m=== 加载教师模型 ===")
    teacher_model = TeacherModel().to(DEVICE)
    try:
        teacher_model.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE))
        print(f"成功加载教师模型: {TEACHER_MODEL_PATH}")
    except Exception as e:
        print(f"\033[31m加载教师模型失败: {str(e)}\033[0m")
        exit(1)
    
    teacher_model.eval()
    student_model = StudentModel().to(DEVICE)
    
    # 计算模型大小
    teacher_size = get_model_size(teacher_model)
    student_size = get_model_size(student_model)
    print(f"教师模型大小: {teacher_size:.2f} MB")
    print(f"学生模型大小: {student_size:.2f} MB")
    print(f"模型大小减少: {(1 - student_size/teacher_size)*100:.1f}%")
    
    # 测量推理速度
    sample_input = X_test[:32].to(DEVICE)
    teacher_time = measure_inference_time(teacher_model, sample_input)
    student_time = measure_inference_time(student_model, sample_input)
    print(f"教师模型推理时间: {teacher_time:.2f} ms/batch")
    print(f"学生模型推理时间: {student_time:.2f} ms/batch")
    print(f"推理速度提升: {(teacher_time/student_time - 1)*100:.1f}%")
    
    # 初始化优化器和调度器
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=MODEL_CONFIG['init_lr'],
        weight_decay=MODEL_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MODEL_CONFIG['init_lr']*10,
        total_steps=MODEL_CONFIG['epochs'] * len(train_loader)
    )
    distill_criterion = DistillationLoss(
        alpha=DISTILL_CONFIG['alpha'],
        temperature=DISTILL_CONFIG['temperature']
    )
    
    # 训练记录
    metrics = {
        'epoch': [],
        'train_loss': [],
        'hard_loss': [],
        'soft_loss': [],
        'val_loss': [],
        'test_acc': [],
        'val_acc': [],
        'learning_rate': [],
        'memory_usage': [],
        'time_per_batch': []
    }

    print("\n\033[35m=== 开始模型蒸馏训练 ===")
    print(f"设备类型: {DEVICE}")
    print(f"训练样本数: {len(X_train):,}")
    print(f"验证样本数: {len(X_val):,}")
    print(f"测试样本数: {len(X_test):,}")
    print(f"蒸馏温度: {DISTILL_CONFIG['temperature']}")
    print(f"蒸馏权重: {DISTILL_CONFIG['alpha']}")
    print(f"最大训练时长: {MAX_TRAIN_HOURS} 小时")
    print(f"早停耐心值: {PATIENCE} 轮次")
    print(f"目标准确率: {TARGET_ACC*100}%\n")

    global_start = time.time()
    for epoch in range(MODEL_CONFIG['epochs']):
        elapsed_time = time.time() - training_state['start_time']
        if elapsed_time > training_state['total_timeout']:
            print(f"\n\033[31m训练已进行 {elapsed_time/3600:.1f} 小时，达到最大时长限制\033[0m")
            break

        epoch_start = time.time()
        student_model.train()
        epoch_train_loss = 0.0
        epoch_hard_loss = 0.0
        epoch_soft_loss = 0.0
        batch_times = []
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_start = time.time()
            
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            
            with torch.no_grad():
                teacher_preds = teacher_model(batch_X)
            
            optimizer.zero_grad()
            student_preds = student_model(batch_X)
            loss, hard_loss, soft_loss = distill_criterion(
                student_preds, teacher_preds, batch_y
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            epoch_hard_loss += hard_loss.item()
            epoch_soft_loss += soft_loss.item()
            metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if (batch_idx+1) % MODEL_CONFIG['val_interval'] == 0:
                student_model.eval()
                val_loss, val_acc = 0.0, 0.0
                with torch.no_grad():
                    for val_X, val_y in val_loader:
                        val_X = val_X.to(DEVICE)
                        val_y = val_y.to(DEVICE)
                        val_preds = student_model(val_X)
                        val_loss += F.cross_entropy(val_preds, val_y).item()
                        val_acc += calculate_accuracy(val_preds, val_y)
                
                avg_val_loss = val_loss / len(val_loader)
                avg_val_acc = val_acc / len(val_loader)
                metrics['val_acc'].append(avg_val_acc)
                metrics['val_loss'].append(avg_val_loss)
                
                print(f"\n[验证] 批次 {batch_idx+1}/{len(train_loader)} | "
                      f"验证损失: {avg_val_loss:.4f} | "
                      f"验证准确率: {avg_val_acc*100:.1f}%")
                student_model.train()

            if (batch_idx+1) % MODEL_CONFIG['log_interval'] == 0:
                mem_usage = process.memory_info().rss / 1024**2
                remaining_time = training_state['total_timeout'] - elapsed_time
                print(f"轮次 [{epoch+1:02d}/{MODEL_CONFIG['epochs']}] | "
                      f"批次 [{batch_idx+1:04d}/{len(train_loader):04d}] | "
                      f"耗时 {batch_time:.2f}s | "
                      f"总损失 {loss.item():.4f} | "
                      f"硬损失 {hard_loss.item():.4f} | "
                      f"软损失 {soft_loss.item():.4f} | "
                      f"学习率 {metrics['learning_rate'][-1]:.2e} | "
                      f"内存 {mem_usage:.1f}MB | "
                      f"剩余时间 {remaining_time//3600:.0f}h {remaining_time%3600//60:.0f}m")

        student_model.eval()
        teacher_model.eval()
        student_test_acc = 0.0
        teacher_test_acc = 0.0
        
        with torch.no_grad():
            for test_X, test_y in test_loader:
                test_X = test_X.to(DEVICE)
                test_y = test_y.to(DEVICE)
                
                student_preds = student_model(test_X)
                teacher_preds = teacher_model(test_X)
                
                student_test_acc += calculate_accuracy(student_preds, test_y)
                teacher_test_acc += calculate_accuracy(teacher_preds, test_y)
                
        avg_student_test_acc = student_test_acc / len(test_loader)
        avg_teacher_test_acc = teacher_test_acc / len(test_loader)
        
        if avg_student_test_acc > training_state['best_acc']:
            training_state['best_acc'] = avg_student_test_acc
            training_state['best_model_state'] = copy.deepcopy(student_model.state_dict())
            training_state['no_improve_epochs'] = 0
            print(f"\n\033[32m[更新最佳模型] 学生模型测试准确率: {avg_student_test_acc*100:.2f}%\033[0m")
        else:
            training_state['no_improve_epochs'] += 1

        stop_training = False
        if avg_student_test_acc >= TARGET_ACC:
            print(f"\n\033[32m达到目标准确率 {TARGET_ACC*100}%，停止训练\033[0m")
            stop_training = True
        elif training_state['no_improve_epochs'] >= PATIENCE:
            print(f"\n\033[31m连续{PATIENCE}轮无提升，提前停止训练\033[0m")
            stop_training = True

        metrics['epoch'].append(epoch+1)
        metrics['train_loss'].append(epoch_train_loss/len(train_loader))
        metrics['hard_loss'].append(epoch_hard_loss/len(train_loader))
        metrics['soft_loss'].append(epoch_soft_loss/len(train_loader))
        metrics['test_acc'].append(avg_student_test_acc)
        metrics['memory_usage'].append(process.memory_info().rss/1024**2)
        metrics['time_per_batch'].append(np.mean(batch_times))

        epoch_time = time.time() - epoch_start
        print(f"\n\033[33m=== 轮次 {epoch+1:02d} 总结 ===")
        print(f"教师模型测试准确率: {avg_teacher_test_acc*100:.2f}%")
        print(f"学生模型测试准确率: {avg_student_test_acc*100:.2f}% | 最佳准确率: {training_state['best_acc']*100:.2f}%")
        print(f"准确率差距: {(avg_teacher_test_acc - avg_student_test_acc)*100:.2f}%")
        print(f"平均训练损失: {metrics['train_loss'][-1]:.4f}")
        print(f"平均硬目标损失: {metrics['hard_loss'][-1]:.4f}")
        print(f"平均软目标损失: {metrics['soft_loss'][-1]:.4f}")
        print(f"平均批次耗时: {metrics['time_per_batch'][-1]:.2f}s")
        print(f"内存使用峰值: {metrics['memory_usage'][-1]:.1f}MB")
        print(f"本轮耗时: {epoch_time:.1f}s\033[0m\n")

        if stop_training:
            break

    if training_state['best_model_state'] is not None:
        best_model_path = os.path.join(SAVE_DIR, "best_student_model.pth")
        torch.save(training_state['best_model_state'], best_model_path)
        print(f"\n\033[36m最佳学生模型已保存至: {best_model_path}")
        print(f"测试集最高准确率: {training_state['best_acc']*100:.2f}%\033[0m")

    final_model_path = os.path.join(SAVE_DIR, "final_student_model.pth")
    torch.save(student_model.state_dict(), final_model_path)
    print(f"最终学生模型已保存至: {final_model_path}")

    print("\n\033[36m=== 最终性能评估 ===\033[0m")
    best_student = StudentModel().to(DEVICE)
    best_student.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    best_student.eval()
    
    teacher_model.eval()
    teacher_test_acc = 0.0
    student_test_acc = 0.0
    
    with torch.no_grad():
        for test_X, test_y in test_loader:
            test_X = test_X.to(DEVICE)
            test_y = test_y.to(DEVICE)
            teacher_preds = teacher_model(test_X)
            student_preds = best_student(test_X)
            teacher_test_acc += calculate_accuracy(teacher_preds, test_y)
            student_test_acc += calculate_accuracy(student_preds, test_y)
    
    avg_teacher_test_acc = teacher_test_acc / len(test_loader)
    avg_student_test_acc = student_test_acc / len(test_loader)
    
    print(f"教师模型测试准确率: {avg_teacher_test_acc*100:.2f}%")
    print(f"学生模型测试准确率: {avg_student_test_acc*100:.2f}%")
    print(f"准确率差距: {(avg_teacher_test_acc - avg_student_test_acc)*100:.2f}%")
    print(f"模型大小减少: {(1 - student_size/teacher_size)*100:.1f}%")
    
    teacher_time = measure_inference_time(teacher_model, sample_input)
    student_time = measure_inference_time(best_student, sample_input)
    print(f"教师模型推理时间: {teacher_time:.2f} ms/batch")
    print(f"学生模型推理时间: {student_time:.2f} ms/batch")
    print(f"推理速度提升: {(teacher_time/student_time - 1)*100:.1f}%")
    
    plt.figure(figsize=(18, 12))
    plt.subplot(2,3,1)
    plt.plot(metrics['epoch'], metrics['train_loss'], 'b-o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(2,3,2)
    plt.plot(metrics['epoch'], metrics['hard_loss'], 'r-s', label='Hard Loss')
    plt.plot(metrics['epoch'], metrics['soft_loss'], 'g-^', label='Soft Loss')
    plt.title('Hard vs Soft Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,3,3)
    plt.plot(metrics['epoch'], metrics['test_acc'], 'g-s')
    plt.axhline(y=TARGET_ACC, color='r', linestyle='--')
    plt.axhline(y=avg_teacher_test_acc, color='b', linestyle='--', label='Teacher Acc')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,3,4)
    plt.plot(metrics['learning_rate'], 'm-')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate (log)')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(2,3,5)
    plt.plot(metrics['epoch'], metrics['memory_usage'], 'c-d')
    plt.title('Memory Usage')
    plt.xlabel('Epoch')
    plt.ylabel('Memory (MB)')
    plt.grid(True)
    
    plt.subplot(2,3,6)
    plt.plot(metrics['epoch'], metrics['time_per_batch'], 'y-o')
    plt.title('Batch Processing Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()
    report_path = os.path.join(SAVE_DIR, "distillation_report.png")
    plt.savefig(report_path, dpi=150)
    print(f"\n蒸馏训练报告已保存至: {report_path}")

    total_time = time.time() - global_start
    print(f"\n\033[36m总训练时间: {total_time//3600:.0f}h {total_time%3600//60:.0f}m {total_time%60:.0f}s\033[0m")

if __name__ == "__main__":
    print("\033[33m=== 系统初始化检查 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"可用设备: {DEVICE}")
    print(f"CPU 核心数: {os.cpu_count()}")
    print(f"初始内存使用: {psutil.Process(os.getpid()).memory_info().rss/1024**2:.1f}MB")
    main()