import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

# ================== 配置参数 ==================
# 数据路径配置
RAW_TRAIN_PATH = r"D:\bysj\server\passdata\train_data.csv"  # 训练数据路径
RAW_TEST_PATH = r"D:\bysj\server\passdata\test_data.csv"   # 测试数据路径
SAVE_DIR = r"D:\毕业设计\模拟训练"                        # 模型保存目录

# 数据读取范围
DATA_CONFIG = {
    'train': {'start': 500000, 'end': 1000000},  # 训练数据行范围
    'test': {'start': 50000, 'end': 85000}      # 测试数据行范围
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

# 训练控制参数
MAX_TRAIN_HOURS = 10
PATIENCE = 5
TARGET_ACC = 0.995
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
            
            print(f"[特征选择] 保留特征数: {X.shape[1]}")
            return X, y
        except Exception as e:
            print(f"\033[31m数据加载失败: {str(e)}\033[0m")
            exit(1)

# ================== CNN-Transformer模型 ==================
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 128
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

# ================== 评估指标 ==================
def calculate_accuracy(y_logits, y_true):
    y_pred = torch.argmax(y_logits, dim=1)
    correct = (y_pred == y_true).float()
    return torch.mean(correct).item()

# ================== 训练流程 ==================
def main():
    process = psutil.Process(os.getpid())
    
    # 初始化训练状态
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
    train_loader = TorchDataLoader(
        TensorDataset(X_train, y_train),
        batch_size=MODEL_CONFIG['batch_size'], shuffle=True,
        pin_memory=torch.cuda.is_available(), num_workers=4
    )
    val_loader = TorchDataLoader(
        TensorDataset(X_val, y_val),
        batch_size=MODEL_CONFIG['batch_size']*2, shuffle=False,
        pin_memory=torch.cuda.is_available(), num_workers=2
    )
    test_loader = TorchDataLoader(
        TensorDataset(X_test, y_test),
        batch_size=MODEL_CONFIG['batch_size']*2, shuffle=False,
        pin_memory=torch.cuda.is_available(), num_workers=2
    )

    # 初始化模型
    model = TransformerModel().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MODEL_CONFIG['init_lr'],
        weight_decay=MODEL_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MODEL_CONFIG['init_lr']*10,
        total_steps=MODEL_CONFIG['epochs'] * len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    
    # 训练记录
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'test_acc': [],
        'val_acc': [],
        'learning_rate': [],
        'memory_usage': [],
        'time_per_batch': []
    }

    print("\n\033[35m=== 开始模型训练 ===")
    print(f"设备类型: {DEVICE}")
    print(f"训练样本数: {len(X_train):,}")
    print(f"验证样本数: {len(X_val):,}")
    print(f"测试样本数: {len(X_test):,}")
    print(f"最大训练时长: {MAX_TRAIN_HOURS} 小时")
    print(f"早停耐心值: {PATIENCE} 轮次")
    print(f"目标准确率: {TARGET_ACC*100}%\n")

    # 训练循环
    global_start = time.time()
    for epoch in range(MODEL_CONFIG['epochs']):
        elapsed_time = time.time() - training_state['start_time']
        if elapsed_time > training_state['total_timeout']:
            print(f"\n\033[31m训练已进行 {elapsed_time/3600:.1f} 小时，达到最大时长限制\033[0m")
            break

        epoch_start = time.time()
        model.train()
        epoch_train_loss = 0.0
        batch_times = []
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_start = time.time()
            
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if (batch_idx+1) % MODEL_CONFIG['val_interval'] == 0:
                model.eval()
                val_loss, val_acc = 0.0, 0.0
                with torch.no_grad():
                    for val_X, val_y in val_loader:
                        val_X = val_X.to(DEVICE)
                        val_y = val_y.to(DEVICE)
                        val_preds = model(val_X)
                        val_loss += criterion(val_preds, val_y).item()
                        val_acc += calculate_accuracy(val_preds, val_y)
                
                avg_val_loss = val_loss / len(val_loader)
                avg_val_acc = val_acc / len(val_loader)
                metrics['val_acc'].append(avg_val_acc)
                metrics['val_loss'].append(avg_val_loss)
                
                print(f"\n[验证] 批次 {batch_idx+1}/{len(train_loader)} | "
                      f"验证损失: {avg_val_loss:.4f} | "
                      f"验证准确率: {avg_val_acc*100:.1f}%")
                model.train()

            if (batch_idx+1) % MODEL_CONFIG['log_interval'] == 0:
                mem_usage = process.memory_info().rss / 1024**2
                remaining_time = training_state['total_timeout'] - elapsed_time
                print(f"轮次 [{epoch+1:02d}/{MODEL_CONFIG['epochs']}] | "
                      f"批次 [{batch_idx+1:04d}/{len(train_loader):04d}] | "
                      f"耗时 {batch_time:.2f}s | "
                      f"损失 {loss.item():.4f} | "
                      f"学习率 {metrics['learning_rate'][-1]:.2e} | "
                      f"内存 {mem_usage:.1f}MB | "
                      f"剩余时间 {remaining_time//3600:.0f}h {remaining_time%3600//60:.0f}m")

        model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for test_X, test_y in test_loader:
                test_X = test_X.to(DEVICE)
                test_y = test_y.to(DEVICE)
                test_preds = model(test_X)
                test_acc += calculate_accuracy(test_preds, test_y)
        avg_test_acc = test_acc / len(test_loader)
        
        if avg_test_acc > training_state['best_acc']:
            training_state['best_acc'] = avg_test_acc
            training_state['best_model_state'] = copy.deepcopy(model.state_dict())
            training_state['no_improve_epochs'] = 0
            print(f"\n\033[32m[更新最佳模型] 测试准确率: {avg_test_acc*100:.2f}%\033[0m")
        else:
            training_state['no_improve_epochs'] += 1

        stop_training = False
        if avg_test_acc >= TARGET_ACC:
            print(f"\n\033[32m达到目标准确率 {TARGET_ACC*100}%，停止训练\033[0m")
            stop_training = True
        elif training_state['no_improve_epochs'] >= PATIENCE:
            print(f"\n\033[31m连续{PATIENCE}轮无提升，提前停止训练\033[0m")
            stop_training = True

        metrics['epoch'].append(epoch+1)
        metrics['train_loss'].append(epoch_train_loss/len(train_loader))
        metrics['test_acc'].append(avg_test_acc)
        metrics['memory_usage'].append(process.memory_info().rss/1024**2)
        metrics['time_per_batch'].append(np.mean(batch_times))

        epoch_time = time.time() - epoch_start
        print(f"\n\033[33m=== 轮次 {epoch+1:02d} 总结 ===")
        print(f"测试准确率: {avg_test_acc*100:.2f}% | 最佳准确率: {training_state['best_acc']*100:.2f}%")
        print(f"平均训练损失: {metrics['train_loss'][-1]:.4f}")
        print(f"平均批次耗时: {metrics['time_per_batch'][-1]:.2f}s")
        print(f"内存使用峰值: {metrics['memory_usage'][-1]:.1f}MB")
        print(f"本轮耗时: {epoch_time:.1f}s\033[0m\n")

        if stop_training:
            break

    if training_state['best_model_state'] is not None:
        best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
        torch.save(training_state['best_model_state'], best_model_path)
        print(f"\n\033[36m最佳模型已保存至: {best_model_path}")
        print(f"测试集最高准确率: {training_state['best_acc']*100:.2f}%\033[0m")

    final_model_path = os.path.join(SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存至: {final_model_path}")

    plt.figure(figsize=(18, 12))
    plt.subplot(2,2,1)
    plt.plot(metrics['epoch'], metrics['train_loss'], 'b-o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(2,2,2)
    plt.plot(metrics['epoch'], metrics['test_acc'], 'g-s')
    plt.axhline(y=0.95, color='b', linestyle='--', label='Baseline')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(2,2,3)
    plt.plot(metrics['learning_rate'], 'm-')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate (log)')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(2,2,4)
    plt.plot(metrics['epoch'], metrics['memory_usage'], 'c-d')
    plt.title('Memory Usage')
    plt.xlabel('Epoch')
    plt.ylabel('Memory (MB)')
    plt.grid(True)
    
    plt.tight_layout()
    report_path = os.path.join(SAVE_DIR, "training_report.png")
    plt.savefig(report_path, dpi=150)
    print(f"\n训练报告已保存至: {report_path}")

    total_time = time.time() - global_start
    print(f"\n\033[36m总训练时间: {total_time//3600:.0f}h {total_time%3600//60:.0f}m\033[0m")

if __name__ == "__main__":
    print("\033[33m=== 系统初始化检查 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"可用设备: {DEVICE}")
    print(f"CPU 核心数: {os.cpu_count()}")
    print(f"初始内存使用: {psutil.Process(os.getpid()).memory_info().rss/1024**2:.1f}MB")
    main()