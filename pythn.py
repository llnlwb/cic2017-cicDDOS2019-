import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import warnings
import re
import chardet

class Config:
    def __init__(self):
        # 数据路径
        self.data_paths = [
            # CIC-IDS2017数据集
            r"D:\毕业设计\数据预处理\raw_data\CIC2017\TrafficLabelling_\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            r"D:\毕业设计\数据预处理\raw_data\CIC2017\TrafficLabelling_\Friday-WorkingHours-Morning.pcap_ISCX.csv",
            r"D:\毕业设计\数据预处理\raw_data\CIC2017\TrafficLabelling_\Monday-WorkingHours.pcap_ISCX.csv",
            r"D:\毕业设计\数据预处理\raw_data\CIC2017\TrafficLabelling_\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            r"D:\毕业设计\数据预处理\raw_data\CIC2017\TrafficLabelling_\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            r"D:\毕业设计\数据预处理\raw_data\CIC2017\TrafficLabelling_\Tuesday-WorkingHours.pcap_ISCX.csv",
            r"D:\毕业设计\数据预处理\raw_data\CIC2017\TrafficLabelling_\Wednesday-workingHours.pcap_ISCX.csv",
            # CIC-DDoS2019数据集
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\BENIGN.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\DrDoS_DNS.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\DrDoS_LDAP.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\DrDoS_MSSQL.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\DrDoS_NetBIOS.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\DrDoS_NTP.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\DrDoS_SNMP.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\DrDoS_UDP.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\LDAP.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\MSSQL.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\NetBIOS.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\Portmap.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\Syn.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\TFTP.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\UDP.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\UDPLag.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\UDP-lag.csv",
            r"D:\毕业设计\数据预处理\raw_data\CICDDoS2019\class_split\WebDDoS.csv"
        ]

        # 输出目录
        self.output_path = r"D:\毕业设计\数据预处理\processed_data"
        
        # 验证路径
        self.validate_paths()

        # 攻击类型标签映射（严格按照您提供的格式）
        self.attack_type_mapping = {
            "BENIGN": int(0),
            "DoS Hulk": int(11),
            "DoS GoldenEye": int(12),
            "DoS slowloris": int(13),
            "DoS Slowhttptest": int(14),
            "DrDoS_NTP": int(21),
            "DrDoS_DNS": int(22),
            "DrDoS_UDP": int(23),
            "DrDoS_MSSQL": int(24),
            "DrDoS_NetBIOS": int(25),
            "DrDoS_LDAP": int(26),
            "DrDoS_SNMP": int(27),
            "PortScan": int(31),
            "Bot": int(41),
            "WebDDoS": int(51),
            "Heartbleed": int(61),
            "Syn": int(71),
            "TFTP": int(72),
            "UDP": int(73),
            "MSSQL": int(74),
            "NetBIOS": int(75),
            "LDAP": int(76),
            "Portmap": int(77),
            "SNMP": int(78),
            "UDPLag": int(81),
            "UDP-lag": int(81)
        }

        # 处理参数
        self.test_size = 0.2
        self.random_state = 42
        self.min_samples_per_class = 10
        self.categorical_cols = ["Protocol"]

    def validate_paths(self):
        """验证所有数据路径是否存在"""
        print("\n[初始化] 正在验证数据路径...")
        valid_paths = []
        
        for path in self.data_paths:
            if os.path.exists(path):
                valid_paths.append(path)
                print(f"  ✓ {os.path.basename(path)}")
            else:
                print(f"  ✗ 路径不存在: {path}")
        
        if not valid_paths:
            raise ValueError("没有找到任何有效数据文件！")
        
        self.data_paths = valid_paths
        print(f"验证完成，共找到 {len(valid_paths)} 个有效文件")


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        warnings.filterwarnings('ignore')

    def detect_encoding(self, file_path):
        """自动检测文件编码"""
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(100000))  # 检测前100KB
        return result['encoding']

    def load_single_file(self, path):
        """加载单个CSV文件"""
        try:
            # 检测文件编码
            encoding = self.detect_encoding(path)
            
            # 尝试读取文件（使用多种编码尝试）
            try:
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    engine='python',
                    on_bad_lines='warn',
                    dtype={'Protocol': 'str'},
                    true_values=['Yes', 'yes', 'TRUE', 'True'],
                    false_values=['No', 'no', 'FALSE', 'False']
                )
            except UnicodeDecodeError:
                df = pd.read_csv(
                    path,
                    encoding='ISO-8859-1',
                    engine='python',
                    on_bad_lines='warn',
                    dtype={'Protocol': 'str'},
                    true_values=['Yes', 'yes', 'TRUE', 'True'],
                    false_values=['No', 'no', 'FALSE', 'False']
                )
            
            # 自动检测标签列
            label_cols = [col for col in df.columns if 'label' in col.lower()]
            if not label_cols:
                print(f"警告: 文件 {os.path.basename(path)} 没有标签列，跳过")
                return None
                
            df = df.rename(columns={label_cols[0]: 'Label'})
            df = df.dropna(subset=['Label'])
            
            if len(df) == 0:
                print(f"警告: 文件 {os.path.basename(path)} 无有效数据")
                return None
                
            # 统一标签格式（保留原始大小写，仅去除前后空格）
            df['Label'] = df['Label'].str.strip()
            return df
            
        except Exception as e:
            print(f"加载文件 {os.path.basename(path)} 失败: {str(e)}")
            return None

    def load_data(self):
        """加载所有数据文件"""
        print("\n[1/6] 正在加载数据文件...")
        dfs = []
        
        for path in tqdm(self.config.data_paths, desc="加载进度"):
            df = self.load_single_file(path)
            if df is not None:
                dfs.append(df)
                print(f"已加载: {os.path.basename(path)} (样本数: {len(df):,})")
        
        if not dfs:
            raise ValueError("未加载到任何有效数据！")
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"\n总加载记录数: {len(combined):,}")
        print("标签类型分布:", combined['Label'].value_counts().to_dict())
        return combined

    def clean_data(self, df):
        """数据清洗"""
        print("\n[2/6] 正在清洗数据...")
        
        # 删除全零列
        zero_cols = df.columns[(df == 0).all()]
        if zero_cols.any():
            df = df.drop(zero_cols, axis=1)
            print(f"已删除全零列: {list(zero_cols)}")
        
        # 处理缺失值
        for col in tqdm(df.columns, desc="处理列"):
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif df[col].dtype == 'object':
                df[col] = df[col].fillna('unknown')
        
        # 处理无穷值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.ffill().bfill()
        
        return df

    def process_features(self, df):
        """特征工程"""
        print("\n[3/6] 正在处理特征...")
        
        # 协议编码
        if 'Protocol' in df.columns:
            le = LabelEncoder()
            df['Protocol'] = le.fit_transform(df['Protocol'].astype(str))
        
        # 数值归一化
        numeric_cols = df.select_dtypes(include=np.number).columns
        numeric_cols = [col for col in numeric_cols if col != 'Label']
        
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return df

    def process_labels(self, df):
        """标签处理"""
        print("\n[4/6] 正在处理标签...")
        
        # 打印原始标签分布
        print("原始标签分布:", df['Label'].value_counts().to_dict())
        
        # 应用标签映射（保留原始标签大小写）
        df['Label'] = df['Label'].map(self.config.attack_type_mapping)
        
        # 统计未知标签
        unknown_labels = df[df['Label'].isna()]['Label'].value_counts()
        if not unknown_labels.empty:
            print("警告: 发现未映射的标签:", unknown_labels.index.tolist())
        
        # 未知标签设为99
        df['Label'] = df['Label'].fillna(99).astype(int)
        
        # 过滤稀有类别
        label_counts = df['Label'].value_counts()
        valid_labels = label_counts[label_counts >= self.config.min_samples_per_class].index
        filtered_df = df[df['Label'].isin(valid_labels)]
        
        print("处理后标签分布:", filtered_df['Label'].value_counts().to_dict())
        return filtered_df

    def split_and_save(self, df):
        """数据划分与保存"""
        print("\n[5/6] 正在保存结果...")
        
        # 数据划分
        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            stratify=df['Label'],
            random_state=self.config.random_state
        )
        
        # 创建输出目录
        os.makedirs(self.config.output_path, exist_ok=True)
        
        # 保存CSV
        train_path = os.path.join(self.config.output_path, "train_data.csv")
        test_path = os.path.join(self.config.output_path, "test_data.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # 保存元数据（确保所有数值转换为Python原生类型）
        metadata = {
            "feature_columns": list(train_df.columns),
            "label_mapping": {k: int(v) for k, v in self.config.attack_type_mapping.items()},
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "label_distribution": {
                "train": {int(k): int(v) for k, v in train_df['Label'].value_counts().items()},
                "test": {int(k): int(v) for k, v in test_df['Label'].value_counts().items()}
            }
        }
        
        with open(os.path.join(self.config.output_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n[完成] 数据处理成功！")
        print(f"训练集: {train_path} (样本数: {len(train_df):,})")
        print(f"测试集: {test_path} (样本数: {len(test_df):,})")

    def run_pipeline(self):
        """执行完整流程"""
        try:
            # 1. 数据加载
            raw_data = self.load_data()
            
            # 2. 数据清洗
            cleaned_data = self.clean_data(raw_data)
            
            # 3. 特征工程
            processed_data = self.process_features(cleaned_data)
            
            # 4. 标签处理
            labeled_data = self.process_labels(processed_data)
            
            # 5. 保存结果
            self.split_and_save(labeled_data)
            return True
            
        except Exception as e:
            print(f"\n[错误] 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    try:
        print("="*50)
        print("网络流量数据预处理管道")
        print("="*50)
        
        config = Config()
        preprocessor = DataPreprocessor(config)
        
        if preprocessor.run_pipeline():
            print("\n✔ 处理完成！结果保存在:", config.output_path)
        else:
            print("\n✗ 处理过程中出现错误！")
            
    except Exception as e:
        print(f"\n[严重错误] 程序初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()