import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from tqdm import tqdm
import json
import warnings
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
        self.output_path = r"D:\bysj\server\passdata"
        
        # 验证路径
        self.validate_paths()

        # 攻击类型标签映射（9大类，基于首位数字）
        self.attack_type_mapping = {
            "BENIGN": 0,
            "DoS Hulk": 1,
            "DoS GoldenEye": 1,
            "DoS slowloris": 1,
            "DoS Slowhttptest": 1,
            "DrDoS_NTP": 2,
            "DrDoS_DNS": 2,
            "DrDoS_UDP": 2,
            "DrDoS_MSSQL": 2,
            "DrDoS_NetBIOS": 2,
            "DrDoS_LDAP": 2,
            "DrDoS_SNMP": 2,
            "PortScan": 3,
            "Bot": 4,
            "WebDDoS": 5,
            "Heartbleed": 6,
            "Syn": 7,
            "TFTP": 7,
            "UDP": 7,
            "MSSQL": 7,
            "NetBIOS": 7,
            "LDAP": 7,
            "Portmap": 7,
            "UDPLag": 8,
            "UDP-lag": 8
        }

        # 处理参数
        self.test_size = 0.15
        self.random_state = 42
        self.min_samples_per_class = 10
        self.categorical_cols = ["Protocol"]
        self.variance_threshold = 0.01  # 方差阈值
        self.top_k_features = 50  # 初始选择top 50个特征，可根据数据调整

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
            result = chardet.detect(f.read(100000))
        return result['encoding']

    def load_single_file(self, path):
        """加载单个CSV文件"""
        try:
            encoding = self.detect_encoding(path)
            try:
                df = pd.read_csv(
                    path, encoding=encoding, engine='python', on_bad_lines='warn',
                    dtype={'Protocol': 'str'}, true_values=['Yes', 'yes', 'TRUE', 'True'],
                    false_values=['No', 'no', 'FALSE', 'False']
                )
            except UnicodeDecodeError:
                df = pd.read_csv(
                    path, encoding='ISO-8859-1', engine='python', on_bad_lines='warn',
                    dtype={'Protocol': 'str'}, true_values=['Yes', 'yes', 'TRUE', 'True'],
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

    def select_discriminative_features(self, df):
        """选择能够明显体现差异的特征"""
        print("\n[3/6] 正在选择高区分度特征...")
        
        # 提取特征和标签
        X = df.drop(columns=['Label'])
        y = df['Label'].map(self.config.attack_type_mapping).fillna(99).astype(int)
        
        # 删除全零列和低方差列
        numeric_cols = X.select_dtypes(include=np.number).columns
        variances = X[numeric_cols].var()
        low_variance_cols = variances[variances < self.config.variance_threshold].index
        X = X.drop(columns=low_variance_cols)
        print(f"已删除低方差列（方差<{self.config.variance_threshold}）: {list(low_variance_cols)}")
        
        # 使用ANOVA F值选择top特征
        numeric_cols = X.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            selector = SelectKBest(score_func=f_classif, k=min(self.config.top_k_features, len(numeric_cols)))
            selector.fit(X[numeric_cols], y)
            scores = pd.Series(selector.scores_, index=numeric_cols)
            selected_cols = scores.sort_values(ascending=False).index[:self.config.top_k_features].tolist()
            print(f"选择的top特征（基于ANOVA F值）: {selected_cols}")
        else:
            selected_cols = []
        
        # 确保包含Protocol（如果存在）
        if 'Protocol' in X.columns and 'Protocol' not in selected_cols:
            selected_cols.append('Protocol')
        
        # 构造最终数据框
        df_selected = df[selected_cols + ['Label']]
        print(f"最终保留特征数: {len(selected_cols)}")
        
        return df_selected

    def process_features(self, df):
        """特征工程"""
        print("\n[4/6] 正在处理特征...")
        
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
        """标签处理（9大类）"""
        print("\n[5/6] 正在处理标签...")
        print("原始标签分布:", df['Label'].value_counts().to_dict())
        
        # 应用标签映射
        df['Label'] = df['Label'].map(self.config.attack_type_mapping)
        
        # 处理未知标签
        unknown_labels = df[df['Label'].isna()]['Label'].value_counts()
        if not unknown_labels.empty:
            print("警告: 发现未映射的标签:", unknown_labels.index.tolist())
        df['Label'] = df['Label'].fillna(99).astype(int)
        
        # 过滤稀有类别
        label_counts = df['Label'].value_counts()
        valid_labels = label_counts[label_counts >= self.config.min_samples_per_class].index
        filtered_df = df[df['Label'].isin(valid_labels)]
        
        print("处理后标签分布:", filtered_df['Label'].value_counts().to_dict())
        return filtered_df

    def split_and_save(self, df):
        """数据划分与保存"""
        print("\n[6/6] 正在保存结果...")
        train_df, test_df = train_test_split(
            df, test_size=self.config.test_size, stratify=df['Label'],
            random_state=self.config.random_state
        )
        
        os.makedirs(self.config.output_path, exist_ok=True)
        train_path = os.path.join(self.config.output_path, "train_data.csv")
        test_path = os.path.join(self.config.output_path, "test_data.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # 计算特征统计
        numeric_cols = train_df.select_dtypes(include=np.number).columns
        numeric_cols = [col for col in numeric_cols if col != 'Label']
        stats = {col: {"max": float(train_df[col].max()), "min": float(train_df[col].min())} for col in numeric_cols}
        
        # 保存元数据
        metadata = {
            "feature_columns": list(train_df.columns),
            "label_mapping": {k: int(v) for k, v in self.config.attack_type_mapping.items()},
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "feature_statistics": stats,
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
        print("元数据已保存至: metadata.json")

    def run_pipeline(self):
        """执行完整流程"""
        try:
            raw_data = self.load_data()
            cleaned_data = self.clean_data(raw_data)
            selected_data = self.select_discriminative_features(cleaned_data)
            processed_data = self.process_features(selected_data)
            labeled_data = self.process_labels(processed_data)
            self.split_and_save(labeled_data)
            return True
        except Exception as e:
            print(f"\n[错误] 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("="*50)
    print("网络流量数据预处理系统 - Transformer专用版")
    print("="*50)
    try:
        config = Config()
        processor = DataPreprocessor(config)
        if processor.run_pipeline():
            print("\n✅ 处理完成！输出目录:", config.output_path)
        else:
            print("\n❌ 处理过程中发生错误")
    except Exception as e:
        print(f"\n🔥 系统初始化失败: {str(e)}")

if __name__ == "__main__":
    main()