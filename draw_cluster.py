import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib as mpl
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入数据集
from data.IRDataSet import IRDataSet
from data.ESPSet import ESPSet
from data.QM9Set import QM9IRDataSet

# 设置科研风格
plt.style.use('default')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
})

def create_smiles_vocabulary(smiles_list):
    """创建SMILES字符词汇表"""
    chars = set()
    for smi in smiles_list:
        chars.update(smi)
    
    # 添加特殊字符
    chars.add('<PAD>')  # 填充符
    chars.add('<UNK>')  # 未知字符
    
    # 创建字符到索引的映射
    char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return char_to_idx, idx_to_char

def smiles_to_onehot(smiles_list, max_length=150):
    """
    将SMILES字符串转换为one-hot编码
    
    Args:
        smiles_list: SMILES字符串列表
        max_length: 序列的最大长度
    
    Returns:
        one_hot_matrix: shape为(n_samples, max_length, vocab_size)的one-hot编码矩阵
        char_to_idx: 字符到索引的映射
        valid_indices: 有效样本的索引
    """
    print("创建SMILES词汇表...")
    char_to_idx, idx_to_char = create_smiles_vocabulary(smiles_list)
    vocab_size = len(char_to_idx)
    
    print(f"词汇表大小: {vocab_size}")
    print(f"词汇表: {sorted(char_to_idx.keys())}")
    
    one_hot_features = []
    valid_indices = []
    
    for i, smiles in enumerate(tqdm(smiles_list, desc="Converting SMILES to one-hot")):
        try:
            # 截断或填充SMILES到固定长度
            if len(smiles) > max_length:
                smiles = smiles[:max_length]
            else:
                smiles = smiles + '<PAD>' * (max_length - len(smiles))
            
            # 创建one-hot编码
            one_hot = np.zeros((max_length, vocab_size))
            
            for j, char in enumerate(smiles):
                if char in char_to_idx:
                    char_idx = char_to_idx[char]
                else:
                    char_idx = char_to_idx['<UNK>']
                one_hot[j, char_idx] = 1.0
            
            # 展平为1D向量
            one_hot_flat = one_hot.flatten()
            one_hot_features.append(one_hot_flat)
            valid_indices.append(i)
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            continue
    
    return np.array(one_hot_features), char_to_idx, valid_indices

def smiles_to_sequence_features(smiles_list, max_length=150):
    """
    将SMILES转换为序列特征（简化版，使用字符频率等统计特征）
    这种方法计算效率更高，适合大规模数据
    """
    print("创建SMILES词汇表...")
    char_to_idx, idx_to_char = create_smiles_vocabulary(smiles_list)
    vocab_size = len(char_to_idx)
    
    features = []
    valid_indices = []
    
    for i, smiles in enumerate(tqdm(smiles_list, desc="Converting SMILES to features")):
        try:
            # 字符频率特征
            char_counts = np.zeros(vocab_size)
            for char in smiles:
                if char in char_to_idx:
                    char_counts[char_to_idx[char]] += 1
                else:
                    char_counts[char_to_idx['<UNK>']] += 1
            
            # 归一化为频率
            char_freqs = char_counts / len(smiles) if len(smiles) > 0 else char_counts
            
            # 添加其他统计特征
            extra_features = [
                len(smiles),  # SMILES长度
                smiles.count('C'),  # 碳原子数量
                smiles.count('N'),  # 氮原子数量
                smiles.count('O'),  # 氧原子数量
                smiles.count('='),  # 双键数量
                smiles.count('('),  # 分支数量
                smiles.count('['),  # 特殊原子数量
                smiles.count('@'),  # 手性中心数量
            ]
            
            # 组合特征
            combined_features = np.concatenate([char_freqs, extra_features])
            features.append(combined_features)
            valid_indices.append(i)
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            continue
    
    return np.array(features), char_to_idx, valid_indices

def extract_ir_features(ir_spectra):
    """提取IR光谱特征"""
    ir_features = []
    for ir in ir_spectra:
        if isinstance(ir, torch.Tensor):
            ir = ir.numpy()
        if ir.ndim > 1:
            ir = ir.flatten()
        ir_features.append(ir)
    
    return np.array(ir_features)

def load_mixed_dataset(datasets_info, samples_per_dataset=2000):
    """
    从多个数据集中加载混合数据
    """
    all_smiles = []
    all_ir = []
    dataset_labels = []
    dataset_names = []
    
    for dataset, dataset_name in datasets_info:
        print(f"从 {dataset_name} 中采样 {samples_per_dataset} 个分子...")
        
        # 随机采样
        n_samples = min(len(dataset), samples_per_dataset)
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        
        dataset_smiles = []
        dataset_ir = []
        
        for idx in tqdm(indices, desc=f"Loading {dataset_name}"):
            try:
                data = dataset[idx]
                dataset_smiles.append(data['smiles'])
                dataset_ir.append(data['ir'])
            except Exception as e:
                continue
        
        # 添加到总数据
        all_smiles.extend(dataset_smiles)
        all_ir.extend(dataset_ir)
        dataset_labels.extend([dataset_name] * len(dataset_smiles))
        dataset_names.append(dataset_name)
        
        print(f"成功从 {dataset_name} 加载 {len(dataset_smiles)} 个分子")
    
    return {
        'smiles': all_smiles,
        'ir': all_ir,
        'dataset_labels': dataset_labels,
        'dataset_names': dataset_names
    }

def calculate_fusion_metrics(cluster_labels, dataset_labels):
    """计算融合度指标"""
    unique_clusters = np.unique(cluster_labels)
    unique_datasets = np.unique(dataset_labels)
    n_datasets = len(unique_datasets)
    
    # 计算每个聚类中不同数据集的分布
    cluster_diversity = []
    cluster_entropy = []
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_dataset_counts = {}
        
        for dataset in unique_datasets:
            count = np.sum((dataset_labels == dataset) & cluster_mask)
            cluster_dataset_counts[dataset] = count
        
        # 计算多样性（不同数据集的数量）
        diversity = sum(1 for count in cluster_dataset_counts.values() if count > 0)
        cluster_diversity.append(diversity)
        
        # 计算熵（分布均匀程度）
        total = sum(cluster_dataset_counts.values())
        if total > 0:
            probs = [count/total for count in cluster_dataset_counts.values() if count > 0]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        else:
            entropy = 0
        cluster_entropy.append(entropy)
    
    # 整体融合指标
    avg_diversity = np.mean(cluster_diversity)
    max_diversity = n_datasets
    fusion_ratio = avg_diversity / max_diversity
    
    avg_entropy = np.mean(cluster_entropy)
    max_entropy = np.log2(n_datasets)
    entropy_ratio = avg_entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        'fusion_ratio': fusion_ratio,
        'entropy_ratio': entropy_ratio,
        'avg_diversity': avg_diversity,
        'avg_entropy': avg_entropy,
        'cluster_diversity': cluster_diversity,
        'cluster_entropy': cluster_entropy
    }

def perform_clustering_and_analysis(mixed_data, n_clusters=12, use_full_onehot=False):
    """执行聚类分析并计算融合指标"""
    smiles_list = mixed_data['smiles']
    ir_spectra = mixed_data['ir']
    dataset_labels = np.array(mixed_data['dataset_labels'])
    
    # 提取特征
    print("提取SMILES特征...")
    if use_full_onehot:
        # 使用完整的one-hot编码（内存占用较大，适合小数据集）
        smiles_features, char_to_idx, valid_indices = smiles_to_onehot(smiles_list, max_length=100)
        print(f"SMILES one-hot特征维度: {smiles_features.shape}")
    else:
        # 使用序列统计特征（推荐，效率更高）
        smiles_features, char_to_idx, valid_indices = smiles_to_sequence_features(smiles_list, max_length=150)
        print(f"SMILES序列特征维度: {smiles_features.shape}")
    
    print("提取IR特征...")
    ir_features = extract_ir_features(ir_spectra)
    
    # 确保特征和标签对应
    if len(valid_indices) < len(ir_features):
        ir_features = ir_features[valid_indices]
        dataset_labels = dataset_labels[valid_indices]
    
    print(f"有效样本数: {len(smiles_features)}")
    print(f"SMILES特征形状: {smiles_features.shape}")
    print(f"IR特征形状: {ir_features.shape}")
    
    # 标准化特征
    scaler_smiles = StandardScaler()
    scaler_ir = StandardScaler()
    smiles_features_scaled = scaler_smiles.fit_transform(smiles_features)
    ir_features_scaled = scaler_ir.fit_transform(ir_features)
    
    # SMILES聚类
    print("执行SMILES聚类...")
    kmeans_smiles = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    smiles_clusters = kmeans_smiles.fit_predict(smiles_features_scaled)
    
    # IR聚类
    print("执行IR聚类...")
    kmeans_ir = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    ir_clusters = kmeans_ir.fit_predict(ir_features_scaled)
    
    # 降维可视化
    print("PCA降维...")
    pca_smiles = PCA(n_components=2, random_state=42)
    pca_ir = PCA(n_components=2, random_state=42)
    smiles_pca = pca_smiles.fit_transform(smiles_features_scaled)
    ir_pca = pca_ir.fit_transform(ir_features_scaled)
    
    print("t-SNE降维...")
    # 限制样本数量以加速t-SNE
    n_tsne_samples = min(3000, len(smiles_features_scaled))
    tsne_indices = np.random.choice(len(smiles_features_scaled), n_tsne_samples, replace=False)
    
    tsne_smiles = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=1000)
    tsne_ir = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=1000)
    
    smiles_tsne = tsne_smiles.fit_transform(smiles_features_scaled[tsne_indices])
    ir_tsne = tsne_ir.fit_transform(ir_features_scaled[tsne_indices])
    
    # 计算融合指标
    smiles_fusion = calculate_fusion_metrics(smiles_clusters, dataset_labels)
    ir_fusion = calculate_fusion_metrics(ir_clusters, dataset_labels)
    
    # 计算轮廓系数
    smiles_silhouette = silhouette_score(smiles_features_scaled, smiles_clusters)
    ir_silhouette = silhouette_score(ir_features_scaled, ir_clusters)
    
    return {
        'smiles_clusters': smiles_clusters,
        'ir_clusters': ir_clusters,
        'smiles_pca': smiles_pca,
        'ir_pca': ir_pca,
        'smiles_tsne': smiles_tsne,
        'ir_tsne': ir_tsne,
        'tsne_indices': tsne_indices,
        'dataset_labels': dataset_labels,
        'smiles_fusion': smiles_fusion,
        'ir_fusion': ir_fusion,
        'smiles_silhouette': smiles_silhouette,
        'ir_silhouette': ir_silhouette,
        'pca_explained_smiles': pca_smiles.explained_variance_ratio_,
        'pca_explained_ir': pca_ir.explained_variance_ratio_,
        'vocab_info': {
            'vocab_size': len(char_to_idx),
            'vocab_chars': list(char_to_idx.keys())
        }
    }

def plot_fusion_analysis(results, mixed_data):
    """绘制融合分析图"""
    dataset_names = mixed_data['dataset_names']
    n_datasets = len(dataset_names)
    
    # 定义颜色映射
    dataset_colors = plt.cm.Set1(np.linspace(0, 1, n_datasets))
    dataset_color_map = {name: color for name, color in zip(dataset_names, dataset_colors)}
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # 1. PCA可视化 - 按数据集着色
    ax1 = plt.subplot(2, 4, 1)
    for i, dataset_name in enumerate(dataset_names):
        mask = results['dataset_labels'] == dataset_name
        plt.scatter(results['smiles_pca'][mask, 0], results['smiles_pca'][mask, 1], 
                   c=[dataset_color_map[dataset_name]], label=dataset_name, 
                   alpha=0.6, s=20)
    plt.xlabel(f'PC1 ({results["pca_explained_smiles"][0]:.1%})')
    plt.ylabel(f'PC2 ({results["pca_explained_smiles"][1]:.1%})')
    plt.title('SMILES One-hot PCA - Dataset Labels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 4, 2)
    for i, dataset_name in enumerate(dataset_names):
        mask = results['dataset_labels'] == dataset_name
        plt.scatter(results['ir_pca'][mask, 0], results['ir_pca'][mask, 1], 
                   c=[dataset_color_map[dataset_name]], label=dataset_name, 
                   alpha=0.6, s=20)
    plt.xlabel(f'PC1 ({results["pca_explained_ir"][0]:.1%})')
    plt.ylabel(f'PC2 ({results["pca_explained_ir"][1]:.1%})')
    plt.title('IR Spectrum PCA - Dataset Labels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. PCA可视化 - 按聚类着色
    ax3 = plt.subplot(2, 4, 3)
    scatter = plt.scatter(results['smiles_pca'][:, 0], results['smiles_pca'][:, 1], 
                         c=results['smiles_clusters'], cmap='tab10', alpha=0.6, s=20)
    plt.xlabel(f'PC1 ({results["pca_explained_smiles"][0]:.1%})')
    plt.ylabel(f'PC2 ({results["pca_explained_smiles"][1]:.1%})')
    plt.title(f'SMILES One-hot PCA - Clusters\nSilhouette: {results["smiles_silhouette"]:.3f}')
    plt.colorbar(scatter, ax=ax3)
    plt.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 4, 4)
    scatter = plt.scatter(results['ir_pca'][:, 0], results['ir_pca'][:, 1], 
                         c=results['ir_clusters'], cmap='tab10', alpha=0.6, s=20)
    plt.xlabel(f'PC1 ({results["pca_explained_ir"][0]:.1%})')
    plt.ylabel(f'PC2 ({results["pca_explained_ir"][1]:.1%})')
    plt.title(f'IR Spectrum PCA - Clusters\nSilhouette: {results["ir_silhouette"]:.3f}')
    plt.colorbar(scatter, ax=ax4)
    plt.grid(True, alpha=0.3)
    
    # 3. t-SNE可视化 - 按数据集着色
    tsne_indices = results['tsne_indices']
    tsne_dataset_labels = results['dataset_labels'][tsne_indices]
    
    ax5 = plt.subplot(2, 4, 5)
    for i, dataset_name in enumerate(dataset_names):
        mask = tsne_dataset_labels == dataset_name
        plt.scatter(results['smiles_tsne'][mask, 0], results['smiles_tsne'][mask, 1], 
                   c=[dataset_color_map[dataset_name]], label=dataset_name, 
                   alpha=0.6, s=20)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('SMILES One-hot t-SNE - Dataset Labels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 4, 6)
    for i, dataset_name in enumerate(dataset_names):
        mask = tsne_dataset_labels == dataset_name
        plt.scatter(results['ir_tsne'][mask, 0], results['ir_tsne'][mask, 1], 
                   c=[dataset_color_map[dataset_name]], label=dataset_name, 
                   alpha=0.6, s=20)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('IR Spectrum t-SNE - Dataset Labels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. t-SNE可视化 - 按聚类着色
    ax7 = plt.subplot(2, 4, 7)
    scatter = plt.scatter(results['smiles_tsne'][:, 0], results['smiles_tsne'][:, 1], 
                         c=results['smiles_clusters'][tsne_indices], cmap='tab10', 
                         alpha=0.6, s=20)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('SMILES One-hot t-SNE - Clusters')
    plt.colorbar(scatter, ax=ax7)
    plt.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(2, 4, 8)
    scatter = plt.scatter(results['ir_tsne'][:, 0], results['ir_tsne'][:, 1], 
                         c=results['ir_clusters'][tsne_indices], cmap='tab10', 
                         alpha=0.6, s=20)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('IR Spectrum t-SNE - Clusters')
    plt.colorbar(scatter, ax=ax8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/fusion_analysis_onehot.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    plt.savefig('figs/fusion_analysis_onehot.pdf', bbox_inches='tight', 
                facecolor='white')
    plt.show()

def plot_fusion_metrics(results, mixed_data):
    """绘制融合度指标对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 融合比例对比
    methods = ['SMILES One-hot', 'IR Spectrum']
    fusion_ratios = [results['smiles_fusion']['fusion_ratio'], 
                    results['ir_fusion']['fusion_ratio']]
    entropy_ratios = [results['smiles_fusion']['entropy_ratio'], 
                     results['ir_fusion']['entropy_ratio']]
    
    colors = ['lightcoral', 'skyblue']
    
    ax1.bar(methods, fusion_ratios, color=colors, alpha=0.8)
    ax1.set_ylabel('Dataset Fusion Ratio')
    ax1.set_title('Dataset Fusion in Clusters\n(Higher = Better Mixing)')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(fusion_ratios):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(methods, entropy_ratios, color=colors, alpha=0.8)
    ax2.set_ylabel('Distribution Entropy Ratio')
    ax2.set_title('Cluster Diversity\n(Higher = More Uniform Distribution)')
    ax2.set_ylim(0, 1)
    for i, v in enumerate(entropy_ratios):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 每个聚类的多样性分布
    ax3.hist(results['smiles_fusion']['cluster_diversity'], bins=range(1, len(mixed_data['dataset_names'])+2), 
             alpha=0.7, label='SMILES One-hot', color='lightcoral', density=True)
    ax3.hist(results['ir_fusion']['cluster_diversity'], bins=range(1, len(mixed_data['dataset_names'])+2), 
             alpha=0.7, label='IR Spectrum', color='skyblue', density=True)
    ax3.set_xlabel('Number of Datasets per Cluster')
    ax3.set_ylabel('Density')
    ax3.set_title('Cluster Diversity Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 聚类质量对比
    silhouette_scores = [results['smiles_silhouette'], results['ir_silhouette']]
    ax4.bar(methods, silhouette_scores, color=colors, alpha=0.8)
    ax4.set_ylabel('Silhouette Score')
    ax4.set_title('Clustering Quality')
    for i, v in enumerate(silhouette_scores):
        ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/fusion_metrics_onehot.png', dpi=600, bbox_inches='tight', 
                facecolor='white')
    plt.savefig('figs/fusion_metrics_onehot.pdf', bbox_inches='tight', 
                facecolor='white')
    plt.show()

def main():
    """主函数"""
    import os
    os.makedirs('figs', exist_ok=True)
    
    print("=== 数据集融合聚类分析 (SMILES One-hot编码) ===")
    
    # 准备数据集
    datasets_info = []
    
    # 加载IRDataSet
    try:
        ir_dataset = IRDataSet('/home/yanggk/SpecGLU/data/5584_res.csv')
        datasets_info.append((ir_dataset, "RTP"))
        print(f"IRDataSet 加载成功，包含 {len(ir_dataset)} 个样本")
    except Exception as e:
        print(f"IRDataSet 加载失败: {e}")
    
    # 加载ESPSet
    try:
        esp_dataset = ESPSet('data/merged_file_rtp.csv')
        datasets_info.append((esp_dataset, "ESP"))
        print(f"ESPSet 加载成功，包含 {len(esp_dataset)} 个样本")
    except Exception as e:
        print(f"ESPSet 加载失败: {e}")
    
    # 加载QM9IRDataSet
    try:
        qm9_path = "/home/yanggk/Data/SpecGLU/QM9/SpecBert/gaussian_summary.csv"
        if os.path.exists(qm9_path):
            qm9_dataset = QM9IRDataSet(qm9_path)
            datasets_info.append((qm9_dataset, "QM9"))
            print(f"QM9IRDataSet 加载成功，包含 {len(qm9_dataset)} 个样本")
    except Exception as e:
        print(f"QM9IRDataSet 加载失败: {e}")
    
    if len(datasets_info) < 2:
        print("至少需要2个数据集才能进行融合分析")
        return
    
    # 加载混合数据
    mixed_data = load_mixed_dataset(datasets_info, samples_per_dataset=1500)  # 减少样本数以适应one-hot编码
    total_samples = len(mixed_data['smiles'])
    print(f"\n总共加载了 {total_samples} 个分子样本")
    
    # 执行聚类分析
    # use_full_onehot=False 使用序列统计特征（推荐）
    # use_full_onehot=True 使用完整one-hot编码（内存占用大）
    results = perform_clustering_and_analysis(mixed_data, n_clusters=12, use_full_onehot=False)
    
    # 绘制结果
    plot_fusion_analysis(results, mixed_data)
    plot_fusion_metrics(results, mixed_data)
    
    # 打印结果
    print("\n=== 分析结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"数据集: {', '.join(mixed_data['dataset_names'])}")
    print(f"SMILES词汇表大小: {results['vocab_info']['vocab_size']}")
    
    print(f"\nSMILES One-hot聚类:")
    print(f"  轮廓系数: {results['smiles_silhouette']:.3f}")
    print(f"  数据集融合比例: {results['smiles_fusion']['fusion_ratio']:.3f}")
    print(f"  分布熵比例: {results['smiles_fusion']['entropy_ratio']:.3f}")
    
    print(f"\nIR光谱聚类:")
    print(f"  轮廓系数: {results['ir_silhouette']:.3f}")
    print(f"  数据集融合比例: {results['ir_fusion']['fusion_ratio']:.3f}")
    print(f"  分布熵比例: {results['ir_fusion']['entropy_ratio']:.3f}")
    
    fusion_improvement = results['ir_fusion']['fusion_ratio'] - results['smiles_fusion']['fusion_ratio']
    print(f"\nIR光谱聚类的融合度提升: {fusion_improvement:.3f}")
    
    if fusion_improvement > 0:
        print("✓ IR光谱聚类确实实现了更好的数据集融合！")
    else:
        print("× IR光谱聚类的融合度没有显著提升")

if __name__ == "__main__":
    main()