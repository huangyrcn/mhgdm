'''
此脚本用于对图神经网络编码器的输出进行降维和可视化。
它加载预先训练的编码器输出的嵌入（embeddings）和相应的标签，
然后使用 t-SNE 将嵌入降到二维，并使用 Matplotlib 绘制散点图。
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from omegaconf import OmegaConf # 用于加载和合并配置

# 将项目根目录添加到 sys.path
# __file__ 是当前脚本 (visualize_encoders.py) 的路径
# os.path.dirname(__file__) 是 demo/
# os.path.dirname(os.path.dirname(__file__)) 是 mhgdm/ (项目根目录)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入您的项目中的相关模块
# 请确保这些路径相对于您的项目根目录是正确的
from models.Encoders import HGCN as HGCNEncoder, GCN as GCNEncoder # 假设的导入路径
from utils.data_utils import MyDataset # 假设的数据加载器导入路径
from utils.graph_utils import node_flags # 假设的工具函数导入路径

# 定义编码器名称和文件路径模板
# ENCODER_NAMES = [
#     "Letter_high_HGCN_encoder_train_20250526_195321",
#     "Letter_high_GCN_encoder_train_20250526_203738"
# ]

# 更新为模型检查点路径
ENCODER_CHECKPOINT_PATHS = {
    "Letter_high_HGCN_encoder_train_20250526_212414": "checkpoints/Letter_high_HGCN_encoder_train/20250526_212414/best.pth",
    "Letter_high_GCN_encoder_train_20250526_212747": "checkpoints/Letter_high_GCN_encoder_train/20250526_212747/best.pth"
}

# 假设嵌入和标签文件存储在 'data/' 目录下
# 您可能需要根据您的实际文件结构修改这些路径
# EMBEDDING_FILE_TEMPLATE = "data/{encoder_name}_embeddings.npy" # 不再使用
LABEL_FILE_TEMPLATE = "data/Letter_high_labels.npy" # 保持，但将从数据加载器获取

# 可视化输出的保存路径
OUTPUT_DIR = "output/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_embeddings(encoder_name, checkpoint_path, labels_path):
    '''
    加载嵌入和标签，执行 t-SNE 降维，并生成散点图。

    参数:
        encoder_name (str): 编码器的名称，用于图表标题和文件名。
        checkpoint_path (str): 检查点文件的路径。
        labels_path (str): 标签文件 (.npy) 的路径。
    '''
    print(f"正在处理编码器: {encoder_name}")

    # --- 1. 加载配置 ---
    try:
        # 基础数据配置路径
        data_cfg_path = "configs/data/Letter_high.yaml"

        # 模型特定配置 - 假设与 checkpoint 在同一目录下，或有标准路径
        # 您可能需要调整此逻辑以找到正确的模型配置文件
        model_name_parts = encoder_name.split('_')
        encoder_type_from_name = model_name_parts[2] # HGCN or GCN
        actual_model_cfg_path = None # 将存储最终找到的模型配置文件路径

        # 假设配置文件在 'configs/encoder/' 目录下
        # 您需要确保这些配置文件的存在和正确性
        if "HGCN" in encoder_type_from_name.upper():
            primary_model_cfg_path = "configs/encoder/HGCN.yaml"
        elif "GCN" in encoder_type_from_name.upper():
            primary_model_cfg_path = "configs/encoder/GCN.yaml"
        else:
            print(f"错误: 无法从名称 {encoder_name} 推断模型类型或找到配置文件。")
            return

        if os.path.exists(primary_model_cfg_path):
            actual_model_cfg_path = primary_model_cfg_path
        else:
            print(f"信息: 主要模型配置文件未找到: {primary_model_cfg_path}")
            # 尝试从 checkpoint 目录加载（如果存在）
            checkpoint_dir = os.path.dirname(checkpoint_path)
            alt_model_cfg_path = os.path.join(checkpoint_dir, "config.yaml") # 假设名称
            if os.path.exists(alt_model_cfg_path):
                print(f"尝试备用配置文件: {alt_model_cfg_path}")
                actual_model_cfg_path = alt_model_cfg_path
            else:
                print(f"错误: 主要模型配置文件 {primary_model_cfg_path} 和备用配置文件 {alt_model_cfg_path} 均未找到。请提供正确的模型配置文件。")
                return
        
        # --- 新的配置加载和解析逻辑 ---
        composite_cfg = OmegaConf.create()

        # 1. 加载数据配置到 composite_cfg.data
        if not os.path.exists(data_cfg_path):
            print(f"错误: 数据配置文件未找到: {data_cfg_path}")
            return
        composite_cfg.data = OmegaConf.load(data_cfg_path)

        # 2. 加载模型配置到 composite_cfg.model
        #    这样模型配置中的 ${data.xxx} 插值可以引用 composite_cfg.data.xxx
        if not actual_model_cfg_path or not os.path.exists(actual_model_cfg_path):
             print(f"错误: 最终模型配置文件路径无效或文件不存在: {actual_model_cfg_path}")
             return
        composite_cfg.model = OmegaConf.load(actual_model_cfg_path)
        
        # model_cfg 将是解析后的模型参数部分
        model_cfg = composite_cfg.model

        # 后备逻辑: 如果模型配置文件中完全没有定义 input_feat_dim，则从数据配置中获取
        if not hasattr(model_cfg, 'input_feat_dim'):
            if hasattr(composite_cfg.data, 'max_feat_num'):
                print(f"  信息: 'input_feat_dim' 未在模型配置 {actual_model_cfg_path} 中定义。"
                      f" 将使用 data_cfg.max_feat_num ({composite_cfg.data.max_feat_num}).")
                model_cfg.input_feat_dim = composite_cfg.data.max_feat_num
            else:
                print(f"  警告: 'input_feat_dim' 未在模型配置中定义，且 'data_cfg.max_feat_num' 在 {data_cfg_path} 中不可用。")

        # 后备逻辑: 针对GCN的'dim'参数
        if "GCN" in encoder_type_from_name.upper():
            if not hasattr(model_cfg, 'dim'):
                # GCN编码器可能从 hidden_dim[0] 或其他逻辑内部推断其有效输入维度
                # 仅当'dim'绝对需要且未由模型配置设置，并且不由hidden_dim隐式处理时，才从data_cfg.max_feat_num设置，作为最后手段。
                if not hasattr(model_cfg, 'hidden_dim'): # 如果模型配置中也没有 hidden_dim
                    if hasattr(composite_cfg.data, 'max_feat_num'):
                        print(f"  信息: GCN 'dim' 未在模型配置 {actual_model_cfg_path} 中定义且无 'hidden_dim'。"
                              f" 将使用 data_cfg.max_feat_num ({composite_cfg.data.max_feat_num}).")
                        model_cfg.dim = composite_cfg.data.max_feat_num
                    else:
                         print(f"  警告: GCN 'dim' 未在模型配置中定义，无 'hidden_dim'，且 'data_cfg.max_feat_num' 在 {data_cfg_path} 中不可用。")
        # --- 配置加载结束 ---

    except OmegaConf.errors.InterpolationResolutionError as e:
        print(f"错误: 配置插值解析失败 - {e}")
        print(f"  这通常意味着配置文件 (例如 {actual_model_cfg_path}) 中的变量 (例如 ${{data.max_feat_num}}) 无法找到其引用的值。")
        print(f"  请检查数据配置文件 ({data_cfg_path}) 是否包含必要的键 (例如 max_feat_num)，以及模型配置中的引用是否正确。")
        return
    except FileNotFoundError as e:
        print(f"错误: 配置文件加载失败 - {e}")
        return
    except Exception as e:
        # 打印更详细的异常类型和信息
        print(f"加载配置时发生未知错误: {type(e).__name__} - {e}")
        return

    print("  配置加载完成。")

    # --- 2. 准备数据 ---
    try:
        # 使用 composite_cfg.data 初始化数据集
        # fsl_task_config 设置为 None，因为我们只需要全量数据进行推理
        dataset = MyDataset(data_config=composite_cfg.data, fsl_task_config=None)
        # 获取一个包含所有测试数据的 DataLoader
        # 或者，如果您的 MyDataset 支持获取所有数据而不通过 DataLoader，则更好
        # 这里假设 get_loaders 返回 (train_loader, test_loader)
        # 我们将使用 test_loader，或者合并 train 和 test 以获得所有数据
        _, data_loader = dataset.get_loaders() # 获取测试数据加载器
        
        # 如果需要所有数据（训练+测试）
        # all_x = torch.cat((dataset.train_x, dataset.test_x), dim=0)
        # all_adjs = torch.cat((dataset.train_adjs, dataset.test_adjs), dim=0)
        # all_labels = torch.cat((dataset.train_labels_remapped, dataset.test_labels_remapped), dim=0)
        # from torch.utils.data import TensorDataset, DataLoader
        # all_dataset = TensorDataset(all_x, all_adjs, all_labels)
        # data_loader = DataLoader(all_dataset, batch_size=composite_cfg.data.batch_size, shuffle=False) # 使用 composite_cfg.data
        
        print(f"  数据集 '{composite_cfg.data.name}' 加载完成。使用测试集进行可视化。") # 使用 composite_cfg.data
    except Exception as e:
        print(f"数据加载和处理时出错: {e}")
        return

    # --- 3. 初始化模型并加载权重 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  使用设备: {device}")

    try:
        if "HGCN" in encoder_name.upper():
            # 确保 model_cfg 包含 HGCNEncoder 需要的所有参数
            # 例如: input_feat_dim, hidden_dim, dim, enc_layers, layer_type, dropout, 
            # edge_dim, normalization_factor, aggregation_method, msg_transform,
            # manifold, c, learnable_c, sum_transform, use_norm
            encoder = HGCNEncoder(config=model_cfg).to(device)
        elif "GCN" in encoder_name.upper():
            # 确保 model_cfg 包含 GCNEncoder 需要的所有参数
            # 例如: input_feat_dim, hidden_dim, dim (可能没有), enc_layers, layer_type, 
            # dropout, edge_dim, normalization_factor, aggregation_method, msg_transform
            encoder = GCNEncoder(config=model_cfg).to(device)
        else:
            print(f"错误: 无法识别的编码器类型 {encoder_name}")
            return

        if not os.path.exists(checkpoint_path):
            print(f"错误: 检查点文件未找到: {checkpoint_path}")
            return
        
        state_dict = torch.load(checkpoint_path, map_location=device)
        # 处理可能的 'model_state_dict' 或 'ema_model_state_dict' 键
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'ema_model_state_dict' in state_dict: # 如果使用了EMA
            state_dict = state_dict['ema_model_state_dict']
        elif 'encoder_state_dict' in state_dict: # 另一种常见模式
             state_dict = state_dict['encoder_state_dict']
        elif 'model' in state_dict and isinstance(state_dict['model'], dict): # 有些保存整个模型的字典
            state_dict = state_dict['model']
            
        # 修正键名 (如果模型保存时带有 'module.' 前缀，例如使用 DataParallel)
        # 同时处理 state_dict 可能就是模型权重本身的情况
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            encoder.load_state_dict(new_state_dict)
        else:
            # 如果没有 'module.' 前缀，直接加载
            # 这也处理了 state_dict 本身就是权重的情况
            encoder.load_state_dict(state_dict)

        encoder.eval()
        print(f"  模型 {encoder_name} 加载权重并设置为评估模式。")

    except FileNotFoundError:
        print(f"错误: 模型检查点文件未找到: {checkpoint_path}")
        return
    except RuntimeError as e:
        print(f"加载模型状态字典时出错: {e}")
        print("这可能意味着模型定义与检查点不匹配，或者检查点文件已损坏。")
        print("请确保您使用的模型配置文件与训练时完全一致。")
        return
    except Exception as e:
        print(f"初始化模型或加载权重时发生未知错误: {e}")
        return

    # --- 4. 获取嵌入 ---
    all_embeddings_list = []
    all_labels_list = []
    print("  正在从数据加载器生成嵌入...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            x, adj, labels_batch = batch_data
            x, adj = x.to(device), adj.to(device)
            
            # 根据 Encoder 的 forward/encode 方法调整输入
            # 假设 Encoders.py 中的 encode 方法是: encode(self, x, adj, node_mask=None)
            # node_mask 可能需要根据 adj 生成
            # flags = node_flags(adj) # 从 utils.graph_utils 导入
            # node_mask = flags.unsqueeze(-1) if flags is not None else None

            # 如果您的 Encoder.encode 需要 node_mask:
            # node_mask_batch = node_flags(adj).unsqueeze(-1).to(device) # (B, N, 1)
            # current_embeddings = encoder.encode(x, adj, node_mask_batch) 
            # 否则，如果只需要 x 和 adj:
            current_embeddings = encoder.encode(x, adj) # (B, N, D_embed) or (B, D_embed)

            # 处理输出: 假设 current_embeddings 是 (Batch, NumNodes, EmbeddingDim)
            # 如果是图级别嵌入，它可能是 (Batch, EmbeddingDim)
            # 如果是节点级嵌入，我们需要决定如何处理它们。
            # 为了可视化，通常需要每个图一个向量。
            # 策略1: 使用图的平均节点嵌入 (如果适用)
            if current_embeddings.ndim == 3: # (B, N, D_embed)
                # 创建节点掩码，只对真实节点进行平均
                # (B, N)
                node_mask_for_avg = node_flags(adj).to(device) 
                # (B, N, 1)
                expanded_mask = node_mask_for_avg.unsqueeze(-1).float()
                # (B, N, D_embed)
                masked_embeddings = current_embeddings * expanded_mask
                # (B, D_embed)
                summed_embeddings = masked_embeddings.sum(dim=1)
                # (B, 1)
                num_valid_nodes = node_mask_for_avg.sum(dim=1, keepdim=True).float().clamp(min=1)
                graph_embeddings = summed_embeddings / num_valid_nodes
            else: # 假设已经是 (B, D_embed)
                graph_embeddings = current_embeddings

            all_embeddings_list.append(graph_embeddings.cpu().numpy())
            all_labels_list.append(labels_batch.cpu().numpy())
            if batch_idx % 10 == 0:
                print(f"    处理完批次 {batch_idx+1}/{len(data_loader)}")

    if not all_embeddings_list:
        print("错误: 未能生成任何嵌入。")
        return

    embeddings = np.concatenate(all_embeddings_list, axis=0)
    labels = np.concatenate(all_labels_list, axis=0)
    
    print(f"  嵌入生成完成。嵌入形状: {embeddings.shape}, 标签形状: {labels.shape}")


    # 检查文件是否存在 (现在不需要了，因为我们动态生成嵌入)
    # if not os.path.exists(embeddings_path):
    #     print(f"错误: 嵌入文件未找到: {embeddings_path}")
    #     print("请确保文件存在，或者修改脚本中的 EMBEDDING_FILE_TEMPLATE。")
    #     return
    # if not os.path.exists(labels_path):
    #     print(f"错误: 标签文件未找到: {labels_path}")
    #     print("请确保文件存在，或者修改脚本中的 LABEL_FILE_TEMPLATE。")
    #     return

    # 加载嵌入和标签 (现在不需要了)
    # try:
    #     embeddings = np.load(embeddings_path)
    #     labels = np.load(labels_path)
    # except Exception as e:
    #     print(f"加载文件时出错: {e}")
    #     return

    # print(f"  嵌入形状: {embeddings.shape}")
    # print(f"  标签形状: {labels.shape}")

    if embeddings.shape[0] != labels.shape[0]:
        print("错误: 嵌入和标签的数量不匹配。")
        return

    # 执行 t-SNE 降维
    print("  正在执行 t-SNE 降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0] - 1))
    embeddings_2d = tsne.fit_transform(embeddings)
    print("  t-SNE 完成。")

    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(f't-SNE Visualization of {encoder_name} Embeddings')
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # 添加颜色条
    try:
        # 尝试获取类别数量，如果标签是数值型的
        num_classes = len(np.unique(labels))
        if num_classes <= 20: # 只为类别较少的情况显示详细的图例
            handles, _ = scatter.legend_elements(prop="colors", alpha=0.7)
            if handles:
                 legend_labels = [f'Class {i}' for i in np.unique(labels)]
                 plt.legend(handles, legend_labels, title="Classes")
        else:
            plt.colorbar(scatter, label='Label')
    except TypeError:
         # 如果标签不是数值型的，或者unique失败，就用简单的colorbar
        plt.colorbar(scatter, label='Label')


    # 保存图像
    output_filename = os.path.join(OUTPUT_DIR, f"{encoder_name}_tsne_visualization.png")
    plt.savefig(output_filename)
    plt.close()
    print(f"  可视化结果已保存到: {output_filename}")


if __name__ == "__main__":
    # if not os.path.exists(LABEL_FILE_TEMPLATE): # 标签现在从数据加载器获取
    #     print(f"警告: 默认标签文件 {LABEL_FILE_TEMPLATE} 不存在。")
    #     # ... (旧的虚拟标签创建代码) ...

    # for encoder_name in ENCODER_NAMES: # 改用 ENCODER_CHECKPOINT_PATHS
    #     embeddings_file = EMBEDDING_FILE_TEMPLATE.format(encoder_name=encoder_name)
    #     visualize_embeddings(encoder_name, embeddings_file, LABEL_FILE_TEMPLATE)

    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA。脚本将在 CPU 上运行，可能会比较慢。")

    for encoder_name, checkpoint_path in ENCODER_CHECKPOINT_PATHS.items():
        # visualize_embeddings 现在需要 encoder_name 和 checkpoint_path
        # 标签路径不再直接传递，而是通过数据加载器内部处理
        visualize_embeddings(encoder_name, checkpoint_path, None) # labels_path 设为 None

    print("\n所有编码器的可视化处理完成。")
    print(f"请检查 '{OUTPUT_DIR}' 目录下的图像文件。")

    # 提醒用户检查文件路径 (部分信息已更改)
    print("\n重要提示:")
    # print(f"请确保以下文件路径是正确的，并且文件确实存在：")
    # print(f"  - 标签文件: {LABEL_FILE_TEMPLATE}") # 标签现在从数据加载器获取
    print(f"  - 数据配置文件: configs/data/Letter_high.yaml")
    print(f"  - 模型配置文件: 例如 configs/encoder/HGCN.yaml, configs/encoder/GCN.yaml (或 checkpoint 目录下的 config.yaml)")
    for encoder_name, ckpt_path in ENCODER_CHECKPOINT_PATHS.items():
        print(f"  - {encoder_name} 的检查点文件: {ckpt_path}")
    print("如果路径不正确或脚本运行出错，请检查：")
    print("  1. 脚本顶部的导入路径是否正确。")
    print("  2. 模型配置文件是否与训练时使用的配置一致。")
    print("  3. `MyDataset` 和编码器类的实现是否符合预期。")
    print("  4. 如果模型使用了 DataParallel (带 'module.' 前缀的键)，确保在加载权重时正确处理。")
