"""
debug_encoder.py - 编码器调试和验证脚本
=====================================

功能描述:
- 训练图变分自编码器(GraphVAE)模型50轮
- 在元学习(meta-learning)测试集上评估支持集和查询集的准确率
- 诊断特征质量，分析模型性能瓶颈

重要更新:
- 现在使用vae_trainer._extract_features进行统一的特征提取
- 特征维度变更: [batch_size, latent_dim] -> [batch_size, latent_dim*2] (mean+max组合池化)
- 流形几何处理: 正确支持双曲空间和欧几里得空间
- 与GraphVAE、Classifier等组件完全一致的特征表征

使用场景:
- 调试训练好的VAE编码器性能
- 验证少样本学习(Few-shot Learning)任务的效果
- 分析特征质量和可分性
"""

# 标准库导入
import os
from datetime import datetime

# 第三方库导入
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 本地模块导入
from vae_trainer import train_vae, meta_eval, _extract_features, _test_with_prototypical_networks
from utils.config_utils import load_config
from utils.graph_utils import node_flags
from utils.manifolds_utils import get_manifold


def evaluate_task_with_prototypical_networks(encoder, task, config, device):
    """
    使用原型网络方法在单个少样本学习任务上评估编码器性能

    原型网络(Prototypical Networks)是一种经典的少样本学习方法，通过计算支持集样本的类别原型，
    然后基于查询集样本与原型的距离进行分类。

    参数:
        encoder: 训练好的图编码器，用于提取图特征
        task: 少样本学习任务，包含支持集和查询集
        config: 配置对象，包含N-way等参数
        device: 计算设备(CPU/GPU)

    返回:
        tuple: (支持集准确率, 查询集准确率)
    """
    # 获取N-way设置(即分类类别数)
    N_way = config.fsl_task.N_way

    # === 数据预处理：提取任务中的支持集和查询集数据 ===
    # 支持集：用于构建类别原型的带标签样本
    support_x = task["support_set"]["x"].to(device)  # 节点特征 [batch_size, num_nodes, feat_dim]
    support_adj = task["support_set"]["adj"].to(
        device
    )  # 邻接矩阵 [batch_size, num_nodes, num_nodes]
    support_labels = task["support_set"]["label"].to(device)  # 图级别标签 [batch_size]

    # 查询集：需要进行分类预测的样本
    query_x = task["query_set"]["x"].to(device)
    query_adj = task["query_set"]["adj"].to(device)
    query_labels = task["query_set"]["label"].to(device)

    # 输出调试信息，便于追踪数据维度
    print(f"  Support set shape: {support_x.shape}, labels: {support_labels}")
    print(f"  Query set shape: {query_x.shape}, labels: {query_labels}")
    print(f"  N_way: {N_way}")

    # === 特征提取：使用编码器提取图的潜在表征 ===
    encoder.eval()  # 设置为评估模式，关闭dropout等
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算
        # 注意：_extract_features现在返回组合特征 [batch_size, latent_dim*2] (mean+max池化)
        # 这些特征已经在切空间中，应该使用欧几里得几何处理
        support_features = _extract_features(encoder, support_x, support_adj, device)
        query_features = _extract_features(encoder, query_x, query_adj, device)

    print(f"  Support features shape: {support_features.shape} (mean+max组合特征)")
    print(f"  Query features shape: {query_features.shape} (mean+max组合特征)")

    # === 特征质量诊断：分析提取特征的统计特性和可分性 ===
    _diagnose_feature_quality(support_features, support_labels, query_features, query_labels, N_way)

    # === 原型网络评估：分别计算支持集和查询集的分类准确率 ===
    # 关键修复：使用欧几里得原型网络，因为特征已经在切空间中
    print("  评估支持集（使用欧几里得原型网络）:")
    # 支持集上的自我验证(用支持集既做原型又做测试)
    acc_support = _test_with_prototypical_networks(
        support_features, support_labels, support_features, support_labels, N_way, device
    )

    # 查询集分类(用支持集特征和标签做原型，查询集做分类)
    print("  评估查询集（使用欧几里得原型网络）:")
    acc_query = _test_with_prototypical_networks(
        support_features, support_labels, query_features, query_labels, N_way, device
    )
    return acc_support, acc_query


def _diagnose_feature_quality(
    support_features, support_labels, query_features, query_labels, n_way
):
    """
    特征质量诊断函数 - 全面分析提取特征的统计特性和可分性

    该函数通过多个维度分析特征质量:
    1. 基本统计信息(均值、方差、范数等)
    2. 类别原型分析(类内聚集性)
    3. 类间距离分析(类间可分性)
    4. 特征有效性检查(数值稳定性)
    5. 改进建议

    参数:
        support_features: 支持集特征 [num_support, feature_dim]
        support_labels: 支持集标签 [num_support]
        query_features: 查询集特征 [num_query, feature_dim]
        query_labels: 查询集标签 [num_query]
        n_way: 分类类别数
    """
    print(f"\n  === 特征质量诊断 ===")

    # === 1. 基本统计信息分析 ===
    # 合并所有特征用于整体统计
    all_features = torch.cat([support_features, query_features], dim=0)
    feature_mean = all_features.mean(dim=0)  # 每个维度的均值
    feature_std = all_features.std(dim=0)  # 每个维度的标准差

    print(f"  特征维度: {all_features.shape[1]}")
    print(f"  特征均值范围: [{feature_mean.min():.4f}, {feature_mean.max():.4f}]")
    print(f"  特征标准差范围: [{feature_std.min():.4f}, {feature_std.max():.4f}]")
    print(
        f"  特征范数: 均值={torch.norm(all_features, dim=1).mean():.4f}, "
        f"标准差={torch.norm(all_features, dim=1).std():.4f}"
    )

    # === 2. 类别原型分析 - 评估类内聚集性 ===
    print(f"  各类别原型分析:")
    class_prototypes = []
    for class_id in range(n_way):
        # 找到当前类别的所有支持集样本
        class_mask = support_labels == class_id
        if class_mask.sum() > 0:
            # 计算该类别的特征均值作为原型
            class_features = support_features[class_mask]
            class_proto = class_features.mean(dim=0)
            class_prototypes.append(class_proto)

            # 计算类内方差(衡量类内聚集性)
            intra_variance = ((class_features - class_proto.unsqueeze(0)) ** 2).mean().item()
            print(f"    类别{class_id}: 样本数={class_mask.sum()}, 类内方差={intra_variance:.4f}")

    # === 3. 类间距离分析 - 评估类间可分性 ===
    if len(class_prototypes) >= 2:
        class_prototypes = torch.stack(class_prototypes)
        inter_distances = []

        # 计算所有类别原型之间的欧几里得距离
        for i in range(len(class_prototypes)):
            for j in range(i + 1, len(class_prototypes)):
                dist = torch.norm(class_prototypes[i] - class_prototypes[j], p=2).item()
                inter_distances.append(dist)

        print(
            f"  类间距离: 最小={min(inter_distances):.4f}, "
            f"最大={max(inter_distances):.4f}, "
            f"平均={sum(inter_distances)/len(inter_distances):.4f}"
        )

        # === 可分性评估：基于类间距离判断分类难度 ===
        avg_inter_dist = sum(inter_distances) / len(inter_distances)
        if avg_inter_dist < 1.0:
            print(f"  ❌ 类间距离过小 (<1.0), 建议增加原型分离损失")
        elif avg_inter_dist > 5.0:
            print(f"  ✅ 类间距离良好 (>5.0)")
        else:
            print(f"  ⚠️  类间距离中等 (1.0-5.0), 可以进一步优化")

    # === 4. 特征有效性检查 - 数值稳定性验证 ===
    # 检查零特征向量(可能表示编码器输出退化)
    zero_features = (all_features.abs() < 1e-6).all(dim=1).sum().item()
    if zero_features > 0:
        print(f"  ❌ 发现{zero_features}个零特征向量")

    # 检查NaN值(可能表示数值计算不稳定)
    nan_features = torch.isnan(all_features).any(dim=1).sum().item()
    if nan_features > 0:
        print(f"  ❌ 发现{nan_features}个包含NaN的特征")

    # 检查无穷值(可能表示梯度爆炸或数值溢出)
    inf_features = torch.isinf(all_features).any(dim=1).sum().item()
    if inf_features > 0:
        print(f"  ❌ 发现{inf_features}个包含Inf的特征")

    # 如果特征数值健康，给出正面反馈
    if zero_features == 0 and nan_features == 0 and inf_features == 0:
        print(f"  ✅ 特征数值健康")

    # === 5. 改进建议 - 基于诊断结果提供优化方向 ===
    # 只有当类间距离过小时才提供建议(避免局部变量未定义错误)
    if "avg_inter_dist" in locals() and avg_inter_dist < 1.0:
        print(f"\n  💡 改进建议:")
        print(f"    1. 增加原型分离损失权重 (sep_proto_weight)")
        print(f"    2. 增加训练轮数 (当前50轮可能不够)")
        print(f"    3. 调整学习率或使用学习率调度")
        print(f"    4. 检查VAE损失权重配置")


def _test_with_hyperbolic_prototypical_networks(
    support_features, support_labels, query_features, query_labels, n_way, device, manifold
):
    """
    使用双曲几何的原型网络进行分类 - 正确处理Poincaré球流形
    """
    try:
        print(
            f"    双曲原型网络输入: support_features={support_features.shape}, support_labels={support_labels}"
        )
        print(f"    query_features={query_features.shape}, query_labels={query_labels}")
        print(f"    流形类型: {manifold.name}, 曲率: {manifold.c}")

        # 检查特征是否在流形上
        print(f"    支持集特征范数: {torch.norm(support_features, dim=1).max().item():.6f}")
        print(f"    查询集特征范数: {torch.norm(query_features, dim=1).max().item():.6f}")

        # 计算每个类别的双曲原型
        support_protos = []
        for class_id in range(n_way):
            # 找到该类别的支持集样本
            class_mask = support_labels == class_id
            if class_mask.sum() > 0:
                class_features = support_features[class_mask]
                # 使用双曲几何的加权中点计算原型
                if len(class_features) == 1:
                    class_proto = class_features[0]
                else:
                    # 在双曲空间中计算等权重中点
                    weights = torch.ones(len(class_features), device=device) / len(class_features)
                    class_proto = manifold.weighted_midpoint(class_features, weights)
                support_protos.append(class_proto)

                # 计算类内双曲距离的统计信息
                if class_features.shape[0] > 1:
                    intra_distances = []
                    for i in range(len(class_features)):
                        for j in range(i + 1, len(class_features)):
                            dist = manifold.dist(class_features[i], class_features[j])
                            intra_distances.append(dist.item())
                    avg_intra_dist = (
                        sum(intra_distances) / len(intra_distances) if intra_distances else 0
                    )
                    print(f"    类别{class_id}内部平均双曲距离: {avg_intra_dist:.4f}")
            else:
                print(f"    警告: 类别{class_id}没有支持集样本")
                # 使用流形的原点作为默认原型
                support_protos.append(manifold.origin(support_features.shape[-1:], device=device))

        support_protos = torch.stack(support_protos)  # [n_way, feature_dim]
        print(f"    原型形状: {support_protos.shape}")

        # 计算原型之间的双曲距离
        print(f"    原型间双曲距离矩阵:")
        for i in range(n_way):
            for j in range(n_way):
                if i != j:
                    dist = manifold.dist(support_protos[i], support_protos[j])
                    print(f"      类别{i}与类别{j}双曲距离: {dist.item():.4f}")

        # 计算查询样本与原型之间的双曲距离
        distances = []
        for i in range(len(query_features)):
            query_dists = []
            for j in range(len(support_protos)):
                dist = manifold.dist(query_features[i], support_protos[j])
                query_dists.append(dist)
            distances.append(torch.stack(query_dists))
        distances = torch.stack(distances)  # [query_size, n_way]

        # 使用距离的负数作为分数（距离越小，分数越高）
        scores = -distances

        # 预测
        y_preds = torch.argmax(scores, dim=1)
        print(f"    预测结果: {y_preds}")
        print(f"    真实标签: {query_labels}")

        # 计算准确率
        correct = (y_preds == query_labels).float().sum().item()
        accuracy = correct / len(query_labels) if len(query_labels) > 0 else 0.0

        print(f"    准确率: {correct}/{len(query_labels)} = {accuracy:.4f}")
        return accuracy

    except Exception as e:
        print(f"    双曲原型网络错误: {e}")
        import traceback

        traceback.print_exc()
        return 0.0


def main():
    # 加载配置
    config_path = "configs/letter_high_optimized.yaml"  # 可根据需要更换

    # 使用OmegaConf直接加载配置文件
    config = OmegaConf.load(config_path)

    # 处理时间戳等动态变量
    from datetime import datetime

    if config.timestamp == "auto":
        config.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.run_name == "letter_high_poincare_auto":
        config.run_name = f"{config.exp_name}_{config.timestamp}"

    config.vae.train.num_epochs = 500  # 训练50轮
    config.vae.train.test_interval = 999  # 设置一个很大的值，避免训练时评估

    # 训练VAE
    result = train_vae(config)
    print(f"训练完成，最佳模型保存在: {result['best_checkpoint']}")

    # 加载最佳模型
    checkpoint = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)

    # 获取设备信息
    from utils.loader import load_device

    device = load_device(config)

    # 直接从checkpoint中重建模型
    from types import SimpleNamespace
    from models.GraphVAE import GraphVAE

    # 重建VAE配置
    saved_config = checkpoint["model_config"]
    vae_config = SimpleNamespace()
    vae_config.pred_node_class = saved_config["vae"]["loss"]["pred_node_class"]
    vae_config.pred_edge = saved_config["vae"]["loss"]["pred_edge"]
    vae_config.use_kl_loss = saved_config["vae"]["loss"]["use_kl_loss"]
    vae_config.use_base_proto_loss = saved_config["vae"]["loss"]["use_base_proto_loss"]
    vae_config.use_sep_proto_loss = saved_config["vae"]["loss"]["use_sep_proto_loss"]

    # 重建encoder_config，确保包含所有必要字段
    encoder_config_dict = saved_config["vae"]["encoder"]
    encoder_config_dict["input_feature_dim"] = saved_config["data"]["max_feat_num"]
    vae_config.encoder_config = SimpleNamespace(**encoder_config_dict)

    # 重建decoder_config
    decoder_config_dict = saved_config["vae"]["decoder"]
    decoder_config_dict["latent_feature_dim"] = saved_config["vae"]["encoder"]["latent_feature_dim"]
    decoder_config_dict["output_feature_dim"] = saved_config["data"]["max_feat_num"]
    vae_config.decoder_config = SimpleNamespace(**decoder_config_dict)

    vae_config.latent_dim = saved_config["vae"]["encoder"]["latent_feature_dim"]
    vae_config.device = device

    # 创建模型并加载权重
    model = GraphVAE(vae_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    encoder = model.encoder

    # Meta-eval: 支持集和查询集acc
    from utils.data_utils import MyDataset

    dataset = MyDataset(config.data, config.fsl_task)

    print("\n[Meta-eval] 训练集任务微调后评估...")

    # 使用训练集任务进行微调和评估
    N_way = config.fsl_task.N_way
    K_shot = config.fsl_task.K_shot
    R_query = config.fsl_task.R_query

    # 采样一个训练集任务
    task = dataset.sample_one_task(
        is_train=True,
        N_way=N_way,
        K_shot=K_shot,
        R_query=R_query,
    )

    if task is not None:
        # 用原型网络方法分别评估支持集和查询集
        support_acc, query_acc = evaluate_task_with_prototypical_networks(
            encoder, task, config, device
        )
        print(f"训练集任务原型网络评估:")
        print(f"  支持集准确率: {support_acc:.4f}")
        print(f"  查询集准确率: {query_acc:.4f}")
    else:
        print("无法采样到训练集任务")


if __name__ == "__main__":
    main()
