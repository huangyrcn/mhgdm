"""
MetaTaskSampler 使用示例
展示如何使用ControlNet风格的适应器进行元测试任务的支持集扩充
"""

import torch
import types
from utils.task_sampler import create_meta_task_sampler
from utils.data_utils import MyDataset
from utils.config_utils import load_config


def create_example_config():
    """创建示例配置"""
    config = types.SimpleNamespace()

    # 设备配置
    config.device = "auto"
    config.device_count = 1

    # 数据配置
    config.data = types.SimpleNamespace()
    config.data.name = "Letter_high"
    config.data.max_node_num = 9
    config.data.max_feat_num = 5
    config.data.degree_as_tag = True

    # FSL任务配置
    config.fsl_task = types.SimpleNamespace()
    config.fsl_task.N_way = 4
    config.fsl_task.K_shot = 1
    config.fsl_task.R_query = 8

    # 采样器配置
    config.sampler = types.SimpleNamespace()
    config.sampler.predictor = "reverse_diffusion"
    config.sampler.corrector = "none"
    config.sampler.snr_x = 0.16
    config.sampler.snr_A = 0.16
    config.sampler.scale_eps_x = 1.0
    config.sampler.scale_eps_A = 1.0

    # 采样配置
    config.sample = types.SimpleNamespace()
    config.sample.use_ema = True
    config.sample.k_augment = 2  # 每个支持样本生成2个增强样本

    # 检查点路径（需要根据实际情况修改）
    config.score_ckpt_path = "checkpoints/score_model.pt"
    config.encoder_ckpt_path = "checkpoints/encoder_model.pt"

    return config


def example_task_augmentation():
    """
    示例：如何使用MetaTaskSampler进行任务增强
    """
    print("🚀 MetaTaskSampler 使用示例")

    # 1. 创建配置
    config = create_example_config()
    print(f"✓ 配置创建完成")

    # 2. 创建数据集并采样一个元测试任务
    try:
        dataset = MyDataset(config.data, config.fsl_task)
        print(
            f"✓ 数据集加载完成: 训练图{len(dataset.train_nx_graphs)}, 测试图{len(dataset.test_nx_graphs)}"
        )

        # 采样一个测试任务
        task = dataset.sample_one_task(
            is_train=False,
            N_way=config.fsl_task.N_way,
            K_shot=config.fsl_task.K_shot,
            R_query=config.fsl_task.R_query,
        )

        if task is None:
            print("❌ 无法采样测试任务")
            return

        print(f"✓ 采样测试任务完成:")
        print(f"  支持集大小: {task['support_set']['x'].shape}")
        print(f"  查询集大小: {task['query_set']['x'].shape}")
        print(f"  N_way: {task.get('N_way', 'Unknown')}")
        print(f"  K_shot: {task.get('K_shot', 'Unknown')}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        # 创建模拟任务用于演示
        task = create_mock_task(config)
        print("✓ 创建模拟任务用于演示")

    # 3. 创建MetaTaskSampler（如果检查点文件不存在，这里会失败）
    try:
        task_sampler = create_meta_task_sampler(
            config, config.score_ckpt_path, config.encoder_ckpt_path
        )
        print("✓ MetaTaskSampler 创建成功")

        # 4. 增强任务
        print("\n🔧 开始任务增强...")
        augmented_task = task_sampler.augment_task(
            task,
            k_augment=config.sample.k_augment,
            finetune_steps=5,  # 为了演示，使用较少的微调步数
            learning_rate=1e-3,
        )

        print(f"✅ 任务增强完成:")
        print(f"  原始支持集大小: {task['support_set']['x'].shape}")
        print(f"  增强后支持集大小: {augmented_task['support_set']['x'].shape}")
        print(f"  查询集大小保持不变: {augmented_task['query_set']['x'].shape}")

        # 5. 分析增强效果
        analyze_augmentation_results(task, augmented_task)

    except Exception as e:
        print(f"❌ MetaTaskSampler 创建失败: {e}")
        print("💡 这通常是因为检查点文件不存在，这是正常的演示情况")

        # 演示任务增强的概念
        demonstrate_augmentation_concept(task, config)


def create_mock_task(config):
    """创建模拟任务用于演示"""
    N_way = config.fsl_task.N_way
    K_shot = config.fsl_task.K_shot
    R_query = config.fsl_task.R_query

    # 创建模拟支持集
    support_x = torch.randn(N_way * K_shot, config.data.max_node_num, config.data.max_feat_num)
    support_adj = torch.rand(N_way * K_shot, config.data.max_node_num, config.data.max_node_num)
    support_adj = (support_adj + support_adj.transpose(-1, -2)) / 2  # 对称化
    support_adj = (support_adj > 0.5).float()  # 二值化
    support_labels = torch.repeat_interleave(torch.arange(N_way), K_shot)

    # 创建模拟查询集
    query_x = torch.randn(R_query, config.data.max_node_num, config.data.max_feat_num)
    query_adj = torch.rand(R_query, config.data.max_node_num, config.data.max_node_num)
    query_adj = (query_adj + query_adj.transpose(-1, -2)) / 2
    query_adj = (query_adj > 0.5).float()
    query_labels = torch.randint(0, N_way, (R_query,))

    task = {
        "support_set": {"x": support_x, "adj": support_adj, "label": support_labels},
        "query_set": {"x": query_x, "adj": query_adj, "label": query_labels},
        "N_way": N_way,
        "K_shot": K_shot,
        "R_query": R_query,
    }

    return task


def demonstrate_augmentation_concept(task, config):
    """演示任务增强的概念（不使用实际的采样器）"""
    print("\n💡 任务增强概念演示:")

    original_support_size = task["support_set"]["x"].shape[0]
    k_augment = config.sample.k_augment
    expected_augmented_size = original_support_size * (1 + k_augment)

    print(f"1. 原始支持集大小: {original_support_size}")
    print(f"2. 增强倍数 (k_augment): {k_augment}")
    print(f"3. 预期增强后大小: {expected_augmented_size}")

    print(f"\n📋 增强流程:")
    print(f"  步骤1: 计算支持集的类别原型 ({task['N_way']}个类别)")
    print(f"  步骤2: 基于原型微调ControlNet风格的适应器")
    print(f"  步骤3: 使用适应器生成 {original_support_size * k_augment} 个增强样本")
    print(f"  步骤4: 合并原始样本和增强样本")

    print(f"\n🎯 关键优势:")
    print(f"  • 主模型保持冻结，不影响预训练权重")
    print(f"  • 适应器轻量级，快速适应新任务")
    print(f"  • 基于任务特定原型的引导生成")
    print(f"  • 保持FSL任务的语义一致性")


def analyze_augmentation_results(original_task, augmented_task):
    """分析增强结果"""
    print(f"\n📊 增强结果分析:")

    original_size = original_task["support_set"]["x"].shape[0]
    augmented_size = augmented_task["support_set"]["x"].shape[0]

    print(f"支持集扩充倍数: {augmented_size / original_size:.1f}x")

    # 分析标签分布
    original_labels = original_task["support_set"]["label"]
    augmented_labels = augmented_task["support_set"]["label"]

    print(f"原始标签分布: {torch.bincount(original_labels).tolist()}")
    print(f"增强后标签分布: {torch.bincount(augmented_labels).tolist()}")

    # 计算数据统计
    original_x_mean = original_task["support_set"]["x"].mean()
    augmented_x_mean = augmented_task["support_set"]["x"].mean()

    print(f"特征均值变化: {original_x_mean:.4f} → {augmented_x_mean:.4f}")


def demonstrate_controlnet_design():
    """演示ControlNet风格设计的优势"""
    print("\n🏗️  ControlNet风格设计说明:")

    print("1. 主模型冻结:")
    print("   • 预训练的分数网络权重保持不变")
    print("   • 避免灾难性遗忘")
    print("   • 保持原有的生成能力")

    print("\n2. 轻量级适应器:")
    print("   • 只有少量可训练参数")
    print("   • 快速微调（几个step即可）")
    print("   • 基于任务原型进行调制")

    print("\n3. 原型引导生成:")
    print("   • 从支持集计算类别原型")
    print("   • 原型包含任务特定的语义信息")
    print("   • 引导生成与任务相关的样本")

    print("\n4. 无缝集成:")
    print("   • 与现有的FSL框架兼容")
    print("   • 不破坏原有的训练流程")
    print("   • 可插拔的增强模块")


if __name__ == "__main__":
    # 运行示例
    example_task_augmentation()

    # 演示设计概念
    demonstrate_controlnet_design()

    print("\n🎉 示例完成！")
    print("💡 要实际使用此功能，请确保有有效的score和encoder检查点文件。")
