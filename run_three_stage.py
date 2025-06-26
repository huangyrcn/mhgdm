#!/usr/bin/env python3
"""
三阶段训练系统主程序
"""

import argparse
import os
import sys
import yaml
from omegaconf import OmegaConf
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vae_trainer import train_vae
from score_trainer import train_score
from meta_test import run_meta_test


def print_stage_header(title):
    """打印阶段标题"""
    print(f"\n{'='*80}")
    print(f"🚀 {title}")
    print(f"{'='*80}")


def ensure_wandb_closed():
    """确保wandb正确关闭"""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except:
        pass


def run_three_stage_training(config):
    """运行完整的三阶段训练"""
    print_stage_header("开始三阶段训练流程")

    # 第一阶段：VAE训练
    print_stage_header("第一阶段：VAE训练")
    vae_result = train_vae(config)
    ensure_wandb_closed()

    if not vae_result or "best_checkpoint" not in vae_result:
        print("❌ VAE训练失败")
        return False

    vae_checkpoint = vae_result["best_checkpoint"]
    print(f"✅ VAE训练完成")
    print(f"   最佳检查点: {vae_checkpoint}")

    # 安全的格式化best_meta_test_acc
    best_meta_test_acc = vae_result.get("best_meta_test_acc", "N/A")
    if isinstance(best_meta_test_acc, (int, float)):
        print(f"   训练过程中最佳meta-test准确率: {best_meta_test_acc:.4f}")
    else:
        print(f"   训练过程中最佳meta-test准确率: {best_meta_test_acc}")

    # 第二阶段：Score模型训练
    print_stage_header("第二阶段：Score模型训练")
    score_result = train_score(config, vae_checkpoint)
    ensure_wandb_closed()

    if not score_result or "best_checkpoint" not in score_result:
        print("❌ Score训练失败")
        return False

    score_checkpoint = score_result["best_checkpoint"]
    print(f"✅ Score训练完成")
    print(f"   最佳检查点: {score_checkpoint}")

    # 第三阶段：Meta-test评估
    print_stage_header("第三阶段：增强Meta-test评估")
    ensure_wandb_closed()
    checkpoint_paths = {"vae_checkpoint": vae_checkpoint, "score_checkpoint": score_checkpoint}

    meta_result = run_meta_test(
        config=config, use_augmentation=True, checkpoint_paths=checkpoint_paths  # 使用数据增强
    )
    ensure_wandb_closed()

    if not meta_result:
        print("❌ Meta-test失败")
        return False

    print(f"✅ 增强Meta-test完成")
    print(f"   最终准确率: {meta_result.get('accuracy', 0):.4f}")
    print(f"   最终F1分数: {meta_result.get('f1', 0):.4f}")
    print(f"   完成任务数: {meta_result.get('num_tasks', 0)}")

    # 总结结果
    print_stage_header("训练流程总结")
    print("🎯 三阶段训练完成！")
    print("\n📊 最终结果对比:")

    # 安全的格式化输出
    vae_meta_test_acc = vae_result.get("best_meta_test_acc", 0)
    final_acc = meta_result.get("accuracy", 0)

    if isinstance(vae_meta_test_acc, (int, float)):
        print(f"   VAE训练中meta-test: {vae_meta_test_acc:.4f}")
    else:
        print(f"   VAE训练中meta-test: {vae_meta_test_acc}")

    print(f"   增强meta-test:      {final_acc:.4f}")

    if isinstance(vae_meta_test_acc, (int, float)) and isinstance(final_acc, (int, float)):
        improvement = final_acc - vae_meta_test_acc
        print(f"   性能提升:           {improvement:+.4f}")

    print("\n🎉 三阶段训练流程成功完成！")
    return True


def run_specific_stage(config, stage):
    """运行特定阶段"""
    if stage == 1:
        # VAE阶段
        print_stage_header("运行VAE训练")
        result = train_vae(config)
        ensure_wandb_closed()
        if result:
            print(f"✅ VAE训练完成，检查点: {result.get('best_checkpoint', 'N/A')}")
        else:
            print("❌ VAE训练失败")

    elif stage == 2:
        # Score阶段 - 需要VAE检查点
        print_stage_header("运行Score模型训练")

        # 查找VAE检查点
        vae_checkpoint = None
        if hasattr(config, "paths") and hasattr(config.paths, "vae_checkpoint"):
            vae_checkpoint = config.paths.vae_checkpoint

        if not vae_checkpoint or not os.path.exists(vae_checkpoint):
            print("❌ 未找到VAE检查点，请先运行阶段1或在配置中指定vae_checkpoint路径")
            return False

        result = train_score(config, vae_checkpoint)
        ensure_wandb_closed()
        if result:
            print(f"✅ Score训练完成，检查点: {result.get('best_checkpoint', 'N/A')}")
        else:
            print("❌ Score训练失败")

    elif stage == 3:
        # Meta-test阶段 - 需要两个检查点
        print_stage_header("运行Meta-test评估")

        # 查找检查点
        vae_checkpoint = (
            getattr(config.paths, "vae_checkpoint", None) if hasattr(config, "paths") else None
        )
        score_checkpoint = (
            getattr(config.paths, "score_checkpoint", None) if hasattr(config, "paths") else None
        )

        if not vae_checkpoint or not os.path.exists(vae_checkpoint):
            print("❌ 未找到VAE检查点，请在配置中指定vae_checkpoint路径")
            return False

        if not score_checkpoint or not os.path.exists(score_checkpoint):
            print("❌ 未找到Score检查点，请在配置中指定score_checkpoint路径")
            return False

        ensure_wandb_closed()
        checkpoint_paths = {"vae_checkpoint": vae_checkpoint, "score_checkpoint": score_checkpoint}

        result = run_meta_test(
            config=config, use_augmentation=True, checkpoint_paths=checkpoint_paths
        )
        ensure_wandb_closed()

        if result:
            print(f"✅ Meta-test完成")
            print(f"   准确率: {result.get('accuracy', 0):.4f}")
            print(f"   F1分数: {result.get('f1', 0):.4f}")
        else:
            print("❌ Meta-test失败")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="三阶段训练系统")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3], help="运行特定阶段 (1=VAE, 2=Score, 3=Meta-test)"
    )

    args = parser.parse_args()

    # 加载配置并处理时间戳
    config = OmegaConf.load(args.config)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置时间戳
    config.timestamp = timestamp

    # 手动解析一些关键的插值变量
    # 解析 run_name
    config.run_name = f"{config.exp_name}_{config.timestamp}"

    # 解析 paths 中的变量
    if hasattr(config, "paths"):
        if hasattr(config.paths, "base_save_dir"):
            base_save_dir = config.paths.base_save_dir
            config.paths.vae_save_dir = f"{base_save_dir}/{config.run_name}/vae"
            config.paths.score_save_dir = f"{base_save_dir}/{config.run_name}/score"

    # 设置环境变量
    os.environ["WANDB_MODE"] = "offline"

    if args.stage:
        # 运行特定阶段
        run_specific_stage(config, args.stage)
    else:
        # 运行完整流程
        success = run_three_stage_training(config)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
