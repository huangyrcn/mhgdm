#!/usr/bin/env python3
"""
三阶段训练运行脚本
自动化执行：VAE训练 -> Score训练 -> Meta-test
"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path


def find_best_checkpoint(checkpoint_dir, pattern="best*.pth"):
    """查找最佳检查点"""
    search_pattern = os.path.join(checkpoint_dir, "**", pattern)
    checkpoints = glob.glob(search_pattern, recursive=True)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found matching {search_pattern}")

    # 返回最新的检查点
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n{'='*60}")
    print(f"🎯 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"❌ {description} failed with return code {result.returncode}")
        return False

    print(f"✅ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="三阶段训练运行脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="只运行特定阶段")
    parser.add_argument("--vae_checkpoint", type=str, help="VAE检查点路径（用于阶段2和3）")
    parser.add_argument("--score_checkpoint", type=str, help="Score检查点路径（用于阶段3）")

    args = parser.parse_args()

    config_path = args.config

    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False

    print(f"🚀 开始三阶段训练实验")
    print(f"配置文件: {config_path}")

    vae_checkpoint = args.vae_checkpoint
    score_checkpoint = args.score_checkpoint

    # 阶段一：VAE训练
    if args.stage is None or args.stage == 1:
        cmd = ["python", "vae_trainer_simple.py", "--config", config_path]

        if not run_command(cmd, "阶段一：VAE训练"):
            return False

        # 查找VAE检查点
        if not vae_checkpoint:
            try:
                # 从配置中读取保存目录
                import yaml

                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

                exp_name = config["exp_name"]
                save_dir = config["paths"]["save_dir"]

                checkpoint_dir = os.path.join(save_dir, exp_name)
                vae_checkpoint = find_best_checkpoint(checkpoint_dir, "best.pth")
                print(f"✓ 找到VAE检查点: {vae_checkpoint}")

            except Exception as e:
                print(f"❌ 无法找到VAE检查点: {e}")
                return False

    # 阶段二：Score训练
    if args.stage is None or args.stage == 2:
        if not vae_checkpoint:
            print(f"❌ 需要VAE检查点路径进行Score训练")
            return False

        cmd = [
            "python",
            "score_trainer_simple.py",
            "--config",
            config_path,
            "--vae_checkpoint",
            vae_checkpoint,
        ]

        if not run_command(cmd, "阶段二：Score训练"):
            return False

        # 查找Score检查点
        if not score_checkpoint:
            try:
                # 从配置中读取保存目录
                import yaml

                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

                exp_name = config["exp_name"]
                save_dir = config["paths"]["save_dir"]

                checkpoint_dir = os.path.join(save_dir, f"{exp_name}_score")
                score_checkpoint = find_best_checkpoint(checkpoint_dir, "best*.pth")
                print(f"✓ 找到Score检查点: {score_checkpoint}")

            except Exception as e:
                print(f"❌ 无法找到Score检查点: {e}")
                return False

    # 阶段三：Meta-test
    if args.stage is None or args.stage == 3:
        if not vae_checkpoint:
            print(f"❌ 需要VAE检查点路径进行Meta-test")
            return False

        cmd = [
            "python",
            "meta_test_simple.py",
            "--config",
            config_path,
            "--vae_checkpoint",
            vae_checkpoint,
        ]

        if score_checkpoint:
            cmd.extend(["--score_checkpoint", score_checkpoint])

        if not run_command(cmd, "阶段三：Meta-test"):
            return False

    print(f"\n🎉 三阶段训练实验完成!")
    print(f"VAE检查点: {vae_checkpoint}")
    if score_checkpoint:
        print(f"Score检查点: {score_checkpoint}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
