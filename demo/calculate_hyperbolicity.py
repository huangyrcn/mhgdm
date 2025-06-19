#!/usr/bin/env python3
"""
计算四个数据集的双曲性：ENZYMES, Letter_high, Reddit, TRIANGLES
"""

import os
import sys
import time
import yaml
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接在这里定义hyperbolicity_sample函数，避免导入问题
def hyperbolicity_sample(G, num_samples=50000):
    """计算图的双曲性采样"""
    curr_time = time.time()
    hyps = []
    for i in tqdm(range(num_samples), desc="计算双曲性"):
        try:
            node_tuple = np.random.choice(list(G.nodes()), 4, replace=False)
            s = []
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    
    ok = True
    if len(hyps) == 0:
        ok = False
        return ok, None
    return ok, max(hyps)


class SimpleConfig:
    """简单的配置类，用于加载数据集配置"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def load_dataset_config(dataset_name: str) -> SimpleConfig:
    """加载数据集配置文件"""
    config_path = Path(f"./configs/data/{dataset_name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return SimpleConfig(**config_dict)


def load_dataset_graphs_simple(dataset_name: str) -> List[nx.Graph]:
    """简单的数据集加载函数，直接读取txt文件"""
    print(f"正在加载数据集: {dataset_name}")
    
    dataset_path = Path(f"./datasets/{dataset_name}/{dataset_name}.txt")
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    graphs = []
    
    with open(dataset_path, 'r') as f:
        num_graphs = int(f.readline().strip())
        print(f"数据集包含 {num_graphs} 个图")
        
        for graph_idx in range(num_graphs):
            # 读取图的元信息
            meta_line = f.readline().strip().split()
            num_nodes = int(meta_line[0])
            graph_label = int(meta_line[1])
            
            # 创建图
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            
            # 读取节点信息和边
            for node_i in range(num_nodes):
                node_line = f.readline().strip().split()
                node_tag = int(node_line[0])
                num_neighbors = int(node_line[1])
                
                # 添加边
                for k in range(num_neighbors):
                    neighbor_id = int(node_line[2 + k])
                    if node_i < neighbor_id:  # 避免重复添加边
                        G.add_edge(node_i, neighbor_id)
            
            # 设置图标签
            G.graph['label'] = graph_label
            graphs.append(G)
            
            if (graph_idx + 1) % 100 == 0:
                print(f"已加载 {graph_idx + 1}/{num_graphs} 个图")
    
    print(f"成功加载 {dataset_name}: {len(graphs)} 个图")
    return graphs


def calculate_dataset_hyperbolicity(
    graph_list: List[nx.Graph], 
    dataset_name: str, 
    num_samples: int = 5000
) -> Dict[str, Any]:
    """计算数据集的双曲性"""
    print(f"\n开始计算 {dataset_name} 的双曲性...")
    print(f"图数量: {len(graph_list)}, 每图采样数: {num_samples}")
    
    start_time = time.time()
    hyp_values = []
    processed_graphs = 0
    failed_graphs = 0
    
    for i, graph in enumerate(tqdm(graph_list, desc=f"处理 {dataset_name}")):
        try:
            # 检查图的连通性
            if not nx.is_connected(graph):
                # 如果不连通，取最大连通分量
                largest_cc = max(nx.connected_components(graph), key=len)
                graph = graph.subgraph(largest_cc).copy()
            
            # 确保图至少有4个节点才能计算双曲性
            if graph.number_of_nodes() < 4:
                failed_graphs += 1
                continue
            
            # 对于大图，减少采样数量以提高速度
            if graph.number_of_nodes() > 1000:
                samples = min(num_samples, 2000)
            else:
                samples = num_samples
            
            ok, hyp_value = hyperbolicity_sample(graph, samples)
            
            if ok and hyp_value is not None:
                hyp_values.append(hyp_value)
                processed_graphs += 1
                if i % 10 == 0:  # 每10个图输出一次进度
                    tqdm.write(f"图 {i+1}: 双曲性 = {hyp_value:.4f}")
            else:
                failed_graphs += 1
                
        except Exception as e:
            failed_graphs += 1
            tqdm.write(f"处理图 {i+1} 时出错: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # 计算统计信息
    if hyp_values:
        mean_hyp = np.mean(hyp_values)
        std_hyp = np.std(hyp_values)
        min_hyp = np.min(hyp_values)
        max_hyp = np.max(hyp_values)
        median_hyp = np.median(hyp_values)
    else:
        mean_hyp = std_hyp = min_hyp = max_hyp = median_hyp = None
    
    results = {
        'dataset_name': dataset_name,
        'total_graphs': len(graph_list),
        'processed_graphs': processed_graphs,
        'failed_graphs': failed_graphs,
        'hyperbolicity_values': hyp_values,
        'mean_hyperbolicity': mean_hyp,
        'std_hyperbolicity': std_hyp,
        'min_hyperbolicity': min_hyp,
        'max_hyperbolicity': max_hyp,
        'median_hyperbolicity': median_hyp,
        'computation_time': total_time,
        'samples_per_graph': num_samples
    }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """打印计算结果"""
    dataset_name = results['dataset_name']
    print(f"\n{'='*50}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*50}")
    print(f"总图数量: {results['total_graphs']}")
    print(f"成功处理: {results['processed_graphs']}")
    print(f"失败数量: {results['failed_graphs']}")
    print(f"计算时间: {results['computation_time']:.2f} 秒")
    
    if results['mean_hyperbolicity'] is not None:
        print(f"\n双曲性统计:")
        print(f"  平均值: {results['mean_hyperbolicity']:.4f}")
        print(f"  标准差: {results['std_hyperbolicity']:.4f}")
        print(f"  最小值: {results['min_hyperbolicity']:.4f}")
        print(f"  最大值: {results['max_hyperbolicity']:.4f}")
        print(f"  中位数: {results['median_hyperbolicity']:.4f}")
    else:
        print("未能计算出有效的双曲性值")


def save_results(all_results: List[Dict[str, Any]], output_file: str) -> None:
    """保存结果到文件"""
    # 准备要保存的数据（移除不能序列化的numpy数组）
    results_to_save = []
    for result in all_results:
        result_copy = result.copy()
        # 将numpy数组转换为列表
        if 'hyperbolicity_values' in result_copy:
            result_copy['hyperbolicity_values'] = [float(x) for x in result_copy['hyperbolicity_values']]
        results_to_save.append(result_copy)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='计算数据集的双曲性')
    parser.add_argument('--datasets', nargs='+', 
                       default=['ENZYMES', 'Letter_high', 'Reddit', 'TRIANGLES'],
                       help='要计算的数据集名称')
    parser.add_argument('--samples', type=int, default=5000,
                       help='每个图的采样数量')
    parser.add_argument('--output', type=str, default='hyperbolicity_results.json',
                       help='输出文件名')
    
    args = parser.parse_args()
    
    print("开始计算数据集双曲性...")
    print(f"目标数据集: {args.datasets}")
    print(f"每图采样数: {args.samples}")
    
    all_results = []
    
    for dataset_name in args.datasets:
        try:
            # 加载数据集
            graph_list = load_dataset_graphs_simple(dataset_name)
            
            # 计算双曲性
            results = calculate_dataset_hyperbolicity(
                graph_list, dataset_name, args.samples
            )
            
            # 打印结果
            print_results(results)
            
            all_results.append(results)
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")
            continue
    
    # 保存所有结果
    if all_results:
        save_results(all_results, args.output)
        
        # 打印总结
        print(f"\n{'='*60}")
        print("总结")
        print(f"{'='*60}")
        for result in all_results:
            name = result['dataset_name']
            mean_hyp = result['mean_hyperbolicity']
            if mean_hyp is not None:
                print(f"{name:12}: 平均双曲性 = {mean_hyp:.4f}")
            else:
                print(f"{name:12}: 计算失败")
    else:
        print("未能成功处理任何数据集")


if __name__ == '__main__':
    main()
