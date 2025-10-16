#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Road Marking Classifier - Usage Examples
道路標示分類システム 使用例
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from legacy import EnhancedWhiteLineExtractor


def example_basic_usage():
    """
    基本的な使用例
    """
    print("=== 例1: 基本的な使用方法 ===")
    
    # 入力・出力ファイルパス
    input_file = "sample_intersection.pcd"  # あなたのPCDファイル
    output_file = "classified_markings.dxf"
    
    # 分類システムを初期化
    extractor = EnhancedWhiteLineExtractor()
    
    # 処理実行
    if os.path.exists(input_file):
        result = extractor.process_pcd_file(input_file, output_file)
        
        if result:
            print("✅ 処理完了!")
            print(f"横断歩道: {len(result['crosswalks'])}個")
            print(f"停止線: {len(result['stop_lines'])}本")
            print(f"車線: {len(result['lanes'])}本")
    else:
        print(f"⚠️ サンプルファイル {input_file} が見つかりません")


def example_custom_config():
    """
    カスタム設定を使用した例
    """
    print("\n=== 例2: カスタム設定使用 ===")
    
    # カスタム設定ファイル
    config_file = "custom_config.json"
    
    # カスタム設定を作成
    custom_config = {
        "hsv": {
            "s_range": [0, 40],    # 彩度範囲を広げる
            "v_range": [150, 255]  # 明度閾値を下げる
        },
        "classification": {
            "crosswalk_min_aspect_ratio": 2.0,  # より厳格な横断歩道判定
            "stop_line_angle_tolerance": 10.0   # より厳格な角度判定
        }
    }
    
    # 設定ファイルに保存
    import json
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(custom_config, f, indent=2, ensure_ascii=False)
    
    # カスタム設定で初期化
    extractor = EnhancedWhiteLineExtractor(config_path=config_file)
    print(f"✅ カスタム設定で初期化: {config_file}")
    
    # 後処理でファイル削除
    if os.path.exists(config_file):
        os.remove(config_file)


def example_batch_processing():
    """
    バッチ処理の例
    """
    print("\n=== 例3: バッチ処理 ===")
    
    # 複数ファイルのリスト
    input_files = [
        "intersection_001.pcd",
        "intersection_002.pcd", 
        "intersection_003.pcd"
    ]
    
    # 分類システムを初期化（一度だけ）
    extractor = EnhancedWhiteLineExtractor()
    
    results = []
    
    for i, input_file in enumerate(input_files):
        output_file = f"result_{i+1:03d}.dxf"
        
        if os.path.exists(input_file):
            print(f"処理中: {input_file} -> {output_file}")
            result = extractor.process_pcd_file(input_file, output_file)
            results.append(result)
        else:
            print(f"⚠️ ファイルが見つかりません: {input_file}")
    
    # 統計情報を表示
    if results:
        total_crosswalks = sum(len(r['crosswalks']) for r in results if r)
        total_stop_lines = sum(len(r['stop_lines']) for r in results if r)
        total_lanes = sum(len(r['lanes']) for r in results if r)
        
        print(f"\n📊 バッチ処理統計:")
        print(f"処理ファイル数: {len([r for r in results if r])}")
        print(f"総横断歩道数: {total_crosswalks}")
        print(f"総停止線数: {total_stop_lines}")
        print(f"総車線数: {total_lanes}")


def example_advanced_analysis():
    """
    高度な解析例
    """
    print("\n=== 例4: 高度な解析 ===")
    
    import numpy as np
    
    # 分類システムを初期化
    extractor = EnhancedWhiteLineExtractor()
    
    # 仮想的な解析
    sample_results = [
        {'crosswalks': [1, 2], 'stop_lines': [1, 2, 3], 'lanes': [1, 2, 3, 4]},
        {'crosswalks': [1], 'stop_lines': [1, 2], 'lanes': [1, 2, 3]},
        {'crosswalks': [1, 2, 3], 'stop_lines': [1, 2, 3, 4], 'lanes': [1, 2, 3, 4, 5]}
    ]
    
    # 統計解析
    crosswalk_counts = [len(r['crosswalks']) for r in sample_results]
    stop_line_counts = [len(r['stop_lines']) for r in sample_results]
    lane_counts = [len(r['lanes']) for r in sample_results]
    
    print(f"📈 統計解析結果:")
    print(f"横断歩道 - 平均: {np.mean(crosswalk_counts):.1f}, 標準偏差: {np.std(crosswalk_counts):.1f}")
    print(f"停止線 - 平均: {np.mean(stop_line_counts):.1f}, 標準偏差: {np.std(stop_line_counts):.1f}")
    print(f"車線 - 平均: {np.mean(lane_counts):.1f}, 標準偏差: {np.std(lane_counts):.1f}")


def example_error_handling():
    """
    エラーハンドリングの例
    """
    print("\n=== 例5: エラーハンドリング ===")
    
    extractor = EnhancedWhiteLineExtractor()
    
    # 存在しないファイルでテスト
    result = extractor.process_pcd_file("nonexistent.pcd", "output.dxf")
    
    if result is None:
        print("✅ 適切にエラーが処理されました")
    
    # 設定の検証
    try:
        # 無効な設定でテスト
        invalid_config = "invalid_config.json"
        with open(invalid_config, 'w') as f:
            f.write("invalid json content")
        
        extractor_with_invalid_config = EnhancedWhiteLineExtractor(config_path=invalid_config)
        print("✅ 無効な設定ファイルが適切に処理されました")
        
        # クリーンアップ
        os.remove(invalid_config)
        
    except Exception as e:
        print(f"設定エラー: {e}")


def main():
    """
    全ての例を実行
    """
    print("🚗 Road Marking Classifier - 使用例")
    print("="*50)
    
    try:
        example_basic_usage()
        example_custom_config() 
        example_batch_processing()
        example_advanced_analysis()
        example_error_handling()
        
        print("\n🎉 全ての使用例が完了しました！")
        print("\n📚 詳細な情報:")
        print("- README.md: プロジェクトの概要")
        print("- configs/default_config.json: 設定パラメータ")
        print("- tests/manual_test.py: テストスクリプト")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
