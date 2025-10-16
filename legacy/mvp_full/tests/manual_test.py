#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Road Marking Classifier - Test Script
道路標示分類システム テストスクリプト
"""

import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import shutil

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from legacy import EnhancedWhiteLineExtractor

CONFIG_PATH = ROOT_DIR / "configs" / "default_config.json"


def test_with_real_data():
    """
    実際のresearch4.ipynbデータを使用したテスト
    """
    print("="*60)
    print("実際のデータを使用したテスト実行")
    print("="*60)
    
    # 元データのパス
    original_data_dir = r"d:\MyWorkspace\PythonProjects\Trust_Project02\test_data\test_data01"
    
    # 使用可能な点群ファイルを確認
    possible_files = [
        "cropped_intersection.pcd",
        "cropped_intersection_color.pcd", 
        "cropped_intersection_neo.pcd",
        "merged_output.pcd",
        "merged_point_cloud.ply"
    ]
    
    test_file = None
    for filename in possible_files:
        filepath = os.path.join(original_data_dir, filename)
        if os.path.exists(filepath):
            print(f"✅ 発見: {filename}")
            test_file = filepath
            break
    
    if not test_file:
        print("❌ 使用可能な点群ファイルが見つかりません。")
        print("以下の場所を確認してください:")
        print(f"  {original_data_dir}")
        return False
    
    # テスト実行
    print(f"\n📂 使用するファイル: {os.path.basename(test_file)}")
    
    # 出力ファイル名
    output_file = f"real_data_test_result.dxf"
    
    try:
        # 分類システムを初期化
        extractor = EnhancedWhiteLineExtractor(config_path=str(CONFIG_PATH))
        
        # 処理実行
        result = extractor.process_pcd_file(test_file, output_file)
        
        if result:
            print(f"\n✅ 実データテスト成功！")
            print(f"結果ファイル: {output_file}")
            print(f"検出結果:")
            print(f"  - 横断歩道: {len(result['crosswalks'])}個")
            print(f"  - 停止線: {len(result['stop_lines'])}本")
            print(f"  - 車線: {len(result['lanes'])}本")
            print(f"  - 縁石: {len(result['curb_lines'])}本")
            print(f"  - 処理点数: {result.get('road_surface_points', 0)}点")
            
            # ファイルサイズ確認
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"  - 出力ファイルサイズ: {file_size:,} bytes")
            
            return True
        else:
            print(f"\n❌ 実データテスト失敗")
            return False
            
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_point_cloud():
    """
    テスト用の点群データを生成
    """
    print("テスト用点群データを生成中...")
    
    # 道路面の点群（灰色）
    road_points = []
    road_colors = []
    
    # 50m x 30m の道路面を生成
    for x in np.linspace(0, 50, 500):
        for y in np.linspace(0, 30, 300):
            z = 0.0 + np.random.normal(0, 0.02)  # 軽微なノイズ
            road_points.append([x, y, z])
            road_colors.append([0.3, 0.3, 0.3])  # 灰色（アスファルト）
    
    # 白線（車線）を追加
    for x in np.linspace(5, 45, 200):
        # 中央線
        y = 15.0
        z = 0.01 + np.random.normal(0, 0.01)
        road_points.append([x, y, z])
        road_colors.append([0.9, 0.9, 0.9])  # 白色
        
        # 左側線
        y = 5.0
        z = 0.01 + np.random.normal(0, 0.01)
        road_points.append([x, y, z])
        road_colors.append([0.9, 0.9, 0.9])  # 白色
        
        # 右側線
        y = 25.0
        z = 0.01 + np.random.normal(0, 0.01)
        road_points.append([x, y, z])
        road_colors.append([0.9, 0.9, 0.9])  # 白色
    
    # 横断歩道を追加
    crosswalk_x = 25.0
    for stripe in range(5):  # 5本の縞
        stripe_y_start = 8 + stripe * 3
        stripe_y_end = stripe_y_start + 1.5
        
        for x in np.linspace(crosswalk_x - 1, crosswalk_x + 1, 20):
            for y in np.linspace(stripe_y_start, stripe_y_end, 15):
                z = 0.01 + np.random.normal(0, 0.01)
                road_points.append([x, y, z])
                road_colors.append([0.95, 0.95, 0.95])  # 明るい白色
    
    # 停止線を追加
    stop_line_x = crosswalk_x - 5
    for y in np.linspace(6, 24, 100):
        for x in np.linspace(stop_line_x - 0.3, stop_line_x + 0.3, 5):
            z = 0.01 + np.random.normal(0, 0.01)
            road_points.append([x, y, z])
            road_colors.append([0.9, 0.9, 0.9])  # 白色
    
    # 縁石を追加
    for x in np.linspace(0, 50, 250):
        # 左側縁石
        y = 2.0
        z = 0.1 + np.random.normal(0, 0.02)
        road_points.append([x, y, z])
        road_colors.append([0.7, 0.7, 0.7])  # 薄い灰色
        
        # 右側縁石
        y = 28.0
        z = 0.1 + np.random.normal(0, 0.02)
        road_points.append([x, y, z])
        road_colors.append([0.7, 0.7, 0.7])  # 薄い灰色
    
    # Open3D点群オブジェクトを作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(road_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(road_colors))
    
    print(f"テスト点群生成完了: {len(road_points)} 点")
    return pcd


def run_test():
    """
    テスト実行
    """
    print("="*60)
    print("Road Marking Classifier - テスト実行")
    print("="*60)
    
    # テスト用点群を生成
    test_pcd = create_test_point_cloud()
    
    # テストファイルパス
    test_input = "test_road.pcd"
    test_output = "test_result.dxf"
    test_config = str(CONFIG_PATH)
    
    try:
        # テスト点群を保存
        print(f"\nテスト点群を保存: {test_input}")
        o3d.io.write_point_cloud(test_input, test_pcd)
        
        # 分類システムを初期化
        extractor = EnhancedWhiteLineExtractor(config_path=test_config)
        
        # 処理実行
        result = extractor.process_pcd_file(test_input, test_output)
        
        if result:
            print(f"\n✅ テスト成功！")
            print(f"結果ファイル: {test_output}")
            print(f"検出結果:")
            print(f"  - 横断歩道: {len(result['crosswalks'])}個")
            print(f"  - 停止線: {len(result['stop_lines'])}本")
            print(f"  - 車線: {len(result['lanes'])}本")
            print(f"  - 縁石: {len(result['curb_lines'])}本")
            
            # ファイルサイズ確認
            if os.path.exists(test_output):
                file_size = os.path.getsize(test_output)
                print(f"  - 出力ファイルサイズ: {file_size} bytes")
            
            return True
        else:
            print(f"\n❌ テスト失敗")
            return False
            
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生: {e}")
        return False
    
    finally:
        # テンポラリファイルを削除
        for temp_file in [test_input]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"テンポラリファイルを削除: {temp_file}")
                except:
                    pass


def main():
    """
    メイン関数
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == "--generate-sample":
            # サンプル点群のみ生成
            test_pcd = create_test_point_cloud()
            output_file = "sample_road.pcd"
            o3d.io.write_point_cloud(output_file, test_pcd)
            print(f"サンプル点群ファイルを生成: {output_file}")
            
            # 可視化
            print("点群を可視化します...")
            o3d.visualization.draw_geometries([test_pcd], window_name="Sample Road Point Cloud")
            return
        
        elif sys.argv[1] == "--real-data":
            # 実データテストのみ実行
            success = test_with_real_data()
            if success:
                print("\n🎉 実データテストが成功しました！")
            else:
                print("\n❌ 実データテストが失敗しました。")
            return
    
    # 両方のテストを実行
    print("🚗 Road Marking Classifier - 総合テスト")
    print("="*60)
    
    # 1. 生成データテスト
    print("\n【テスト1: 生成データテスト】")
    synthetic_success = run_test()
    
    # 2. 実データテスト
    print("\n【テスト2: 実データテスト】")
    real_data_success = test_with_real_data()
    
    # 結果まとめ
    print("\n" + "="*60)
    print("テスト結果まとめ")
    print("="*60)
    print(f"生成データテスト: {'✅ 成功' if synthetic_success else '❌ 失敗'}")
    print(f"実データテスト: {'✅ 成功' if real_data_success else '❌ 失敗'}")
    
    if synthetic_success and real_data_success:
        print("\n🎉 全テストが正常に完了しました！")
        print("\n次のステップ:")
        print("1. 生成されたDXFファイルをCADソフトで開いて結果を確認:")
        print("   - test_result.dxf (生成データ)")
        print("   - real_data_test_result.dxf (実データ)")
        print("2. 300GBの大容量データでバッチ処理を試す:")
        print("   python main.py --batch /path/to/large/dataset /path/to/output")
        print("3. 設定を調整してパフォーマンスを最適化")
    else:
        print("\n❌ 一部のテストが失敗しました。")
        print("ログを確認して問題を解決してください。")
        sys.exit(1)


if __name__ == "__main__":
    main()
