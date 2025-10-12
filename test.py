#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Road Marking Classifier - Test Script
道路標示分類システム テストスクリプト
"""

import sys
import os
import numpy as np
import open3d as o3d
from main import EnhancedWhiteLineExtractor


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
    test_config = "config.json"
    
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
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-sample":
        # サンプル点群のみ生成
        test_pcd = create_test_point_cloud()
        output_file = "sample_road.pcd"
        o3d.io.write_point_cloud(output_file, test_pcd)
        print(f"サンプル点群ファイルを生成: {output_file}")
        
        # 可視化
        print("点群を可視化します...")
        o3d.visualization.draw_geometries([test_pcd], window_name="Sample Road Point Cloud")
        return
    
    # 通常のテスト実行
    success = run_test()
    
    if success:
        print("\n🎉 全テストが正常に完了しました！")
        print("\n次のステップ:")
        print("1. test_result.dxf をCADソフトで開いて結果を確認")
        print("2. 実際の点群データで試してみる:")
        print("   python main.py your_data.pcd output.dxf")
        print("3. 設定を調整してパフォーマンスを最適化")
    else:
        print("\n❌ テストが失敗しました。")
        print("ログを確認して問題を解決してください。")
        sys.exit(1)


if __name__ == "__main__":
    main()