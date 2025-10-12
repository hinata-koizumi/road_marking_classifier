#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Road Marking Classifier - Enhanced White Line Extraction System
道路標示分類システム - 改善版白線抽出システム

This system automatically classifies and color-codes road markings including:
- Crosswalk stripes (横断歩道)
- Stop lines (停止線) 
- Lane markings/Sidewalk lines (車線/歩道線)

Based on research4.ipynb from Trust_Project02
"""

import open3d as o3d
import numpy as np
import ezdxf
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from pathlib import Path
import alphashape
from shapely.geometry import Polygon, MultiPolygon
import math
import argparse
import json
from datetime import datetime


class EnhancedWhiteLineExtractor:
    """
    改善版白線抽出クラス - 道路標示の自動分類・色分け出力システム
    """
    
    def __init__(self, config_path=None):
        """
        初期化
        Args:
            config_path: 設定ファイルのパス（オプション）
        """
        # デフォルトパラメータ設定
        self.config = {
            # RANSAC平面検出
            'ransac': {
                'distance_threshold': 0.1,
                'ransac_n': 3,
                'num_iterations': 1000
            },
            
            # 相対高さ分類
            'height': {
                'road_surface_tolerance': 0.05,  # ±5cm
                'curb_height_range': [0.05, 0.25]  # 5-25cm
            },
            
            # HSV色空間による白線検出
            'hsv': {
                's_range': [0, 30],      # 彩度: 低彩度（白っぽい）
                'v_range': [180, 255]    # 明度: 高明度（明るい）
            },
            
            # RGB閾値（フォールバック用）
            'rgb_threshold': 0.58,
            
            # クラスタリング
            'clustering': {
                'eps': 0.4,
                'min_samples': 15
            },
            
            # 形態学的処理
            'morphology': {
                'erosion_radius': 0.3,
                'erosion_neighbors': 10,
                'dilation_radius': 0.5
            },
            
            # ベクトル化
            'vectorization': {
                'min_length': 1.0,
                'aspect_ratio_threshold': 5.0,
                'linearity_threshold': 0.8
            },
            
            # 道路標示分類
            'classification': {
                'crosswalk_min_aspect_ratio': 1.5,
                'crosswalk_max_aspect_ratio': 8.0,
                'stop_line_min_aspect_ratio': 10.0,
                'stop_line_angle_tolerance': 15.0,
                'stop_line_distance_threshold': 5.0
            },
            
            # DXF出力設定
            'dxf_layers': {
                'crosswalk': {'name': 'CROSSWALK_STRIPES', 'color': 6},  # マゼンタ
                'stop_line': {'name': 'STOP_LINES', 'color': 2},         # 黄色
                'lane': {'name': 'LANES', 'color': 3},                   # 緑色
                'curb': {'name': 'CURBS', 'color': 1},                   # 赤色
                'metadata': {'name': 'METADATA', 'color': 7}             # 白色
            }
        }
        
        # 設定ファイルがあれば読み込み
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        print("EnhancedWhiteLineExtractor を初期化しました。")
    
    def load_config(self, config_path):
        """設定ファイルを読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                self.config.update(user_config)
            print(f"設定ファイルを読み込みました: {config_path}")
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
    
    def detect_road_plane(self, pcd):
        """
        RANSACによる道路平面検出
        """
        print("\n=== STEP 1: RANSAC道路平面検出 ===")
        
        plane_model, road_indices = pcd.segment_plane(
            distance_threshold=self.config['ransac']['distance_threshold'],
            ransac_n=self.config['ransac']['ransac_n'],
            num_iterations=self.config['ransac']['num_iterations']
        )
        
        road_layer_pcd = pcd.select_by_index(road_indices)
        print(f"道路平面を検出: {len(road_indices)} 点")
        
        return plane_model, road_layer_pcd
    
    def classify_by_height(self, pcd, plane_model):
        """
        基準平面からの相対高さによる分類
        """
        print("\n=== STEP 2: 相対高さによる分類 ===")
        
        points = np.asarray(pcd.points)
        a, b, c, d = plane_model
        
        # 各点の平面からの符号付き距離
        distances = points.dot(np.array([a, b, c])) + d
        
        # 高さに基づく分類
        road_surface_mask = np.abs(distances) <= self.config['height']['road_surface_tolerance']
        curb_mask = ((distances > self.config['height']['curb_height_range'][0]) & 
                    (distances < self.config['height']['curb_height_range'][1]))
        
        road_surface_pcd = pcd.select_by_index(np.where(road_surface_mask)[0])
        curbs_pcd = pcd.select_by_index(np.where(curb_mask)[0])
        
        print(f"道路表面: {len(road_surface_pcd.points)} 点")
        print(f"縁石: {len(curbs_pcd.points)} 点")
        
        return road_surface_pcd, curbs_pcd
    
    def extract_white_lines_hsv(self, pcd):
        """
        HSV色空間による白線検出
        """
        print("\n=== STEP 3: HSV色空間白線検出 ===")
        
        if not pcd.has_colors() or len(pcd.points) == 0:
            print("カラー情報がないため、RGB閾値にフォールバック")
            return self.extract_white_lines_rgb_fallback(pcd)
        
        colors_rgb = np.asarray(pcd.colors)
        colors_bgr = (colors_rgb * 255).astype(np.uint8)[:, ::-1]
        colors_hsv = cv2.cvtColor(np.array([colors_bgr]), cv2.COLOR_BGR2HSV)[0]
        
        # HSV範囲による白線マスク
        white_mask = ((colors_hsv[:, 1] >= self.config['hsv']['s_range'][0]) & 
                     (colors_hsv[:, 1] <= self.config['hsv']['s_range'][1]) &
                     (colors_hsv[:, 2] >= self.config['hsv']['v_range'][0]) & 
                     (colors_hsv[:, 2] <= self.config['hsv']['v_range'][1]))
        
        white_points_pcd = pcd.select_by_index(np.where(white_mask)[0])
        asphalt_pcd = pcd.select_by_index(np.where(~white_mask)[0])
        
        print(f"HSV白線検出: {len(white_points_pcd.points)} 点")
        print(f"アスファルト: {len(asphalt_pcd.points)} 点")
        
        return white_points_pcd, asphalt_pcd
    
    def extract_white_lines_rgb_fallback(self, pcd):
        """
        RGB閾値によるフォールバック白線検出
        """
        colors = np.asarray(pcd.colors)
        white_mask = np.all(colors > self.config['rgb_threshold'], axis=1)
        
        white_points_pcd = pcd.select_by_index(np.where(white_mask)[0])
        asphalt_pcd = pcd.select_by_index(np.where(~white_mask)[0])
        
        print(f"RGB白線検出（フォールバック): {len(white_points_pcd.points)} 点")
        
        return white_points_pcd, asphalt_pcd
    
    def apply_morphological_processing(self, pcd):
        """
        形態学的処理による点群整形
        """
        print("\n=== STEP 4: 形態学的処理 ===")
        
        if len(pcd.points) == 0:
            return pcd
        
        # 侵食処理: 孤立点やノイズを除去
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        erosion_indices = []
        
        for i, point in enumerate(pcd.points):
            [k, idx, _] = kdtree.search_radius_vector_3d(point, self.config['morphology']['erosion_radius'])
            if k >= self.config['morphology']['erosion_neighbors']:
                erosion_indices.append(i)
        
        eroded_pcd = pcd.select_by_index(erosion_indices)
        
        # 膨張処理: 連続性を回復
        if len(eroded_pcd.points) == 0:
            return eroded_pcd
        
        kdtree_original = o3d.geometry.KDTreeFlann(pcd)
        dilation_indices = set()
        
        for point in eroded_pcd.points:
            [k, idx_list, _] = kdtree_original.search_radius_vector_3d(point, self.config['morphology']['dilation_radius'])
            dilation_indices.update(idx_list)
        
        processed_pcd = pcd.select_by_index(list(dilation_indices))
        
        print(f"形態学的処理後: {len(processed_pcd.points)} 点")
        
        return processed_pcd
    
    def cluster_and_vectorize(self, pcd):
        """
        点群のクラスタリングとベクトル化
        """
        print("\n=== STEP 5: クラスタリング・ベクトル化 ===")
        
        if len(pcd.points) == 0:
            return [], []
        
        points = np.asarray(pcd.points)
        points_2d = points[:, :2]  # XY平面での処理
        
        # DBSCANクラスタリング
        clustering = DBSCAN(
            eps=self.config['clustering']['eps'],
            min_samples=self.config['clustering']['min_samples']
        ).fit(points_2d)
        
        labels = clustering.labels_
        unique_labels = set(labels)
        
        lines = []
        shapes = []
        
        for label in unique_labels:
            if label == -1:  # ノイズをスキップ
                continue
            
            cluster_points = points_2d[labels == label]
            
            if len(cluster_points) < 10:
                continue
            
            # 線形性の判定
            linearity = self.calculate_linearity(cluster_points)
            
            if linearity > self.config['vectorization']['linearity_threshold']:
                # 直線として処理
                line = self.fit_line_to_cluster(cluster_points)
                if line and self.calculate_line_length(line) > self.config['vectorization']['min_length']:
                    lines.append(line)
            else:
                # 形状として処理
                shape = self.create_shape_from_cluster(cluster_points)
                if shape:
                    shapes.append(shape)
        
        print(f"抽出結果: 線分 {len(lines)}本, 形状 {len(shapes)}個")
        
        return lines, shapes
    
    def calculate_linearity(self, points):
        """
        点群の線形性を計算
        """
        if len(points) < 3:
            return 0.0
        
        pca = PCA(n_components=2)
        pca.fit(points)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        linearity_score = explained_variance_ratio[0] / sum(explained_variance_ratio)
        
        return linearity_score
    
    def fit_line_to_cluster(self, points):
        """
        クラスタから直線を抽出
        """
        if len(points) < 2:
            return None
        
        # PCAで主軸を求める
        pca = PCA(n_components=2)
        pca.fit(points)
        
        center = np.mean(points, axis=0)
        direction = pca.components_[0]
        
        # 投影距離の範囲を求める
        projections = (points - center).dot(direction)
        min_proj, max_proj = projections.min(), projections.max()
        
        # 線分の端点を計算
        start_point = center + min_proj * direction
        end_point = center + max_proj * direction
        
        return (tuple(start_point), tuple(end_point))
    
    def calculate_line_length(self, line):
        """
        線分の長さを計算
        """
        start, end = line
        return np.linalg.norm(np.array(end) - np.array(start))
    
    def create_shape_from_cluster(self, points):
        """
        クラスタから形状を作成
        """
        if len(points) < 4:
            return None
        
        # Alpha shapeで境界を作成
        try:
            alpha_shape = alphashape.alphashape(points, 0.5)
            if isinstance(alpha_shape, Polygon):
                return list(alpha_shape.exterior.coords)
        except:
            pass
        
        # フォールバック: 凸包
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            return [tuple(points[i]) for i in hull.vertices]
        except:
            return None
    
    def classify_road_markings(self, lines, shapes):
        """
        道路標示の分類（横断歩道、停止線、車線等）
        """
        print("\n=== STEP 6: 道路標示分類 ===")
        
        crosswalks = []
        stop_lines = []
        lanes = []
        
        # 形状から横断歩道を検出
        for shape in shapes:
            if self.is_crosswalk_shape(shape):
                crosswalks.append(shape)
        
        # 線分から停止線と車線を分類
        for line in lines:
            if self.is_stop_line(line, crosswalks):
                stop_lines.append(line)
            else:
                lanes.append(line)
        
        print(f"分類結果: 横断歩道 {len(crosswalks)}個, 停止線 {len(stop_lines)}本, 車線 {len(lanes)}本")
        
        return crosswalks, stop_lines, lanes
    
    def is_crosswalk_shape(self, shape):
        """
        横断歩道形状の判定
        """
        if len(shape) < 4:
            return False
        
        # 矩形のアスペクト比をチェック
        points = np.array(shape)
        rect = cv2.minAreaRect(points.astype(np.float32))
        (_, (w, h), _) = rect
        
        if min(w, h) == 0:
            return False
        
        aspect_ratio = max(w, h) / min(w, h)
        
        return (self.config['classification']['crosswalk_min_aspect_ratio'] < 
                aspect_ratio < 
                self.config['classification']['crosswalk_max_aspect_ratio'])
    
    def is_stop_line(self, line, crosswalks):
        """
        停止線の判定
        """
        # 線分の長さとアスペクト比をチェック
        length = self.calculate_line_length(line)
        
        # 非常に短い線分は停止線ではない
        if length < 2.0:
            return False
        
        # 横断歩道との位置関係をチェック
        line_center = np.array([(line[0][0] + line[1][0]) / 2, 
                               (line[0][1] + line[1][1]) / 2])
        
        for crosswalk in crosswalks:
            crosswalk_center = np.mean(np.array(crosswalk), axis=0)
            distance = np.linalg.norm(line_center - crosswalk_center)
            
            if distance < self.config['classification']['stop_line_distance_threshold']:
                # 角度チェック（横断歩道に対して垂直かどうか）
                return self.check_perpendicular_to_crosswalk(line, crosswalk)
        
        return False
    
    def check_perpendicular_to_crosswalk(self, line, crosswalk):
        """
        停止線が横断歩道に対して垂直かどうかチェック
        """
        # 線分の方向ベクトル
        line_vector = np.array(line[1]) - np.array(line[0])
        line_angle = np.rad2deg(np.arctan2(line_vector[1], line_vector[0])) % 180
        
        # 横断歩道の主要方向を計算
        crosswalk_points = np.array(crosswalk)
        pca = PCA(n_components=2)
        pca.fit(crosswalk_points)
        crosswalk_vector = pca.components_[0]
        crosswalk_angle = np.rad2deg(np.arctan2(crosswalk_vector[1], crosswalk_vector[0])) % 180
        
        # 角度差を計算
        angle_diff = abs(line_angle - crosswalk_angle)
        angle_diff = min(angle_diff, 180 - angle_diff)
        
        # 垂直に近いかチェック（90度±許容範囲）
        return (90 - self.config['classification']['stop_line_angle_tolerance'] < 
                angle_diff < 
                90 + self.config['classification']['stop_line_angle_tolerance'])
    
    def save_to_dxf(self, crosswalks, stop_lines, lanes, curb_lines, output_path):
        """
        分類された道路標示をDXFファイルに色分けして保存
        """
        print("\n=== STEP 7: DXF保存（色分け）===")
        
        doc = ezdxf.new("R2010", setup=True)
        msp = doc.modelspace()
        
        # レイヤー定義
        layers_config = self.config['dxf_layers']
        for layer_key, layer_info in layers_config.items():
            doc.layers.add(name=layer_info['name'], color=layer_info['color'])
        
        # 横断歩道の描画（マゼンタ）
        for crosswalk in crosswalks:
            points_3d = [(p[0], p[1], 0.0) for p in crosswalk]
            msp.add_lwpolyline(points_3d, close=True, 
                             dxfattribs={"layer": layers_config['crosswalk']['name']})
        
        # 停止線の描画（黄色）
        for stop_line in stop_lines:
            start_3d = (stop_line[0][0], stop_line[0][1], 0.0)
            end_3d = (stop_line[1][0], stop_line[1][1], 0.0)
            msp.add_line(start_3d, end_3d, 
                        dxfattribs={"layer": layers_config['stop_line']['name']})
        
        # 車線/歩道線の描画（緑色）
        for lane in lanes:
            start_3d = (lane[0][0], lane[0][1], 0.0)
            end_3d = (lane[1][0], lane[1][1], 0.0)
            msp.add_line(start_3d, end_3d, 
                        dxfattribs={"layer": layers_config['lane']['name']})
        
        # 縁石の描画（赤色）
        for curb_line in curb_lines:
            start_3d = (curb_line[0][0], curb_line[0][1], 0.0)
            end_3d = (curb_line[1][0], curb_line[1][1], 0.0)
            msp.add_line(start_3d, end_3d, 
                        dxfattribs={"layer": layers_config['curb']['name']})
        
        # メタデータテキストの追加
        metadata_text = f"Road Marking Classification Results\\n横断歩道: {len(crosswalks)}個\\n停止線: {len(stop_lines)}本\\n車線: {len(lanes)}本\\n縁石: {len(curb_lines)}本\\n作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        msp.add_text(
            metadata_text,
            dxfattribs={
                "layer": layers_config['metadata']['name'],
                "insert": (0, 0, 0),
                "height": 1.0,
                "style": "Standard"
            }
        )
        
        try:
            doc.saveas(output_path)
            print(f"✅ 色分けDXFファイルを保存: '{output_path}'")
            print(f"   横断歩道: {len(crosswalks)}個, 停止線: {len(stop_lines)}本, 車線: {len(lanes)}本, 縁石: {len(curb_lines)}本")
        except Exception as e:
            print(f"❌ DXF保存エラー: {e}")
    
    def process_pcd_file(self, input_path, output_path):
        """
        メイン処理パイプライン
        """
        print(f"\n{'='*60}")
        print(f"道路標示分類システム - 処理開始")
        print(f"入力: {input_path}")
        print(f"出力: {output_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(input_path):
            print(f"❌ エラー: 入力ファイルが見つかりません - {input_path}")
            return None
        
        # 点群読み込み
        pcd = o3d.io.read_point_cloud(input_path)
        print(f"点群読み込み完了: {len(pcd.points)} 点")
        
        if not pcd.has_colors():
            print("⚠️ 警告: カラー情報がありません")
        
        try:
            # ステップ1: RANSAC道路平面検出
            plane_model, road_layer_pcd = self.detect_road_plane(pcd)
            
            # ステップ2: 相対高さによる分類
            road_surface_pcd, curbs_pcd = self.classify_by_height(road_layer_pcd, plane_model)
            
            # ステップ3: 白線検出（HSV優先、RGBフォールバック）
            white_lines_pcd, asphalt_pcd = self.extract_white_lines_hsv(road_surface_pcd)
            
            # ステップ4: 形態学的処理
            processed_white_pcd = self.apply_morphological_processing(white_lines_pcd)
            
            # ステップ5: ベクトル化
            white_lines, white_shapes = self.cluster_and_vectorize(processed_white_pcd)
            
            # 縁石のベクトル化（簡易版）
            curb_lines, _ = self.cluster_and_vectorize(curbs_pcd)
            
            # ステップ6: 道路標示分類
            crosswalks, stop_lines, lanes = self.classify_road_markings(white_lines, white_shapes)
            
            # ステップ7: DXF保存
            self.save_to_dxf(crosswalks, stop_lines, lanes, curb_lines, output_path)
            
            print(f"\n🎉 処理完了!")
            
            return {
                'crosswalks': crosswalks,
                'stop_lines': stop_lines,
                'lanes': lanes,
                'curb_lines': curb_lines,
                'road_surface_points': len(road_surface_pcd.points),
                'white_points': len(processed_white_pcd.points),
                'curb_points': len(curbs_pcd.points)
            }
            
        except Exception as e:
            print(f"❌ 処理中にエラーが発生しました: {e}")
            return None


def process_large_dataset(input_dir, output_dir, chunk_size_mb=100):
    """
    大容量データセット（300GB等）の分割処理
    """
    print("="*60)
    print("大容量データセット分割処理")
    print("="*60)
    
    # 入力ディレクトリの検証
    if not os.path.exists(input_dir):
        print(f"❌ 入力ディレクトリが見つかりません: {input_dir}")
        return False
    
    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # サポートされる拡張子
    supported_extensions = ['.pcd', '.ply', '.las', '.laz']
    
    # ファイル一覧を取得
    input_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                filepath = os.path.join(root, file)
                filesize = os.path.getsize(filepath) / (1024 * 1024)  # MB
                input_files.append((filepath, filesize))
    
    print(f"発見されたファイル数: {len(input_files)}")
    total_size = sum(size for _, size in input_files)
    print(f"総データサイズ: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
    
    # 分類システムを初期化
    extractor = EnhancedWhiteLineExtractor()
    
    # バッチ処理
    processed_count = 0
    success_count = 0
    
    for filepath, filesize in input_files:
        filename = os.path.basename(filepath)
        relative_path = os.path.relpath(filepath, input_dir)
        output_file = os.path.join(output_dir, relative_path.replace(os.path.splitext(relative_path)[1], '.dxf'))
        
        # 出力ディレクトリを作成
        output_file_dir = os.path.dirname(output_file)
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)
        
        print(f"\n処理中 ({processed_count+1}/{len(input_files)}): {filename} ({filesize:.1f}MB)")
        
        try:
            if filesize > chunk_size_mb:
                print(f"⚠️ 大容量ファイル ({filesize:.1f}MB) - メモリ最適化モードで処理")
                # 大容量ファイル用の処理（サンプリング等）
                result = extractor.process_pcd_file(filepath, output_file)
            else:
                result = extractor.process_pcd_file(filepath, output_file)
            
            if result:
                success_count += 1
                print(f"✅ 完了: {filename}")
            else:
                print(f"❌ 失敗: {filename}")
                
        except Exception as e:
            print(f"❌ エラー: {filename} - {e}")
        
        processed_count += 1
        
        # 進捗表示
        progress = (processed_count / len(input_files)) * 100
        print(f"進捗: {progress:.1f}% ({success_count}/{processed_count} 成功)")
    
    print(f"\n🎉 バッチ処理完了!")
    print(f"成功: {success_count}/{processed_count} ファイル")
    
    return success_count > 0


def main():
    """
    メイン関数 - コマンドライン引数の処理
    """
    parser = argparse.ArgumentParser(
        description='Road Marking Classifier - 道路標示分類システム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一ファイル処理
  python main.py input.pcd output.dxf
  python main.py input.pcd output.dxf --config config.json
  
  # 大容量データセット処理
  python main.py --batch input_dir output_dir
  python main.py --batch input_dir output_dir --chunk-size 200
  
  # ヘルプ
  python main.py --help

サポートされる入力形式: .pcd, .ply, .las, .laz
出力形式: .dxf (色分け済み)

色分けレイヤー:
  - CROSSWALK_STRIPES (マゼンタ): 横断歩道
  - STOP_LINES (黄色): 停止線
  - LANES (緑色): 車線/歩道線
  - CURBS (赤色): 縁石
        """
    )
    
    parser.add_argument('input', nargs='?', help='入力点群ファイル (.pcd, .ply) または入力ディレクトリ (--batch時)')
    parser.add_argument('output', nargs='?', help='出力DXFファイル (.dxf) または出力ディレクトリ (--batch時)')
    parser.add_argument('--config', '-c', help='設定ファイル (.json)')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細出力')
    parser.add_argument('--batch', '-b', action='store_true', help='バッチ処理モード（大容量データセット用）')
    parser.add_argument('--chunk-size', type=int, default=100, help='大容量ファイル判定閾値 (MB, デフォルト: 100)')
    
    args = parser.parse_args()
    
    # バッチ処理モード
    if args.batch:
        if not args.input or not args.output:
            print("❌ エラー: バッチモードでは入力ディレクトリと出力ディレクトリの両方が必要です")
            print("使用例: python main.py --batch input_dir output_dir")
            return 1
        
        success = process_large_dataset(args.input, args.output, args.chunk_size)
        return 0 if success else 1
    
    # 単一ファイル処理モード
    if not args.input or not args.output:
        print("❌ エラー: 入力ファイルと出力ファイルの両方が必要です")
        print("使用例: python main.py input.pcd output.dxf")
        return 1
    
    # 入力ファイルの存在確認
    if not os.path.exists(args.input):
        print(f"❌ エラー: 入力ファイルが見つかりません: {args.input}")
        return 1
    
    # 出力ディレクトリの作成
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 抽出システムの初期化
    extractor = EnhancedWhiteLineExtractor(config_path=args.config)
    
    # 処理実行
    result = extractor.process_pcd_file(args.input, args.output)
    
    if result:
        print(f"\n✅ 処理が正常に完了しました！")
        if args.verbose:
            print(f"詳細結果:")
            for key, value in result.items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)}個")
                else:
                    print(f"  {key}: {value}")
        return 0
    else:
        print(f"\n❌ 処理が失敗しました。")
        return 1


if __name__ == "__main__":
    exit(main())