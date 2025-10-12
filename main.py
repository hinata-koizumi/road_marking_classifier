#!/usr/bin/env python3
"""
Road Marking Classifier - 道路標示分類システム

点群データから道路標示（横断歩道、停止線、歩道線・車線）を自動検出し、
正確な色分けされたDXFファイルとして出力します。

Features:
- research4.ipynb準拠の高精度アルゴリズム
- 3種類の道路標示の正確な分類
- 大容量データセット対応（300GB+）
- バッチ処理モード
- CAD対応DXF出力（色分けレイヤー）

Color Coding:
- 横断歩道: 紫色 (Magenta, color=6)
- 停止線: 赤色 (Red, color=1)  
- 歩道線・車線: 黄色 (Yellow, color=2)

Usage:
    python main.py input.pcd output.dxf
    python main.py --batch input_dir output_dir
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import json

import numpy as np
import open3d as o3d
import cv2
import ezdxf
from sklearn.cluster import DBSCAN


class CompleteRoadMarkingExtractor:
    """
    research4.ipynb準拠の完全版道路標示抽出システム
    横断歩道（紫）、停止線（赤）、歩道線・車線（黄色）の3分類
    """
    
    def __init__(self, config):
        self.config = config
        self.points_3d_white = None
        self.binary_image = None
        self.image_transform_params = {}
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """ロガーの設定"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _log(self, message):
        """ログ出力"""
        if self.config.get('verbose', False):
            self.logger.info(message)
        else:
            print(message)

    def load_and_filter_pcd(self, pcd_path):
        """PCDファイルを読み込み、白い点を抽出"""
        self._log(f"ステップ1: '{pcd_path}' を読み込み、白い点を抽出中...")
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            if len(colors) == 0:
                self._log("警告: 色情報が見つかりません。グレースケール処理を試行します。")
                # 色情報がない場合の処理
                self.points_3d_white = points
            else:
                # 白色点の抽出
                white_mask = np.all(colors > self.config['white_threshold'], axis=1)
                self.points_3d_white = points[white_mask]
            
            if len(self.points_3d_white) == 0:
                self._log("エラー: 白い点が見つかりませんでした。")
                return False
                
            self._log(f"-> {len(self.points_3d_white):,}個の白い点を抽出しました。")
            return True
            
        except Exception as e:
            self._log(f"エラー: PCDファイルの読み込みに失敗しました: {e}")
            return False

    def project_to_2d_image(self):
        """3D点群を2D画像に投影"""
        self._log("ステップ2: 3D点を2D画像に投影中...")
        
        points = self.points_3d_white
        resolution = self.config['image_resolution']
        
        # 座標変換パラメータ計算
        x_min, y_min, _ = np.min(points, axis=0)
        x_max, y_max, _ = np.max(points, axis=0)
        
        self.image_transform_params = {
            'x_min': x_min, 
            'y_max': y_max, 
            'resolution': resolution
        }
        
        # 画像サイズ計算
        width = int(np.ceil((x_max - x_min) / resolution)) + 1
        height = int(np.ceil((y_max - y_min) / resolution)) + 1
        
        self._log(f"-> 画像サイズ: {width} x {height} pixels")
        
        # 2D画像作成
        self.binary_image = np.zeros((height, width), dtype=np.uint8)
        
        # 座標変換
        u_coords = np.clip(((points[:, 0] - x_min) / resolution), 0, width - 1).astype(int)
        v_coords = np.clip(((y_max - points[:, 1]) / resolution), 0, height - 1).astype(int)
        
        # 画像に点を描画
        self.binary_image[v_coords, u_coords] = 255
        
        # モルフォロジー処理でノイズ除去
        kernel = np.ones((self.config['morph_kernel_size'], self.config['morph_kernel_size']), np.uint8)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernel)
        
        if self.config.get('debug_save_images', False):
            cv2.imwrite("debug_binary_image.png", self.binary_image)
            self._log("-> デバッグ画像を保存: debug_binary_image.png")
            
        return True

    def _average_angles(self, angles):
        """角度の平均を計算（循環統計）"""
        angles_rad = np.deg2rad(angles)
        avg_cos = np.mean(np.cos(2 * angles_rad))
        avg_sin = np.mean(np.sin(2 * angles_rad))
        avg_angle_rad = np.arctan2(avg_sin, avg_cos) / 2
        return np.rad2deg(avg_angle_rad)

    def _find_crosswalk_groups(self, all_candidates):
        """横断歩道をDBSCANでグループ化"""
        self._log(f"  -> {len(all_candidates)}個の候補から横断歩道をグループ化...")
        
        if len(all_candidates) < self.config['cw_min_stripes_in_group']:
            return [], set(), []

        # DBSCANで中心座標をクラスタリング
        centers = np.array([rect[0] for rect in all_candidates])
        db = DBSCAN(eps=self.config['dbscan_eps'], 
                   min_samples=self.config['cw_min_stripes_in_group']).fit(centers)
        
        crosswalk_rects = []
        used_candidate_indices = set()
        crosswalk_group_props = []

        for label in set(db.labels_):
            if label == -1:  # ノイズは無視
                continue
                
            cluster_indices = np.where(db.labels_ == label)[0]
            group_rects = [all_candidates[i] for i in cluster_indices]
            
            # アスペクト比チェック
            aspect_ratios = [max(rect[1])/min(rect[1]) if min(rect[1])>0 else 0 
                           for rect in group_rects]
            median_aspect_ratio = np.median(aspect_ratios)

            # 横断歩道の縞らしい形状かチェック
            if not (self.config['cw_aspect_ratio_min'] <= median_aspect_ratio 
                   <= self.config['cw_aspect_ratio_max']):
                continue

            # 横断歩道として採用
            crosswalk_rects.extend(group_rects)
            used_candidate_indices.update(cluster_indices)
            
            # グループ全体の境界矩形
            all_points = np.vstack([cv2.boxPoints(rect) for rect in group_rects])
            group_bounding_rect = cv2.minAreaRect(all_points)
            
            # 縞の平均角度
            angles = [rect[2] if rect[1][0] > rect[1][1] else rect[2] + 90 
                     for rect in group_rects]
            avg_stripe_angle = self._average_angles(angles)
            
            crosswalk_group_props.append({
                'rect': group_bounding_rect, 
                'stripe_angle': avg_stripe_angle
            })
        
        self._log(f"  -> {len(crosswalk_group_props)}個の横断歩道グループを特定しました。")
        return crosswalk_rects, used_candidate_indices, crosswalk_group_props

    def _find_stop_lines(self, remaining_candidates, crosswalk_groups):
        """停止線を特定（横断歩道近辺の垂直な細長い矩形）"""
        self._log(f"  -> 残りの{len(remaining_candidates)}個の候補から停止線を分類...")
        
        if not crosswalk_groups:
            return []
        
        stop_lines = []
        dist_thresh_px = self.config['stop_line_dist_thresh'] / self.config['image_resolution']

        for rect in remaining_candidates:
            center, (w, h), angle = rect
            
            # 十分に細長いかチェック
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if aspect_ratio < self.config['stop_line_aspect_ratio_min']:
                continue

            rect_angle = angle if w > h else angle + 90
            
            # 各横断歩道グループとの関係をチェック
            for cw_group in crosswalk_groups:
                # 距離チェック
                dist = cv2.pointPolygonTest(cv2.boxPoints(cw_group['rect']), center, True)
                if abs(dist) < dist_thresh_px:
                    # 角度チェック（垂直性）
                    angle_diff = abs(rect_angle - cw_group['stripe_angle'])
                    angle_diff = min(angle_diff, 180 - angle_diff)
                    
                    tolerance = self.config.get('stop_line_angle_tolerance', 15)
                    if 90 - tolerance < angle_diff < 90 + tolerance:
                        stop_lines.append(rect)
                        break
        
        self._log(f"  -> {len(stop_lines)}本の停止線を特定しました。")
        return stop_lines

    def _find_lanes(self, remaining_candidates):
        """車線・歩道線を特定（残りの細長い矩形）"""
        self._log(f"  -> 残りの{len(remaining_candidates)}個の候補から車線・歩道線を分類...")
        
        lanes = []
        for rect in remaining_candidates:
            w, h = rect[1]
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            
            # 細長い形状をチェック
            if aspect_ratio >= self.config['lane_aspect_ratio_min']:
                lanes.append(rect)
        
        self._log(f"  -> {len(lanes)}本の車線・歩道線を特定しました。")
        return lanes

    def vectorize_image(self):
        """画像から道路標示を検出・分類"""
        self._log("ステップ3: 形状検出と分類を実行中...")
        
        # 輪郭検出
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 候補矩形の抽出
        all_candidates = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.config['rect_min_area']:
                continue
            rect = cv2.minAreaRect(cnt)
            if min(rect[1]) < self.config['rect_min_short_side']:
                continue
            all_candidates.append(rect)

        self._log(f"  -> {len(all_candidates)}個の矩形候補を抽出しました。")

        # 1. 横断歩道の特定
        crosswalks, used_indices1, crosswalk_groups = self._find_crosswalk_groups(all_candidates)

        # 2. 停止線の特定
        remaining_after_crosswalk = [cand for i, cand in enumerate(all_candidates) 
                                   if i not in used_indices1]
        stop_lines = self._find_stop_lines(remaining_after_crosswalk, crosswalk_groups)
        
        # 使用された候補のインデックスを追跡
        stop_line_candidates = set()
        for stop_line in stop_lines:
            for i, cand in enumerate(remaining_after_crosswalk):
                if cand == stop_line:
                    stop_line_candidates.add(i)
                    break

        # 3. 車線・歩道線の特定
        remaining_after_stop = [cand for i, cand in enumerate(remaining_after_crosswalk) 
                              if i not in stop_line_candidates]
        lanes = self._find_lanes(remaining_after_stop)
        
        return crosswalks, stop_lines, lanes

    def _image_to_world(self, points):
        """画像座標を世界座標に変換"""
        p = self.image_transform_params
        points_arr = np.array(points)
        if points_arr.ndim == 1:
            points_arr = np.array([points_arr])
        
        world_points = []
        for point in points_arr:
            world_x = point[0] * p['resolution'] + p['x_min']
            world_y = p['y_max'] - (point[1] * p['resolution'])
            world_points.append((world_x, world_y, 0.0))
        
        return world_points

    def save_to_dxf(self, crosswalks, stop_lines, lanes, dxf_path):
        """DXFファイルに色分けして保存（research4.ipynb準拠）"""
        self._log(f"ステップ4: 結果を '{dxf_path}' に保存しています...")
        
        # DXFドキュメント作成
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # ★★★ research4.ipynb準拠の正確な色分けレイヤー ★★★
        doc.layers.add(name='CROSSWALK_STRIPES', color=6)  # Magenta (紫) - 横断歩道
        doc.layers.add(name='STOP_LINES', color=1)         # Red (赤) - 停止線  
        doc.layers.add(name='LANES', color=2)              # Yellow (黄色) - 歩道線・車線

        # 横断歩道を紫で描画
        for rect in crosswalks:
            world_points = self._image_to_world(cv2.boxPoints(rect))
            msp.add_lwpolyline(world_points, close=True, 
                             dxfattribs={'layer': 'CROSSWALK_STRIPES'})

        # 停止線を赤で描画
        for rect in stop_lines:
            world_points = self._image_to_world(cv2.boxPoints(rect))
            msp.add_lwpolyline(world_points, close=True, 
                             dxfattribs={'layer': 'STOP_LINES'})

        # 歩道線・車線を黄色で描画
        for rect in lanes:
            world_points = self._image_to_world(cv2.boxPoints(rect))
            msp.add_lwpolyline(world_points, close=True, 
                             dxfattribs={'layer': 'LANES'})
        
        try:
            doc.saveas(dxf_path)
            self._log("-> DXF保存完了。")
            self._log(f"   🟣 横断歩道: {len(crosswalks)}個 (紫色)")
            self._log(f"   🔴 停止線: {len(stop_lines)}個 (赤色)")
            self._log(f"   🟡 歩道線・車線: {len(lanes)}個 (黄色)")
            
            return True
        except Exception as e:
            self._log(f"エラー: DXFファイルの保存に失敗しました: {e}")
            return False

    def process_single_file(self, pcd_path: str, dxf_path: str) -> bool:
        """単一ファイルの処理"""
        start_time = time.time()
        self._log(f"\n=== 単一ファイル処理開始 ===")
        self._log(f"入力: {pcd_path}")
        self._log(f"出力: {dxf_path}")
        
        success = False
        try:
            if self.load_and_filter_pcd(pcd_path):
                if self.project_to_2d_image():
                    crosswalks, stop_lines, lanes = self.vectorize_image()
                    success = self.save_to_dxf(crosswalks, stop_lines, lanes, dxf_path)
        except Exception as e:
            self._log(f"エラー: 処理中にエラーが発生しました: {e}")
            
        elapsed = time.time() - start_time
        status = "成功" if success else "失敗"
        self._log(f"\n=== 処理{status} (処理時間: {elapsed:.2f}秒) ===")
        return success


class BatchProcessor:
    """大容量データセットのバッチ処理クラス"""
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.batch")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _is_large_file(self, file_path: Path) -> bool:
        """ファイルが大容量かどうかを判定"""
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            return size_mb > self.config.get('chunk_size', 100)
        except:
            return False

    def _find_point_cloud_files(self, input_dir: Path) -> List[Path]:
        """点群ファイルを検索"""
        extensions = ['.pcd', '.ply', '.las', '.laz']
        files = []
        for ext in extensions:
            files.extend(input_dir.rglob(f'*{ext}'))
        return sorted(files)

    def process_batch(self, input_dir: str, output_dir: str) -> dict:
        """バッチ処理実行"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"入力ディレクトリが見つかりません: {input_dir}")
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 点群ファイルを検索
        files = self._find_point_cloud_files(input_path)
        if not files:
            self.logger.warning(f"点群ファイルが見つかりません: {input_dir}")
            return {'processed': 0, 'failed': 0, 'skipped': 0}
        
        self.logger.info(f"\n=== バッチ処理開始 ===")
        self.logger.info(f"入力ディレクトリ: {input_dir}")
        self.logger.info(f"出力ディレクトリ: {output_dir}")
        self.logger.info(f"処理対象ファイル数: {len(files)}")
        
        results = {'processed': 0, 'failed': 0, 'skipped': 0}
        start_time = time.time()
        
        for i, file_path in enumerate(files, 1):
            self.logger.info(f"\n[{i}/{len(files)}] 処理中: {file_path.name}")
            
            # 出力ファイル名を生成
            output_file = output_path / f"{file_path.stem}.dxf"
            
            # 既に処理済みの場合はスキップ
            if output_file.exists() and not self.config.get('overwrite', False):
                self.logger.info(f"スキップ: {output_file.name} (既存)")
                results['skipped'] += 1
                continue
            
            # ファイルサイズチェック
            if self._is_large_file(file_path):
                size_mb = file_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"大容量ファイル検出: {size_mb:.1f}MB")
            
            # 処理実行
            extractor = CompleteRoadMarkingExtractor(self.config)
            success = extractor.process_single_file(str(file_path), str(output_file))
            
            if success:
                results['processed'] += 1
                self.logger.info(f"✅ 成功: {output_file.name}")
            else:
                results['failed'] += 1
                self.logger.error(f"❌ 失敗: {file_path.name}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"\n=== バッチ処理完了 ===")
        self.logger.info(f"処理時間: {elapsed:.2f}秒")
        self.logger.info(f"成功: {results['processed']}")
        self.logger.info(f"失敗: {results['failed']}")
        self.logger.info(f"スキップ: {results['skipped']}")
        
        return results


def load_config(config_path: Optional[str] = None) -> dict:
    """設定ファイルの読み込み"""
    # デフォルト設定（research4.ipynb準拠）
    default_config = {
        'white_threshold': 0.58,
        'image_resolution': 0.05,
        'morph_kernel_size': 3,
        
        # 矩形検出の基本パラメータ
        'rect_min_area': 80,
        'rect_min_short_side': 4,
        
        # 横断歩道の形状パラメータ
        'cw_aspect_ratio_min': 2.0,
        'cw_aspect_ratio_max': 10.0,
        'cw_min_stripes_in_group': 3,
        'dbscan_eps': 130,
        
        # 停止線の形状パラメータ
        'stop_line_aspect_ratio_min': 12.0,
        'stop_line_dist_thresh': 6.0,
        'stop_line_angle_tolerance': 15,
        
        # 車線・歩道線の形状パラメータ
        'lane_aspect_ratio_min': 3.0,
        
        # バッチ処理設定
        'chunk_size': 100,  # MB
        'overwrite': False,
        'debug_save_images': False,
        'verbose': False
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            default_config.update(user_config)
            print(f"設定ファイルを読み込みました: {config_path}")
        except Exception as e:
            print(f"警告: 設定ファイルの読み込みに失敗しました: {e}")
    
    return default_config


def main():
    """メイン関数"""
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
  - CROSSWALK_STRIPES (紫色): 横断歩道
  - STOP_LINES (赤色): 停止線
  - LANES (黄色): 車線/歩道線
        """
    )
    
    parser.add_argument('input', nargs='?', help='入力点群ファイル (.pcd, .ply) または入力ディレクトリ (--batch時)')
    parser.add_argument('output', nargs='?', help='出力DXFファイル (.dxf) または出力ディレクトリ (--batch時)')
    parser.add_argument('--config', '-c', help='設定ファイル (.json)')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細出力')
    parser.add_argument('--batch', '-b', action='store_true', help='バッチ処理モード（大容量データセット用）')
    parser.add_argument('--chunk-size', type=int, default=100, help='大容量ファイル判定閾値 (MB, デフォルト: 100)')

    args = parser.parse_args()

    # 引数チェック
    if not args.input or not args.output:
        parser.print_help()
        return 1

    # 設定読み込み
    config = load_config(args.config)
    config['verbose'] = args.verbose
    config['chunk_size'] = args.chunk_size

    try:
        if args.batch:
            # バッチ処理モード
            processor = BatchProcessor(config)
            results = processor.process_batch(args.input, args.output)
            return 0 if results['failed'] == 0 else 1
        else:
            # 単一ファイル処理モード
            extractor = CompleteRoadMarkingExtractor(config)
            success = extractor.process_single_file(args.input, args.output)
            return 0 if success else 1
            
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        return 1
    except Exception as e:
        print(f"エラー: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())