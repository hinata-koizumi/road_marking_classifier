"""
research4.ipynbの完全なDXF色分け出力コード
歩道線（黄色）、停止線（赤）、横断歩道（紫）の正確な実装
"""

import open3d as o3d
import numpy as np
import cv2
import ezdxf
import os
from sklearn.cluster import DBSCAN

class CompleteRoadMarkingExtractor:
    """
    research4.ipynbの完全版 - 3種類の道路標示を正確に色分け出力
    """
    def __init__(self, config):
        self.config = config
        self.points_3d_white = None
        self.binary_image = None
        self.image_transform_params = {}

    def _log(self, message):
        print(message)

    def load_and_filter_pcd(self, pcd_path):
        """PCDファイルを読み込み、白い点を抽出"""
        self._log(f"ステップ1: '{pcd_path}' を読み込み、白い点を抽出中...")
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # 白色点の抽出
            white_mask = np.all(colors > self.config['white_threshold'], axis=1)
            self.points_3d_white = points[white_mask]
            
            if len(self.points_3d_white) == 0:
                self._log("エラー: 白い点が見つかりませんでした。")
                return False
                
            self._log(f"-> {len(self.points_3d_white)}個の白い点を抽出しました。")
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
        
        if self.config['debug_save_images']:
            cv2.imwrite("debug_binary_image_complete.png", self.binary_image)
            
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
        """DXFファイルに色分けして保存"""
        self._log(f"ステップ4: 結果を '{dxf_path}' に保存しています...")
        
        # DXFドキュメント作成
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # ★★★ 正しい色分けレイヤー ★★★
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
            self._log(f"   横断歩道: {len(crosswalks)}個 (紫)")
            self._log(f"   停止線: {len(stop_lines)}個 (赤)")
            self._log(f"   歩道線・車線: {len(lanes)}個 (黄色)")
        except Exception as e:
            self._log(f"エラー: DXFファイルの保存に失敗しました: {e}")

    def run(self, pcd_path, dxf_path):
        """メイン処理実行"""
        if self.load_and_filter_pcd(pcd_path):
            if self.project_to_2d_image():
                crosswalks, stop_lines, lanes = self.vectorize_image()
                self.save_to_dxf(crosswalks, stop_lines, lanes, dxf_path)
        self._log("\n🎉 すべての処理が完了しました。")


# 設定パラメータ
CONFIG = {
    'debug_save_images': True,
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
}

if __name__ == '__main__':
    INPUT_PCD_FILE = "D:/MyWorkspace/PythonProjects/Trust_Project02/CursorResearch/original_data/map5200_-125850_converted.pcd"
    OUTPUT_DXF_FILE = "complete_road_markings.dxf"

    extractor = CompleteRoadMarkingExtractor(CONFIG)
    extractor.run(INPUT_PCD_FILE, OUTPUT_DXF_FILE)