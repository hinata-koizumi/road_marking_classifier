# Road Marking Classifier - 共有パッケージ

## 📦 プロジェクト概要

**research4.ipynb準拠の高精度道路標示分類システム**

### ✨ 主要機能
- 🟣 横断歩道検出（紫色）
- 🔴 停止線検出（赤色）  
- 🟡 歩道線・車線検出（黄色）
- 🚀 300GB大容量データ対応
- ⚡ 高速処理（22,645点を0.79秒）

## 🎯 実証済み性能

### テスト結果
```
入力データ: map5200_-125850_converted.pcd (22,645点)
処理時間: 0.79秒
出力結果:
- 横断歩道: 15個 (紫色)
- 停止線: 0個 (赤色)
- 歩道線・車線: 18個 (黄色)
- DXFファイル: 28,772バイト
```

## 🛠️ 使用方法

### 基本使用法
```bash
# 単一ファイル処理
python main.py input.pcd output.dxf

# 大容量データセット処理
python main.py --batch input_dir output_dir --chunk-size 100

# 詳細ログ付き
python main.py input.pcd output.dxf --verbose
```

### 設定ファイル（configs/default_config.json）
```json
{
    "white_threshold": 0.58,
    "image_resolution": 0.05,
    "cw_aspect_ratio_min": 2.0,
    "cw_aspect_ratio_max": 10.0,
    "dbscan_eps": 130,
    "stop_line_aspect_ratio_min": 12.0,
    "lane_aspect_ratio_min": 3.0
}
```

## 📋 必要な環境

### 依存関係
```txt
open3d>=0.18.0
opencv-python>=4.8.0
ezdxf>=1.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
matplotlib>=3.7.0
```

### Pythonバージョン
- Python 3.8 以上推奨

## 🎨 出力形式

### DXFレイヤー構成
- `CROSSWALK_STRIPES`: 横断歩道（color=6, 紫色）
- `STOP_LINES`: 停止線（color=1, 赤色）
- `LANES`: 歩道線・車線（color=2, 黄色）

### 対応形式
- **入力**: .pcd, .ply, .las, .laz
- **出力**: .dxf (AutoCAD互換)

## 🚀 特徴

### research4.ipynb準拠の高精度アルゴリズム
1. **白色点抽出**: HSV閾値フィルタリング
2. **DBSCANクラスタリング**: 横断歩道の縞模様グループ化
3. **幾何学的分析**: アスペクト比・角度・距離による分類
4. **世界座標変換**: 正確なCAD出力

### 大容量データ対応
- **バッチ処理**: 複数ファイル自動処理
- **メモリ効率**: 100MB単位でのチャンク処理
- **進捗表示**: リアルタイム処理状況
- **エラーハンドリング**: 堅牢な例外処理

## 📞 サポート

### 問題が発生した場合
1. Python環境の確認
2. 依存関係の再インストール
3. 入力データ形式の確認
4. 設定パラメータの調整

### パフォーマンス最適化
- `image_resolution`: 精度と速度のバランス調整
- `chunk_size`: メモリ使用量調整
- `dbscan_eps`: クラスタリング精度調整

---

## 📁 ファイル構成

```
road-marking-classifier/
├── main.py                         # CLIラッパー
├── setup.py                        # パッケージ設定
├── requirements.txt                # 依存関係一覧
├── configs/
│   └── default_config.json         # デフォルト設定
├── data/
│   ├── raw/                        # 生データの配置先
│   ├── interim/                    # 中間生成物
│   ├── processed/                  # 処理済みデータ
│   └── external/                   # 外部入手データ
├── src/
│   └── road_marking_classifier/
│       ├── __init__.py
│       └── cli.py                  # メイン実装
├── examples/
│   └── usage_examples.py           # 使用例
├── tests/
│   └── manual_test.py              # 手動テストスクリプト
├── legacy/
│   └── enhanced_extractor.py       # 旧実装
├── assets/
│   └── debug_binary_image_complete.png
└── docs/
    └── SHARING_GUIDE.md
```

**作成日**: 2025年10月12日  
**バージョン**: v1.0.0  
**準拠**: research4.ipynb アルゴリズム
