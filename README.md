# Road Marking Classifier

最小限のコンポーネントで「点群 → DXF」変換を行うシンプルな道路標示分類システム。

## Quick Start

```bash
# 環境セットアップ
git clone <repo_url>
cd road-marking-classifier
python -m venv .venv && source .venv/bin/activate
pip install -e .

# サンプルデータ生成
python data/samples/generate_dummy.py --out data/samples/site01.las --epsg 6677

# パイプライン実行
python scripts/run_pipeline.py \
  --in data/samples/site01.las \
  --out data/output/site01.dxf \
  --epsg 6677
```

## 📁 プロジェクト構造

```
road-marking-classifier/
├── src/road_marking_classifier/    # メインソースコード
│   ├── io/                        # 入出力処理
│   │   ├── pointcloud.py         # 点群読み込み
│   │   └── dxf.py                # DXF出力
│   ├── processing/               # データ処理
│   │   ├── preprocess.py         # 前処理
│   │   ├── bev.py                # BEV生成
│   │   ├── detect.py             # 検出処理
│   │   └── classify.py           # 分類処理
│   ├── cli/                      # CLI関連
│   │   └── main.py               # CLIエントリーポイント
│   ├── config.py                 # 設定管理
│   ├── types.py                  # データ型定義
│   └── pipeline.py               # メインパイプライン
├── data/                         # データディレクトリ
│   ├── samples/                  # サンプルデータ
│   ├── input/                    # 入力データ
│   └── output/                   # 出力データ
├── scripts/                      # ユーティリティスクリプト
├── tests/                        # テストコード
├── docs/                         # ドキュメント
└── legacy/                       # 旧バージョン（アーカイブ）
```

## 🔄 パイプライン処理フロー

1. **点群読み込み** - LAS/LAZ/PCDファイルの読み込み
2. **前処理** - ボクセルダウンサンプリング、路面推定
3. **BEV生成** - 鳥瞰図ラスタライズ
4. **検出** - ライン検出（PPHT）、横断歩道検出
5. **分類** - ROAD_LINE/STOP_LINE/CROSSWALK分類
6. **DXF出力** - CAD互換形式での出力

## 設定パラメータ

```bash
python scripts/run_pipeline.py \
  --in input.las \
  --out output.dxf \
  --epsg 6677 \
  --roi 35.0 \
  --bev 0.05 \
  --stop-line 6.0 \
  --dp 0.02
```

- `--roi`: BEV生成時の半径（m）
- `--bev`: BEVピクセル解像度（m/pixel）
- `--stop-line`: 停止線判定長さ（m）
- `--dp`: Douglas-Peucker簡略化許容誤差（m）

## 出力形式

### DXFファイル構造
- **レイヤー**: `ROAD_LINE`, `STOP_LINE`, `CROSSWALK`, `QC_REVIEW`
- **エンティティ**: `LINE`, `LWPOLYLINE`
- **XDATA**: クラス名、信頼度、ソースタイル、タイムスタンプ

## テスト

```bash
# 単体テスト
python -m pytest tests/

# 統合テスト
python scripts/run_pipeline.py --in data/samples/site01.las --out test_output.dxf
```

## 依存関係

- `numpy>=1.23`
- `scipy>=1.9`
- `laspy>=2.4`
- `open3d>=0.17`
- `opencv-python>=4.7`
- `ezdxf>=1.1`
- `shapely>=2.0`

## 開発

```bash
# 開発モードでインストール
pip install -e .

# コードフォーマット
black src/
isort src/

# 型チェック
mypy src/
```

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照してください。