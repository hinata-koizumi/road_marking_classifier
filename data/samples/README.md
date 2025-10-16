## Dummy Data

`generate_dummy_las.py` creates synthetic 100m road corridors with white lines, stop lines, crosswalks, and curbs for functional testing.

### Generate

```bash
python data_samples/generate_dummy_las.py --out data_samples/site01.las --epsg 6677
```

### Real Data

Place real LAS/LAZ/PCD files under `data_samples/` and update `eval/bench_list.csv`.
