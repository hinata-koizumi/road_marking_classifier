from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import ezdxf
import pandas as pd
import streamlit as st


def _load_dxf(path: Path) -> pd.DataFrame:
    doc = ezdxf.readfile(path)
    records: List[Dict] = []
    for entity in doc.modelspace():
        layer = entity.dxf.layer
        if entity.dxftype() not in {"LINE", "ARC", "LWPOLYLINE"}:
            continue
        meta = entity.get_xdata("RMC_METADATA")
        metadata = {}
        if meta:
            for i in range(0, len(meta), 2):
                key = meta[i][1]
                value = meta[i + 1][1]
                metadata[key] = value
        records.append(
            {
                "layer": layer,
                "type": entity.dxftype(),
                "geometry": entity,
                "metadata": metadata,
                "score": float(metadata.get("score", 0.0)),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", type=str, required=False, help="DXF file path")
    parser.add_argument("--config", type=str, required=False, help="UI config (unused placeholder)")
    args, _ = parser.parse_known_args()
    dxf_path = Path(args.dxf) if args.dxf else None

    st.set_page_config(layout="wide")
    st.title("Road Marking Review")
    if dxf_path and dxf_path.exists():
        df = _load_dxf(dxf_path)
        st.sidebar.header("Filters")
        unique_layers = sorted(df.layer.unique())
        score_threshold = st.sidebar.slider("Score threshold", 0.0, 1.0, 0.55, 0.01)
        layer_visibility: Dict[str, bool] = {}
        for layer in unique_layers:
            layer_visibility[layer] = st.sidebar.checkbox(layer, True)
        st.sidebar.write("Editing tools")
        edit_action = st.sidebar.selectbox(
            "Action", ["None", "Split Line", "Merge Selection", "Move Vertex"]
        )
        st.sidebar.button("Undo")
        st.sidebar.button("Redo")

        filtered = df[(df.layer.map(layer_visibility)) & (df.score >= score_threshold)]
        st.write(f"{len(filtered)} entities above threshold")
        st.dataframe(filtered[["layer", "type", "metadata"]], use_container_width=True)

        if edit_action != "None":
            st.info(f"{edit_action} executed (simulation placeholder).")
    else:
        st.info("DXF not loaded. Use --dxf option.")

    st.markdown("### Notes")
    st.text_area("Review notes", "")
    st.button("Export edits (stub)")


if __name__ == "__main__":
    main()
