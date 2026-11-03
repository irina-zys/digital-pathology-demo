"""Run: uv run --locked streamlit run app.py --server.address 127.0.0.1."""

import base64
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path(os.environ.get("DIGITAL_PATHOLOGY_OUTPUTS", ROOT / "outputs")).expanduser().resolve()
PREVIEWS = {"Original": "thumbnail.png", "Retained mask": "clean_mask.png",
            "Filtered": "filtered.png", "Overlay": "overlay.png"}
OPTIONAL_PREVIEWS = {"Tissue + holes": "tissue_holes.png"}
EXPORT_VIEWS = {f"{size} output px {label}": (size, kind)
                for size in (256, 4096)
                for label, kind in (("coordinate coverage", "coverage"), ("sampled pixels", "pixels"))}


def artifact(run, relative):
    path = (run / relative).resolve()
    if not path.is_relative_to(run.resolve()):
        raise ValueError("Artifact path leaves the selected run")
    return path


def read_preview(path, *, pixels=True):
    if path.stat().st_size > 20_000_000:
        raise ValueError(f"Preview exceeds 20 MB: {path.name}")
    with Image.open(path) as image:
        if image.format not in {"PNG", "JPEG"} or max(image.size) > 4096 or image.width * image.height > 16_777_216:
            raise ValueError("Expected a bounded preprocessing PNG or JPEG, maximum 4096 pixels per edge")
        return image.copy() if pixels else None


def export_preview(run, slide, size, kind):
    layout = next((item for item in slide.get("exports", {}).get("layouts", [])
                   if item["patch_size"] == size), {})
    relative = layout.get("coverage_stitch" if kind == "coverage" else "sampled_pixel_stitch")
    # Earlier exporter versions record the common CLAM layout rather than the stitch path.
    legacy_pixels = kind == "pixels" and not layout.get("stitch_type")
    coverage = kind == "coverage" and layout.get("stitch_type") == "thumbnail_pixels_within_exported_coordinate_coverage"
    if not relative and (legacy_pixels or coverage) and layout.get("coords"):
        relative = str(Path(layout["folder"]) / "stitches" / (Path(layout["coords"]).stem + ".jpg"))
    path = artifact(run, relative) if relative else None
    return path if path is not None and path.is_file() else None


def export_sizes(manifest):
    return sorted({layout["patch_size"] for slide in manifest["slides"]
                   for layout in slide.get("exports", {}).get("layouts", [])})


def preview_path(run, slide, view):
    if view in PREVIEWS:
        return artifact(run, str(Path(slide["folder"]) / PREVIEWS[view]))
    if view in OPTIONAL_PREVIEWS:
        path = artifact(run, str(Path(slide["folder"]) / OPTIONAL_PREVIEWS[view]))
        return path if path.is_file() else None
    return export_preview(run, slide, *EXPORT_VIEWS[view])


@st.cache_data(max_entries=250, show_spinner=False)
def thumbnail_uri(path_text, modified_ns):
    image = read_preview(Path(path_text)).convert("RGB")
    image.thumbnail((320, 180), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@st.cache_data(max_entries=20, show_spinner=False)
def load_run(manifest_text, modified_ns):
    path = Path(manifest_text)
    if path.stat().st_size > 5_000_000:
        raise ValueError("Manifest exceeds the demo's 5 MB limit")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete" or not 1 <= len(manifest.get("slides", [])) <= 100:
        raise ValueError("Select a completed preprocessing run containing 1–100 slides")
    rows = []
    sizes = export_sizes(manifest)
    for index, slide in enumerate(manifest["slides"]):
        mpp = slide.get("mpp") or (None, None)
        dimensions = slide.get("dimensions") or (None, None)
        levels = slide.get("level_dimensions")
        layouts = {item["patch_size"]: item for item in slide.get("exports", {}).get("layouts", [])}
        patch_counts = {}
        for size in sizes:
            layout = layouts.get(size, {})
            exported, eligible = layout.get("exported_count"), layout.get("eligible_count")
            effective_mpp = layout.get("effective_mpp") or (None, None)
            patch_counts[f"target_mpp_{size}"] = layout.get("target_mpp")
            patch_counts[f"effective_mpp_x_{size}"] = effective_mpp[0]
            patch_counts[f"effective_mpp_y_{size}"] = effective_mpp[1]
            patch_counts[f"source_patch_size_{size}"] = layout.get("source_patch_size")
            patch_counts[f"output_patch_size_{size}"] = layout.get("patch_size")
            patch_counts[f"coords_{size}"] = exported
            patch_counts[f"eligible_{size}"] = eligible
            patch_counts[f"pixel_previews_{size}"] = layout.get("pixel_preview_count", exported)
            patch_counts[f"exported_fraction_{size}"] = 100 * exported / eligible if eligible and exported is not None else None
        if 256 in sizes:
            patch_counts["patch_images_256"] = layouts.get(256, {}).get("patch_image_count")
        if 4096 in sizes:
            region_layout = layouts.get(4096, {})
            patch_counts["regions_4096"] = region_layout.get("region_count",
                region_layout.get("exported_count") if region_layout.get("region_tar") else (0 if region_layout else None))
        dataset = slide.get("project") or Path(slide.get("gallery", "Unknown")).name
        prefilter_excluded = (slide.get("preprocessing") or {}).get("prefilter_excluded_fraction")
        patient_id = next((value.strip() for key in ("patient_id", "case_submitter_id", "case_id")
                           if isinstance(value := slide.get(key), str) and value.strip()), None)
        patient = patient_id or "Unknown"
        # Legacy manifests used name for submitter IDs, but also for anonymous filenames.
        if (patient_id and "patient_id" not in slide and slide.get("name")
                and slide["name"] not in {Path(slide["input"]).name, Path(slide["input"]).stem}):
            patient = slide["name"]
        rows.append({"_index": index, "dataset": dataset, "patient": patient,
            "patient_id": patient_id,
            "slide_id": Path(slide["input"]).stem, "stain": slide.get("stain") or "Unknown",
            "label": slide.get("label") or "Unknown", "sample_id": slide.get("sample_id") or "Unknown",
            "sample_id_source": slide.get("sample_id_source") or "Unknown",
            "case_id": slide.get("case_id") or "Unknown", "file_id": slide.get("file_id") or "Unknown",
            "original_filename": slide.get("file_name") or "Unknown",
            "objective_power": slide.get("objective_power"), "mpp_x": mpp[0], "mpp_y": mpp[1],
            "width_px": dimensions[0], "height_px": dimensions[1],
            "levels": len(levels) if levels is not None else None,
            "vendor": slide.get("vendor") or "Unknown", "scanner_model": slide.get("scanner_model") or "Unknown",
            "file_size_mb": slide["input_bytes"] / 1_000_000 if slide.get("input_bytes") is not None else None,
            "input_bytes": slide.get("input_bytes"), "reader": slide.get("reader") or "Unknown",
            "age_binned": slide.get("age_binned") or "Unknown", "sex": slide.get("sex") or "Unknown",
            "tissue_percent": 100 * slide["tissue_fraction"],
            "retained_percent": 100 * slide["retained_fraction"],
            "excluded_percent": None if slide.get("removed_tissue_fraction") is None else 100 * slide["removed_tissue_fraction"],
            "prefilter_excluded_percent": None if prefilter_excluded is None else 100 * prefilter_excluded,
            **patch_counts,
            "adjudication": slide.get("primary_adjudicated_category") or "Unknown",
            "warnings": "; ".join(slide.get("warnings", [])), "source_path": slide["input"]})
    return manifest, pd.DataFrame(rows)


@st.cache_data(max_entries=100, show_spinner=False)
def is_slide_run(path_text, modified_ns, include_validation=True):
    path = Path(path_text)
    try:
        if path.stat().st_size > 5_000_000:
            return False
        manifest = json.loads(path.read_text())
        if manifest.get("run_kind", "preprocessing") != "preprocessing":
            return False
        if not include_validation:
            purpose = manifest.get("run_purpose")
            legacy_validation = purpose is None and any(
                part.startswith(("validation", "verification", "review")) for part in path.relative_to(OUTPUTS).parts)
            if purpose == "validation" or legacy_validation:
                return False
        slides = manifest.get("slides")
        if manifest.get("status") != "complete" or not isinstance(slides, list) or not 1 <= len(slides) <= 100:
            return False
        required = {"input", "folder", "tissue_fraction", "retained_fraction"}
        if not all(isinstance(slide, dict) and required <= slide.keys() for slide in slides):
            return False
        for slide in slides:
            for filename in PREVIEWS.values():
                image_path = artifact(path.parent, str(Path(slide["folder"]) / filename))
                read_preview(image_path, pixels=False)
        return True
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def discover_runs(include_validation):
    paths = []
    if OUTPUTS.exists():
        for path in OUTPUTS.glob("**/manifest.json"):
            if not path.resolve().is_relative_to(OUTPUTS.resolve()):
                continue
            if "archive" in path.relative_to(OUTPUTS).parts:
                continue
            if is_slide_run(str(path), path.stat().st_mtime_ns, include_validation):
                paths.append(path)
    preferred = {"tcga-mpp05": 0, "kpmp-ckd-demo": 1, "kpmp-demo": 2, "tcga-full-coverage": 3, "tcga-demo": 4}
    return sorted(paths, key=lambda p: (preferred.get(p.parent.name, 5), str(p)))


def main():
    st.set_page_config(page_title="Digital pathology · Slide browser", page_icon="🔬", layout="wide")
    st.title("Slide browser")
    st.caption("Browse tissue previews, compare stains, and export selected slide metadata.")
    with st.sidebar:
        st.header("Browse slides")
        if st.button("Refresh available runs", width="stretch"):
            st.cache_data.clear()
        show_validation = st.checkbox("Show validation runs", value=False)
        runs = discover_runs(show_validation)
        if not runs:
            st.info("No completed runs yet. Run preprocessing first, then refresh.")
            st.stop()
        selected_run = st.selectbox("Preprocessing run", runs,
            format_func=lambda p: str(p.parent.relative_to(OUTPUTS)))
    try:
        manifest, table = load_run(str(selected_run), selected_run.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Cannot open this run: {error}")
        st.stop()
    sizes = export_sizes(manifest)
    mask_method = manifest.get("mask_method") or "Unknown"
    mask_label = {"clam_saturation": "CLAM saturation", "tia_otsu": "TIA Otsu"}.get(mask_method, mask_method)
    tiling_methods = {layout.get("tiling_method") for slide in manifest["slides"]
                     for layout in slide.get("exports", {}).get("layouts", [])}
    tiling_labels = sorted({{"clam_four_pt": "CLAM four-point contours", "coverage_grid": "retained-mask coverage grid"}.get(method, method or "not recorded") for method in tiling_methods})
    st.caption(f"Masking: {mask_label} · Patch placement: {', '.join(tiling_labels) or 'not recorded'}")
    if any(layout.get("target_mpp") is None for slide in manifest["slides"]
           for layout in slide.get("exports", {}).get("layouts", [])):
        st.warning("This run has patch exports without a recorded target resolution. A common patch scale is not established; native scanner resolutions may differ.")
    with st.sidebar:
        for column, label in (("dataset", "Dataset"), ("patient", "Patient"), ("stain", "Stain")):
            options = sorted(table[column].unique())
            chosen = st.multiselect(label, options, default=options, key=f"{selected_run}:{column}")
            table = table[table[column].isin(chosen)]
        query = st.text_input("Search slides", placeholder="Patient, slide ID, label or adjudication")
        if query.strip():
            searchable = table[["patient", "slide_id", "label", "adjudication"]].astype(str).agg(" ".join, axis=1)
            table = table[searchable.str.contains(query.strip(), case=False, regex=False)]
        available_views = list(PREVIEWS) + [view for view in OPTIONAL_PREVIEWS
            if any(preview_path(selected_run.parent, slide, view) for slide in manifest["slides"])]
        available_views += [view for view, (size, kind) in EXPORT_VIEWS.items()
            if any(export_preview(selected_run.parent, slide, size, kind) for slide in manifest["slides"])]
        default_view = "Tissue + holes" if "Tissue + holes" in available_views else "Filtered"
        preview = st.selectbox("Thumbnail view", available_views, index=available_views.index(default_view))
        columns = st.multiselect("Table columns", [c for c in table.columns if c not in {"_index", "source_path"}],
            default=["patient", "dataset", "stain", "sample_id", "objective_power", "mpp_x", "mpp_y", "width_px", "height_px", "levels", "vendor", "file_size_mb"])
        st.caption("Previews use existing small images. Full-resolution navigation remains available in QuPath. Unknown means the run did not record that metadata.")
    a, b, c = st.columns(3)
    a.metric("Slides shown", len(table))
    known_patients = table.loc[table["patient_id"].notna(), ["dataset", "patient_id"]]
    patient_count = len(known_patients.drop_duplicates()) if len(known_patients) or table.empty else "Unknown"
    b.metric("Known patients shown", patient_count)
    known_stains = table.loc[table["stain"] != "Unknown", "stain"]
    stain_count = known_stains.nunique() if len(known_stains) or table.empty else "Unknown"
    c.metric("Stains shown", stain_count)
    if table.empty:
        st.info("No slides match these filters.")
        st.stop()
    table = table.reset_index(drop=True)
    display = table.copy()
    previews = []
    try:
        for index in table["_index"]:
            slide = manifest["slides"][index]
            path = preview_path(selected_run.parent, slide, preview)
            previews.append(thumbnail_uri(str(path), path.stat().st_mtime_ns) if path is not None else None)
    except (OSError, ValueError) as error:
        st.error(f"Cannot read a preview: {error}")
        st.stop()
    display.insert(0, "preview", previews)
    configuration = {
        "preview": st.column_config.ImageColumn("Thumbnail", width="large"),
        "patient": "Patient", "patient_id": "Patient ID", "stain": "Stain", "label": "Label", "dataset": "Dataset",
        "slide_id": "Slide ID", "sample_id": "Sample ID", "sample_id_source": "Sample ID source",
        "case_id": "Case ID", "file_id": "Public file ID", "original_filename": "Original filename",
        "objective_power": "Scanner objective (×)", "mpp_x": "Native MPP X (µm/px)", "mpp_y": "Native MPP Y (µm/px)",
        "width_px": "Width (px)", "height_px": "Height (px)", "levels": "Pyramid levels",
        "vendor": "Vendor", "scanner_model": "Scanner model", "file_size_mb": "File size (MB)",
        "input_bytes": "File size (bytes)", "reader": "Reader", "age_binned": "Age bin", "sex": "Sex",
        "patch_images_256": st.column_config.NumberColumn("256 output px saved PNG patches", format="%d", help="Actual individual PNG patches saved by the exporter. Unknown for historical runs that did not record this count."),
        "regions_4096": st.column_config.NumberColumn("4096 output px saved crops", format="%d", help="Actual PNG regions saved in the region TAR, which may be fewer than exported coordinates."),
        "adjudication": "Primary adjudication", "warnings": "Review warnings",
    }
    for size in sizes:
        configuration[f"target_mpp_{size}"] = st.column_config.NumberColumn(f"{size} px target MPP", format="%.4f", help="Requested output resolution in µm per pixel, separate from native scanner resolution.")
        for axis in ("x", "y"):
            configuration[f"effective_mpp_{axis}_{size}"] = st.column_config.NumberColumn(f"{size} px effective MPP {axis.upper()}", format="%.7f", help="Actual output resolution in µm per pixel after integer native-footprint rounding.")
        configuration[f"source_patch_size_{size}"] = st.column_config.NumberColumn(f"{size} px native footprint", format="%d", help="Square patch edge in level-0 source pixels; coordinates remain level-0 x/y.")
        configuration[f"output_patch_size_{size}"] = st.column_config.NumberColumn(f"{size} px output edge", format="%d", help="Saved square image edge in output pixels, after any recorded resampling.")
        configuration[f"coords_{size}"] = st.column_config.NumberColumn(f"{size} output px coordinates", format="%d", help="Coordinates actually written to HDF5. Older bounded runs may contain only a sampled subset.")
        configuration[f"eligible_{size}"] = st.column_config.NumberColumn(f"{size} output px eligible", format="%d", help="Full grid positions that passed the configured retained-tissue coverage threshold and boundary checks.")
        configuration[f"exported_fraction_{size}"] = st.column_config.NumberColumn(f"{size} px exported %", format="%.1f", help="Exported coordinate count divided by eligible count, not percentage of tissue area. Empty eligible sets have no percentage.")
        configuration[f"pixel_previews_{size}"] = st.column_config.NumberColumn(f"{size} px preview reads", format="%d", help="Number of source patches decoded for the pixel stitch/region previews. This remains bounded even when all eligible coordinates are exported.")
    for field, label in (("tissue_percent", "Candidate tissue %"), ("retained_percent", "Retained %")):
        configuration[field] = st.column_config.NumberColumn(label, format="%.1f", help="Percentage of all thumbnail pixels.")
    configuration["prefilter_excluded_percent"] = st.column_config.NumberColumn("Prefilter excluded %", format="%.1f", help="Pixels excluded by the early CODAIT prefilter as a percentage of all thumbnail pixels. Missing when not recorded.")
    configuration["excluded_percent"] = st.column_config.NumberColumn("Final cleanup %", format="%.1f", help="Candidate-mask tissue removed by final cleanup, divided by candidate-mask tissue pixels. Excludes any earlier CODAIT prefilter exclusion.")
    identity = str(selected_run) + ":" + ",".join(table["_index"].astype(str))
    selection_key = "slides_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    styled = display[["preview", *columns]].style.format({
        "objective_power": "{:.0f}", "mpp_x": "{:.4f}", "mpp_y": "{:.4f}",
        "file_size_mb": "{:.1f}"}, na_rep="Unknown")
    selection = st.dataframe(styled, column_config=configuration,
        hide_index=True, width="stretch", height=600, row_height=140,
        on_select="rerun", selection_mode="multi-row", key=selection_key)
    selected_rows = selection.selection.rows
    selected = table.iloc[selected_rows] if selected_rows else table.iloc[0:0]
    st.caption(f"{len(selected)} selected · Select rows using the left checkboxes. Click column headings to sort.")
    st.caption("Blank preview cells mean that image was not produced for the slide. Coordinate export % compares counts, not tissue area. Pixel stitches and saved crops can show a bounded sample even when the HDF5 contains all eligible coordinates. Older 128-coordinate runs remain labelled by their actual counts.")
    left, right = st.columns(2)
    export_columns = [c for c in table.columns if c != "_index"]
    left.download_button("Download selected metadata", selected[export_columns].to_csv(index=False, na_rep="Unknown"),
        file_name="selected-slides.csv", mime="text/csv", disabled=selected.empty, width="stretch")
    right.download_button("Download filtered metadata", table[export_columns].to_csv(index=False, na_rep="Unknown"),
        file_name="filtered-slides.csv", mime="text/csv", width="stretch")
    st.subheader("Slide detail")
    detail_rows = selected if not selected.empty else table
    indices = detail_rows["_index"].tolist()
    lookup = table.set_index("_index")
    selected_index = st.selectbox("View slide", indices,
        format_func=lambda i: f"{lookup.loc[i, 'patient']} · {lookup.loc[i, 'stain']} · {lookup.loc[i, 'slide_id']}",
        key="detail_" + selection_key + "_" + ",".join(map(str, indices)))
    slide = manifest["slides"][selected_index]
    detail_counts = lookup.loc[selected_index]
    for size in sizes:
        exported, eligible = detail_counts[f"coords_{size}"], detail_counts[f"eligible_{size}"]
        preview_count = detail_counts[f"pixel_previews_{size}"]
        if pd.notna(exported):
            eligible_text = f"{int(eligible):,}" if pd.notna(eligible) else "unknown"
            preview_text = f"{int(preview_count):,}" if pd.notna(preview_count) else "unknown"
            text = f"{size} output px: {int(exported):,} exported coordinates / {eligible_text} eligible · {preview_text} pixel preview reads"
            if size == 256 and pd.notna(detail_counts["patch_images_256"]):
                text += f" · {int(detail_counts['patch_images_256']):,} saved PNG patches"
            if size == 4096 and pd.notna(detail_counts["regions_4096"]):
                text += f" · {int(detail_counts['regions_4096']):,} saved PNG crops"
            st.caption(text)
            target = detail_counts[f"target_mpp_{size}"]
            effective_x, effective_y = detail_counts[f"effective_mpp_x_{size}"], detail_counts[f"effective_mpp_y_{size}"]
            native_size = detail_counts[f"source_patch_size_{size}"]
            output_size = detail_counts[f"output_patch_size_{size}"]
            target_text = f"{target:g}" if pd.notna(target) else "Unknown"
            effective_text = f"{effective_x:.7g} × {effective_y:.7g}" if pd.notna(effective_x) and pd.notna(effective_y) else "Unknown"
            native_text = f"{int(native_size):,} × {int(native_size):,}" if pd.notna(native_size) else "Unknown"
            output_text = f"{int(output_size):,} × {int(output_size):,}" if pd.notna(output_size) else "Unknown"
            st.caption(f"Output: {output_text} px · target: {target_text} µm/px · effective X/Y: {effective_text} µm/px · native footprint: {native_text} level-0 px")
    st.caption("Patch target and effective resolution describe exported images. Scanner objective and native MPP describe the source WSI.")
    if slide.get("warnings"):
        st.warning(" · ".join(slide["warnings"]))
    detail_views = list(PREVIEWS) + [view for view in OPTIONAL_PREVIEWS
        if preview_path(selected_run.parent, slide, view) is not None]
    panels = st.columns(len(detail_views))
    for panel, view in zip(panels, detail_views, strict=True):
        with panel:
            st.image(read_preview(preview_path(selected_run.parent, slide, view)),
                caption=view, width="stretch")
    if "Tissue + holes" in detail_views:
        st.caption("Tissue + holes: green outlines tissue boundaries and blue outlines enclosed-background hole boundaries on the original thumbnail, without fills or tints. These describe mask topology, not anatomical annotations.")
    export_images = [(view, export_preview(selected_run.parent, slide, size, kind))
                     for view, (size, kind) in EXPORT_VIEWS.items()]
    export_images = [(view, path) for view, path in export_images if path is not None]
    if export_images:
        st.subheader("Patch export overviews")
        st.caption("Coordinate coverage uses thumbnail pixels inside exported coordinate footprints. Sampled pixels show only patches actually read from the WSI; these images exist only where pixel previews were saved.")
        for panel, (view, path) in zip(st.columns(len(export_images)), export_images, strict=True):
            with panel:
                st.image(read_preview(path), caption=view, width="stretch")
    with st.expander("Recorded slide metadata"):
        st.caption("Metadata comes from the preprocessing manifest. Unknown values are not inferred. Mask percentages describe thumbnail filtering, not tumour content or diagnoses. KPMP stains are paired by participant; serial-section correspondence is not established.")
        st.json(slide)


if __name__ == "__main__":
    main()
