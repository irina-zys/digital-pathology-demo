"""A local, static report with no server or external assets."""

from html import escape



def write_report(output, slides, method):
    sections = []
    for slide in slides:
        folder = escape(slide["folder"], quote=True)
        panels = []
        for filename, title in (("thumbnail", "Original thumbnail"), ("tissue_mask", "Candidate tissue"),
                                ("artifact_mask", "Suspected artefacts"), ("clean_mask", "Retained tissue"),
                                ("filtered", "Filtered preview"), ("overlay", "Review overlay")):
            panels.append(f'<figure><a href="{folder}/{filename}.png"><img loading="lazy" src="{folder}/{filename}.png" alt="{title}"></a><figcaption>{title}</figcaption></figure>')
        for name, title in (("codait_filtered", "After CODAIT filtering"),
                            ("tia_otsu", "TIA preliminary tissue"),
                            ("clam_mask", "CLAM-style tissue"), ("filled_holes", "Filled small holes"),
                            ("before_smoothing", "Before morphological smoothing"),
                            ("smoothed_mask", "After morphological smoothing")):
            path = slide.get("preprocessing_stages", {}).get(name)
            if path:
                safe_path = escape(path, quote=True)
                panels.append(f'<figure><a href="{safe_path}"><img loading="lazy" src="{safe_path}" alt="{title}"></a><figcaption>{title}</figcaption></figure>')
        prefilter_fraction = slide.get("preprocessing", {}).get("prefilter_excluded_fraction")
        prefilter = (f"<p>CODAIT prefilter excluded {prefilter_fraction:.1%} of all thumbnail pixels.</p>"
                     if prefilter_fraction is not None else "")
        warnings = " · ".join(escape(w) for w in slide["warnings"])
        exports = []
        for layout in slide.get("exports", {}).get("layouts", []):
            exports.append(f'<a href="{escape(layout["coords"], quote=True)}">{layout["patch_size"]}-pixel coordinates ({layout["exported_count"]} selected)</a>')
        stain = f" · {escape(slide['stain'])}" if slide.get("stain") else ""
        width, height = slide["dimensions"]
        sections.append(f'''<section><h2>{escape(slide['name'])} <small>{escape(slide['label'])}{stain}</small></h2>
<p>{width:,} × {height:,} pixels · candidate tissue {slide['tissue_fraction']:.1%} · retained {slide['retained_fraction']:.1%}
· final cleanup excluded {slide['removed_pixels']:,} candidate pixels</p>{prefilter}<p class="warning">{warnings}</p>
<div class="grid">{''.join(panels)}</div>
<p><a href="{folder}/metadata.json">Slide metadata and counts</a> ·
<a href="{folder}/removed_tissue.png">Final cleanup mask</a></p>
<p>{' · '.join(exports)}</p></section>''')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>WSI preprocessing demo</title>
<style>body{font:16px/1.55 system-ui,sans-serif;background:#f4f5f3;color:#18342e;margin:0}
main{max-width:1400px;margin:auto;padding:32px}h1{font-size:36px;margin:0}h2{font-size:23px}
small{font-size:14px;background:#e4ede8;padding:6px 10px;border-radius:6px}a{color:#126653}
section{background:white;padding:24px;margin:28px 0;border:1px solid #d9e2dd;border-radius:12px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}figure{margin:0}
img{width:100%;height:270px;object-fit:contain;background:#f8f8f8;border:1px solid #e2e6e4}
figcaption{font-weight:600;padding:8px 0}.warning{color:#98331c}header p{max-width:950px}
@media(max-width:800px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}main{padding:16px}}
@media(max-width:500px){.grid{grid-template-columns:1fr}}</style><main><header>
<p>Digital pathology · preprocessing review</p><h1>From slide to tissue mask</h1>
<p>Green overlay: retained candidate tissue. Orange overlay: candidate tissue excluded during final cleanup.
Masks use white for selected pixels. Filtered previews replace excluded areas with white;
the source slides are unchanged. Click a preview for its PNG.</p>
<p>Pen, dark-pixel and size rules are heuristics. They can remove real tissue and miss artefacts.
This thumbnail review does not detect blur or folds, establish tumour content, or validate a classifier.</p>
'''
    page += f'<p>Method: <strong>{escape(method)}</strong> · {len(slides)} slides · <a href="manifest.json">Run manifest</a></p></header>'
    (output / "index.html").write_text(page + "".join(sections) + "</main></html>\n")
