# Computational pathology: evaluation considerations

[Quick start](../README.md) · [Datasets](datasets.md) ·
[Preprocessing](preprocessing.md) · [Annotation conversion](annotation-conversion.md)

Define the intended use, prediction unit and target population before choosing a
split or metric. A pipeline that reads slides and produces plausible masks has
passed a mechanics check; biological accuracy and generalisability require
separate evidence. The points below are study-design guidance, not validation
results for this demo.

## 1. Generalisability across slides and platforms

WSIs vary with tissue, disease, specimen preparation, fixation, section thickness,
staining, laboratory, scanner, resolution, compression and patient population.
Site-specific image signatures can correlate with the target label, allowing a
model to exploit acquisition differences. This has been demonstrated in TCGA
histology by [Howard et al.](https://doi.org/10.1038/s41467-021-24698-1).

- **Name the shift being tested.** New slides from a familiar laboratory, new
  patients, an unseen scanner and an external hospital are different settings.
  Record patient, specimen, slide, site, stain and scanner identifiers where
  available, along with physical pixel size and preprocessing versions.
- **Inspect confounding and coverage.** Compare label frequencies and case mix
  across sites/platforms; report performance by domain and relevant subgroups.
  If scanner and disease are perfectly confounded, a pooled score cannot
  establish scanner-independent disease recognition.
- **Treat harmonisation as a candidate mitigation.** Physical-scale calibration,
  stain augmentation/normalisation and diverse training data need evaluation on
  untouched domains. Check that preprocessing preserves relevant morphology;
  colour similarity alone does not establish successful transfer.
- **Use paired rescans when possible.** Imaging the same slides on different
  scanners helps isolate technical repeatability. Agreement between scans does
  not prove that either prediction is biologically correct or generalises to
  new patients.

## 2. Match the held-out unit to the generalisability claim

Randomly splitting tiles from the same WSI mainly tests new regions of familiar
slides. Neighbouring or overlapping tiles share tissue and acquisition features.
Group before tiling, and keep derived crops, augmentations, rescans and duplicate
files with their source. Patient grouping is necessary when multiple slides
belong to one patient; see the
[WSI reproducibility study](https://doi.org/10.1371/journal.pdig.0000145).

| Held-out design | Claim it can support | Main limitation |
| --- | --- | --- |
| **WSI-level:** all tiles from each slide stay together | Performance on unseen slides within the sampled setting | If patients overlap, this is transfer to new slides from potentially familiar patients. Serial sections and rescans need grouping too. |
| **Patient-level:** all specimens, slides, stains and visits from a patient stay together | Performance on unseen patients from the represented population and acquisition settings | Shared sites/scanners do not test an unseen platform or institution. |
| **Dataset-/site-level:** reserve an entire independent cohort or institution | Transfer to the specific external population and acquisition setting evaluated | Different dataset names do not guarantee different patients, sites or scanners. Audit overlaps and label definitions; one external cohort does not establish universal transfer. |
| **Platform-level:** reserve a scanner/protocol domain, with patient-disjoint evaluation for new-patient claims | Transfer to that unseen acquisition platform | Platform may be confounded with site or case mix. Paired rescans answer a narrower repeatability question. |

Choose validation folds at the same grouping level as the intended claim where
feasible. Select models, thresholds, stain references and learned preprocessing
using training/validation data; lock the final test set. If target-domain data
are used for adaptation or calibration, declare that setting and reserve separate
patients for its final evaluation. Audit pretrained-model data overlap where
possible and disclose unknown provenance.

Report independent patient/slide/domain counts, not only tile counts. Use
patient-clustered uncertainty estimates when patients contribute multiple
observations, and show domain-specific results; a patient bootstrap within one
hospital cannot quantify uncertainty over unseen hospitals. Missing patient IDs
limit claims to the independence that can actually be verified.

## 3. Cell-level metrics and whole-slide MIL tasks answer different questions

**Cell/nucleus tasks** need references for the objects being evaluated:

- Report detection precision/recall/F1 with an explicit one-to-one matching rule
  and distance or overlap threshold. Assess boundary quality on matched objects
  alongside missed, spurious, merged and split objects. Foreground Dice alone can
  hide instance errors.
- For cell typing, report per-class results and clarify whether classification
  is measured only on matched detections or across the complete pipeline.
  Validate counts, densities or spatial summaries separately if those are the
  intended downstream outputs.
- If reporting panoptic quality (PQ) for benchmark comparability, specify the
  matching threshold, averaging and empty-class handling, and retain separate
  detection and segmentation measures. PQ has documented limitations for small
  nuclei and can obscure distinct failure modes; see
  [Foucart et al.](https://doi.org/10.1038/s41598-023-35605-7).

**Multiple instance learning (MIL)** typically aggregates tile features into a
slide or patient prediction using labels at that level; it can reduce the need
for exhaustive region annotations, as illustrated by
[CLAM](https://doi.org/10.1038/s41551-020-00682-w). Evaluate the actual endpoint:
for classification, report AUROC and precision–recall performance, prevalence,
calibration and sensitivity/specificity at a validation-chosen threshold.
For patient outcomes, predefine how multiple slides form one patient prediction;
survival tasks additionally require censoring-aware evaluation.

A high slide-level score does not validate cell boundaries or types, and accurate
cell segmentation does not establish a useful slide-level predictor. Attention
heatmaps identify candidate regions for review; they are not automatically
validated localisation maps or causal explanations. Localisation claims require
independent region-level assessment. Keep the **prediction unit**, **split unit**
and **uncertainty/resampling unit** explicit even when they differ.

## 4. Mitigate limited or uncertain ground truth

- **Build a small, independent reference subset.** Sample across patients,
  platforms, classes and difficult morphology. Use written annotation rules,
  blinded expert review and, where feasible, multiple readers with adjudication.
  Retain disagreement and uncertain regions rather than treating consensus as
  infallible. Keep a representative evaluation subset separate from targeted
  hard-case review or active-learning samples.
- **Match the reference to the claim.** Nucleus masks do not establish whole-cell
  boundaries; a slide diagnosis does not label every tile. IHC, molecular assays
  or outcomes can provide complementary evidence when they measure the intended
  target. Check specimen identity, timing and registration; serial sections are
  not identical cell-by-cell references.
- **Use weak supervision for training with explicit limits.** Slide labels,
  pseudo-labels and model-assisted annotation can reduce annotation cost, but
  a model's own predictions or agreement between related models are not an
  independent accuracy benchmark. Audit a held-out subset with external evidence.
- **Prespecify missing/ambiguous-label handling.** Unannotated regions are not
  automatically negative. State ignore rules, excluded area/object counts and
  the population to which the score applies. NuInsSeg provides explicit vague
  area masks for this purpose; see its
  [dataset paper](https://doi.org/10.1038/s41597-024-03117-2) and the local
  [mask guidance](datasets.md#nuinsseg).
- **Use label-free checks to expose failures.** Rescan consistency, perturbation
  sensitivity, tissue coverage and plausible size/count distributions can flag
  problems. They establish stability or plausibility, not accuracy. Without a
  suitable independent reference, report that accuracy remains unmeasured.

For this project, TCGA/KPMP examples support preprocessing review, while
CytoNuke/NuInsSeg provide patch-level object annotations with different scopes.
They do not constitute one shared benchmark. See the
[dataset guide](datasets.md) for provenance and annotation limits; successful
conversion or visually plausible output does not extend those claims.
