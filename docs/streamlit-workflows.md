# Streamlit for collaborative review and custom workflows

Streamlit is worth considering when collaborators need to explore Python-generated
results without editing code or running notebook cells. A small app can put
images, metadata, filters and downloads in one browser view. For this project,
that makes it easier to discuss which slides need attention and which outputs
are suitable for the next step.

The app is a presentation layer over saved results. Reusable analysis belongs in
`src/`, configuration in `configs/`, and operational entrypoints in `scripts/`
or the existing Python modules. This keeps the same processing available to the
CLI, notebooks and future interfaces; see [Hydra and project structure](hydra-and-project-structure.md).

## What the existing viewer offers

[app.py](../app.py) reads one completed preprocessing run at a time:

- Filter by dataset, patient or stain, search slides, and choose metadata columns.
- Review thumbnails, masks, artefact overlays and green tissue/blue hole contours.
- Inspect coordinate coverage alongside bounded pixel previews, with recorded
  patch sizes and resolution.
- Select slides and download selected or filtered metadata as CSV.

This supports practical review meetings: choose a KPMP participant, compare
available stain previews, inspect exclusions, then export the slides requiring
follow-up. The detail panels put the original thumbnail beside its derived
outputs. Missing metadata stays unknown, and recorded counts distinguish patch
coordinates from saved images.

The viewer uses small existing images rather than decoding whole slides during
interaction. It does not launch preprocessing, save review decisions or edit
annotations. Use [QuPath](datasets.md#qupath) for full-resolution WSI navigation.

## Sharing with collaborators

**Local use or screen sharing.** From the repository root, run:

```sh
uv sync --locked
uv run --locked streamlit run app.py
```

Open <http://127.0.0.1:8501>. The checked-in configuration binds to localhost and
disables usage telemetry. This address works only on the machine running the
server; it is not a link a colleague can open on their own computer.

**A colleague runs their own copy.** Share the repository and the approved run
artifacts separately. Git excludes slides, bulk `outputs/` run directories and
environments; validated figures/tables embedded in demo notebooks are retained.
Preserve the run's relative artifact paths, and place it under `outputs/` or point the viewer
to a directory containing completed runs:

```sh
DIGITAL_PATHOLOGY_OUTPUTS=/absolute/path/to/runs uv run --locked streamlit run app.py
```

Browsing saved previews does not require copying the original WSIs. Manifests and
CSV downloads can include identifiers and absolute source paths; select suitable
artifacts and metadata before sharing them.

**A hosted team app.** A shared server can offer one URL and one maintained copy
of the app and results. Python executes on that host, which needs access to the
artifacts and sufficient resources for its users; a remote browser does not make
its local files available automatically. See Streamlit's
[client-server architecture](https://docs.streamlit.io/develop/concepts/architecture/architecture).
Hosting requires a deployment environment, artifact storage, and appropriate
access control. The current repository does not configure a hosted service or
user authentication; see the official
[deployment concepts](https://docs.streamlit.io/deploy/concepts).

## Extending it for a custom workflow

A useful next extension could let a reviewer record a slide ID, decision and
reason, or choose a bounded set of slides for reprocessing. These are possible
extensions, not current features. Keep the interface small:

1. Collect explicit inputs and validate them using shared functions in `src/`.
2. For processing, record the resolved configuration and submit a bounded job
   through the existing entrypoint. Keep lengthy WSI processing outside the
   interactive page, with status and logs available to the reviewer.
3. Save durable results or review decisions separately from source slides, keyed
   by run and slide identity. Show the saved record when the page is reopened.

This gives collaborators a focused interface while preserving a reproducible
processing path. A browser selection alone is not a saved review record.

## Practical limits

Streamlit normally reruns the script when widgets change. Cache repeat reads,
include artifact versions or modification times in cache keys, and avoid
starting expensive work on every rerun. This viewer caches manifest and thumbnail
reads and provides **Refresh available runs**. Shared caches also need care if a
future app serves users with different data permissions. See
[Streamlit caching](https://docs.streamlit.io/develop/concepts/architecture/caching).

Choose the interface to match the work: Streamlit for interactive filtering and
review; notebooks for explaining analysis steps; the existing HTML report or
Finder gallery for a fixed snapshot; QuPath for detailed WSI inspection and
annotation. Persistent collaborative editing, large job queues or many concurrent
users need additional storage and service design. An attractive viewer makes
outputs easier to inspect; it does not establish their scientific validity.
