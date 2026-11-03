# Linting and the Git pre-commit hook

[README](../README.md) · [Project structure](hydra-and-project-structure.md)

Ruff checks Python source for issues such as undefined names, unused imports and
selected basic style errors. This project's configuration in
[pyproject.toml](../pyproject.toml) selects `E4`, `E7`, `E9` and `F`, targets Python
3.12, and excludes `third_party/`, `data/`, `outputs/` and `notebooks/`.
The hook checks code without formatting or automatically fixing it.
See [Ruff's linter documentation](https://docs.astral.sh/ruff/linter/).

## Run lint locally

From the repository root, install the locked environment, including the default
`dev` dependency group, then check the working tree:

```sh
uv sync --locked
uv run --locked ruff check .
```

To check one file, replace this example path with a real Python source file:

```sh
uv run --locked ruff check src/example.py
```

Fix reported issues deliberately. For an optional automatic fix on that file:

```sh
uv run --locked ruff check src/example.py --fix
git diff -- src/example.py
```

Review the diff before staging it, then rerun the check. `--fix` changes files;
the ordinary check does not. Do not run fixes on vendored code in `third_party/`.
The exclusions keep vendored code and generated artifacts outside this policy.

## Install the hook once per clone

The repository tracks [.githooks/pre-commit](../.githooks/pre-commit), but Git does
not enable it automatically after cloning. First inspect existing hook settings:

```sh
git config --show-origin --get core.hooksPath
git rev-parse --git-path hooks
```

The first command can return no value when no override is configured. Inspect
the directory printed by the second command for active executable hooks;
`*.sample` files are inactive examples. If another hook path or custom hook is
already in use, preserve it and arrange an explicit integration before changing
the path. Changing `core.hooksPath` replaces the hook directory Git consults.
See [Git's hook documentation](https://git-scm.com/docs/githooks).

When no custom hook setup needs preserving, enable this repository's hook:

```sh
git config --local core.hooksPath .githooks
```

This setting belongs to the local clone and is not distributed by a Git push.
The hook requires `uv` on the PATH of the process running Git, including a GUI
Git client. Run `uv sync --locked` before committing so the project environment
contains Ruff. A missing tool or failed lint check stops the commit.

## What a commit checks

The hook exports every tracked file from Git's index into a temporary directory
using [`git checkout-index`](https://git-scm.com/docs/git-checkout-index) with
`--ignore-skip-worktree-bits`, including files omitted by sparse checkout. It runs
Ruff with the project's uv environment against that snapshot, using the staged
`pyproject.toml` configuration and its exclusions. Ruff itself is installed from
the working checkout's dependency files; stage intended dependency/configuration
changes together. Unchanged tracked files are included, so this checks the whole staged source tree, not only changed lines.

For a partially staged file, Ruff sees the staged version. An unstaged fix can
therefore pass `ruff check .` while the commit still fails. Stage the intended
fix and retry. Likewise, stage an intended lint configuration change before
expecting the hook to apply it.

The hook does not rewrite source, stage files, run a formatter or execute data
preprocessing. There is no separate `pre-commit` framework package. This local
hook is a developer check, not a server-enforced publication policy.

## What linting cannot establish

A clean lint result does not prove that imports succeed, a WSI opens, coordinate
exports match the source, or a scientific assumption is appropriate. Notebook
code is excluded from this lint check and still needs its own bounded execution.

For changes to processing behavior, retain the project's bounded real-entrypoint
execution and artifact inspection described in the
[preprocessing guide](preprocessing.md#validation-and-review). Keep runtime and scientific
validation separate from the static checks performed at commit time.
