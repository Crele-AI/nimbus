# Crele Nimbus: the inference repository

Context for anyone, person or agent, working in this repository.

**What it is.** The code that runs Crele Nimbus v1: load the weights, stream audio in,
get a yes, no or can't-tell answer per question every 40 ms and events out, plus a
browser playground. The weights live on Hugging Face (`crele/nimbus-v1`) with their model
card; this repository holds no weights, no model card and no training code.

**What never goes in.** Training code, data pipelines, training configuration or metrics,
internal names and paths, and performance numbers beyond the README's table, which is
worded as measured.

**How to work.**

- `uv sync --extra dev` (or `pip install -e ".[dev]"`), then `pytest` for the fast tests
  (tiny random models, no download) and `ruff check` and `ruff format --check` before a
  commit.
- `NIMBUS_WEIGHTS=<folder> pytest tests/test_parity.py` runs the parity test on the real
  weights: the answers on `tests/data/reference.npz` must match the reference to 5e-5
  and the states and events exactly. A change to the encoder, the head, the voice model,
  the text encoder, the event layer or the session is not done until it passes.
- The two CAM++ files under `src/crele_nimbus/speaker/` are copies (`PROVENANCE.md`) and
  are not edited here.
- Commits are Conventional Commits. Plain sentences in docs and comments.
