# Educational Shorts

An end-to-end Python pipeline for generating short educational videos with a local language model.

The project turns a structured knowledge tree into narrated vertical videos through topic selection, script generation, retrieval-based fact checking, text-to-speech, captions, and FFmpeg video rendering.

## Pipeline

1. Build or expand a knowledge tree.
2. Generate educational topic candidates.
3. Store approved topics in a SQLite topic library.
4. Generate a structured outline.
5. Generate and edit the narration script.
6. Extract and verify factual claims using retrieved web evidence.
7. Rewrite unsupported or overstated claims.
8. Generate publication metadata.
9. Synthesize narration with Kokoro TTS.
10. Generate timed captions.
11. Render a vertical video with gameplay footage.
12. Record the result and topic status in SQLite.

Notebook 12 runs the complete optimized pipeline.

## Main Features

- Local LLM generation through Ollama
- Structured Pydantic outputs
- SQLite topic queue
- Retrieval-grounded fact checking
- Parallel evidence retrieval
- Cached claims, evidence, and fact-check decisions
- Manual-review publication gate
- Kokoro text-to-speech
- ASS caption generation
- FFmpeg vertical video rendering
- Intermediate artifact reuse after interrupted runs

## Requirements

- Python 3.12
- Ollama
- The `qwen3:8b` Ollama model
- FFmpeg and FFprobe
- Jupyter Notebook or JupyterLab
- Local background gameplay footage

The configured Ollama model can be changed in:

```text
educational_shorts/config.py
```

## Project Structure

```text
EDUCATIONAL_SHORTS/
├── educational_shorts/     # Reusable pipeline modules
├── notebooks/              # Numbered development and pipeline notebooks
├── prompts/                # LLM system prompts
├── data/                   # Local generated data; ignored by Git
├── .gitignore
└── README.md
```

Generated scripts, databases, audio, captions, videos, caches, and gameplay footage are stored locally under `data/` and are not committed to Git.

## Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/trevorhitchcock/EDUCATIONAL_SHORTS.git
cd EDUCATIONAL_SHORTS
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the Python dependencies after creating the dependency manifest:

```bash
pip install -r requirements.txt
```

Install and start Ollama, then download the configured model:

```bash
ollama pull qwen3:8b
```

Confirm FFmpeg is available:

```bash
ffmpeg -version
ffprobe -version
```

Place background footage under:

```text
data/gameplay/subway_surfers/
```

## Running the Pipeline

Open:

```text
notebooks/12_batch_pipeline.ipynb
```

Run the notebook from top to bottom.

Notebook 12 claims the next approved topic from:

```text
data/topic_library.db
```

A successful topic is marked completed. A topic whose fact-check report requires human review is rendered as a preview but is not marked completed.

## Topic States

The topic library uses these workflow states:

- `approved`: ready to process
- `processing`: currently claimed by a pipeline run
- `completed`: successfully rendered and accepted
- `failed`: failed during processing or held for manual review
- `rejected`: intentionally excluded

## Safety and Fact Checking

The pipeline extracts important factual claims and retrieves evidence for each one. Claims may be classified as:

- `supported`
- `contradicted`
- `overstated`
- `insufficient_evidence`

Evidence quotes and URLs are validated in Python after the model responds. When post-validation rejects a previously supported claim, the script receives an additional correction-only rewrite.

Videos requiring manual review are prevented from being recorded as completed.

## Current Status

The project successfully generates complete educational shorts locally, including narration, captions, metadata, and final vertical video output.

The primary remaining work is automated testing, dependency packaging, improved retrieval filtering, and optional cloud-model support.
