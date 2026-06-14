"""smpl-transcribe — Whisper speech/lyrics transcription, a smplstream *filter* tool.

The heavy ASR backend (openai-whisper + torch) is isolated in THIS tool's own venv behind
the `whisper` extra (two-tier model). The default install is light: with Whisper absent the
tool runs, passes input frames through, and emits a clean `unsupported` error frame plus a
stderr install hint — it never imports torch/whisper at module top.

The export side (`--format srt|lrc|vtt`) is pure-Python and needs NO heavy dep: it renders
already-produced `marker` (timestamp) + `text` (lyrics) frames into subtitle/lyric files,
so a transcript produced on a GPU box can be exported anywhere.
"""

__version__ = "0.1.0"
