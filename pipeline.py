#!/usr/bin/env python3
"""Local audio -> timestamped transcript -> deterministic downstream JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unicodedata
import wave

SAMPLE_RATE = 16_000


class PipelineError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def run_media(command: list[str], timeout: int) -> str:
    """Never invoke a shell or include decoder stderr in a public error."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=True
        )
        return completed.stdout
    except FileNotFoundError as exc:
        raise PipelineError("dependency_missing", "Install FFmpeg and ffprobe.") from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("decode_timeout", "Audio inspection or decoding timed out.") from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError("invalid_audio", "Audio could not be inspected or decoded.") from exc


def validate_input(path: Path, max_bytes: int, max_seconds: float) -> dict:
    if not path.is_file():
        raise PipelineError("invalid_input", "Input must be an existing regular file.")
    size = path.stat().st_size
    if size == 0:
        raise PipelineError("empty_input", "Input file is empty.")
    if size > max_bytes:
        raise PipelineError("file_too_large", "Input exceeds the configured byte limit.")
    probe = run_media([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", "a:0", "-show_entries",
        "stream=codec_name,sample_rate,channels:format=duration",
        "-of", "json", str(path),
    ], timeout=30)
    try:
        data = json.loads(probe)
        if not data.get("streams"):
            raise PipelineError("no_audio", "Input contains no audio stream.")
        duration = float(data.get("format", {}).get("duration", "nan"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise PipelineError("invalid_audio", "Invalid audio metadata.") from exc
    if math.isfinite(duration) and duration > max_seconds:
        raise PipelineError("audio_too_long", "Audio exceeds the configured duration limit.")
    return data["streams"][0]


def normalize_audio(source: Path, target: Path, max_seconds: float) -> float:
    # Limit decoded output even when source duration metadata is missing or false.
    run_media([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-xerror",
        "-protocol_whitelist", "file,pipe", "-i", str(source),
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", "-t", str(max_seconds + 1), str(target),
    ], timeout=600)
    with wave.open(str(target), "rb") as audio:
        frames = audio.getnframes()
        duration = frames / audio.getframerate()
    if not frames:
        raise PipelineError("empty_audio", "Decoded audio contains no samples.")
    if duration > max_seconds:
        raise PipelineError("audio_too_long", "Decoded audio exceeds the duration limit.")
    return duration


def audio_chunks(normalized: Path, directory: Path, chunk_seconds: float):
    """Yield one WAV at a time, with offsets derived from integer sample counts."""
    chunk_frames = max(1, int(chunk_seconds * SAMPLE_RATE))
    with wave.open(str(normalized), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, SAMPLE_RATE):
            raise PipelineError("invalid_pcm", "Expected 16 kHz mono 16-bit PCM.")
        consumed = 0
        while pcm := source.readframes(chunk_frames):
            frames = len(pcm) // 2
            path = directory / "chunk.wav"
            with wave.open(str(path), "wb") as target:
                target.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
                target.writeframes(pcm)
            yield path, consumed / SAMPLE_RATE, frames / SAMPLE_RATE
            consumed += frames
            path.unlink(missing_ok=True)


class MockModel:
    """Deterministic plumbing demo only; it does not recognize speech."""
    def transcribe(self, path: str, **options):
        with wave.open(path, "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        segment = SimpleNamespace(start=0.0, end=duration, text="Mock transcription.")
        return iter([segment]), SimpleNamespace(language=options.get("language") or "en")


def create_model(name: str, device: str, compute_type: str, mock: bool, cache: str | None):
    if mock:
        return MockModel()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PipelineError("dependency_missing", "Run pip install -r requirements.txt.") from exc
    try:
        return WhisperModel(name, device=device, compute_type=compute_type, download_root=cache)
    except Exception as exc:
        raise PipelineError("model_load_failed", "Could not load the speech model; check model path, network and device.") from exc


def transcribe_chunks(model, chunks, language: str | None) -> tuple[list[dict], list[str]]:
    result, languages = [], set()
    for chunk_index, (path, offset, duration) in enumerate(chunks):
        try:
            segments, info = model.transcribe(
                str(path), task="transcribe", language=language, beam_size=5,
                vad_filter=True, condition_on_previous_text=False,
            )
            # Iterate inside the try block: faster-whisper performs inference lazily.
            for segment in segments:
                start, end = float(segment.start), float(segment.end)
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end and start <= duration):
                    raise PipelineError("invalid_timestamps", "Recognizer returned invalid segment timestamps.")
                raw_text = segment.text
                if not isinstance(raw_text, str):
                    raise PipelineError("invalid_transcript", "Recognizer returned non-text output.")
                if not raw_text.strip():
                    continue
                languages.add(info.language)
                result.append({
                    "id": len(result), "chunk_index": chunk_index,
                    "start": round(offset + min(start, duration), 3),
                    "end": round(offset + min(end, duration), 3),
                    "timestamp_clamped": end > duration,
                    "text": raw_text,
                })
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError("transcription_failed", f"Transcription failed in chunk {chunk_index}.") from exc
    return result, sorted(languages)


def process_text(segments: list[dict], keywords: list[str]) -> dict:
    # Keep verbatim segment text; do not silently correct names or remove repeated words.
    raw_text = " ".join(item["text"].strip() for item in segments)
    cleaned = " ".join(unicodedata.normalize("NFC", raw_text).split())
    hits = []
    for keyword in dict.fromkeys(word.strip() for word in keywords if word.strip()):
        pattern = re.compile(r"(?<!\w)" + re.escape(unicodedata.normalize("NFC", keyword)) + r"(?!\w)", re.IGNORECASE)
        for segment in segments:
            text = " ".join(unicodedata.normalize("NFC", segment["text"]).split())
            count = len(pattern.findall(text))
            if count:
                hits.append({"keyword": keyword, "segment_id": segment["id"], "count": count,
                             "start": segment["start"], "end": segment["end"]})
    return {"text": raw_text, "cleaned_text": cleaned,
            "token_count": len(re.findall(r"\w+(?:['’]\w+)*", cleaned, re.UNICODE)),
            "keyword_hits": hits}


def atomic_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".transcript-", delete=False) as file:
            temporary = Path(file.name)
            json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_pipeline(args, model=None) -> dict:
    source, output = args.audio.expanduser().resolve(), args.output.expanduser().resolve()
    if source == output or (source.exists() and output.exists() and os.path.samefile(source, output)):
        raise PipelineError("invalid_output", "Output must not overwrite the source audio.")
    metadata = validate_input(source, args.max_bytes, args.max_seconds)
    digest = hashlib.sha256()
    with source.open("rb") as file:
        while data := file.read(1024 * 1024):
            digest.update(data)
    with tempfile.TemporaryDirectory(prefix="transcribe-") as temporary:
        directory = Path(temporary)
        normalized = directory / "normalized.wav"
        duration = normalize_audio(source, normalized, args.max_seconds)
        if model is None:
            model = create_model(args.model, args.device, args.compute_type, args.mock, args.model_cache)
        chunks = audio_chunks(normalized, directory, args.chunk_seconds)
        segments, languages = transcribe_chunks(model, chunks, args.language)
    result = {
        "schema_version": "1.0", "status": "completed", "mock": args.mock,
        "source": {"filename": source.name, "sha256": digest.hexdigest(), "audio_stream": metadata},
        "duration_seconds": round(duration, 3), "timestamp_unit": "seconds",
        "languages": languages,
        "configuration": {"model": "mock" if args.mock else args.model,
                          "device": args.device, "compute_type": args.compute_type,
                          "chunk_seconds": args.chunk_seconds, "language": args.language,
                          "sample_rate": SAMPLE_RATE, "vad_filter": True, "beam_size": 5},
        "segments": segments,
        **process_text(segments, args.keyword),
    }
    atomic_json(output, result)
    return result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("audio", type=Path)
    command.add_argument("--output", type=Path, default=Path("transcript.json"))
    command.add_argument("--model", default="base", help="Whisper model name or local converted-model directory")
    command.add_argument("--language", default=None, help="Language code, e.g. en; omitted means detect per chunk")
    command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    command.add_argument("--compute-type", default="int8")
    command.add_argument("--chunk-seconds", type=float, default=300)
    command.add_argument("--max-seconds", type=float, default=7200)
    command.add_argument("--max-bytes", type=int, default=250 * 1024 * 1024)
    command.add_argument("--keyword", action="append", default=[])
    command.add_argument("--model-cache", default=None)
    command.add_argument("--mock", action="store_true", help="Exercise pipeline with labeled fake text; no speech recognition")
    return command


def main(argv=None) -> int:
    command = parser()
    args = command.parse_args(argv)
    if not math.isfinite(args.chunk_seconds) or not 1 <= args.chunk_seconds <= 900:
        command.error("--chunk-seconds must be between 1 and 900")
    if not math.isfinite(args.max_seconds) or not 1 <= args.max_seconds <= 7200:
        command.error("--max-seconds must be between 1 and 7200")
    if args.max_bytes <= 0:
        command.error("--max-bytes must be positive")
    try:
        run_pipeline(args)
        print(json.dumps({"status": "completed", "output": str(args.output)}))
        return 0
    except PipelineError as exc:
        error = {"code": exc.code, "message": str(exc)}
    except (OSError, EOFError, wave.Error):
        error = {"code": "io_error", "message": "Could not read audio, temporary files or output."}
    except KeyboardInterrupt:
        error = {"code": "cancelled", "message": "Transcription cancelled."}
    print(json.dumps({"error": error}), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
