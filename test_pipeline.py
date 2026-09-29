"""Contract and decoder integration tests; no model download is needed."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import pipeline as p


def make_wav(path, frames=40000, rate=16000, channels=1):
    with wave.open(str(path), "wb") as file:
        file.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        file.writeframes(struct.pack("<h", 200) * frames * channels)


class FileFixture:
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.audio = self.root / "audio.wav"
        self.output = self.root / "result.json"
        make_wav(self.audio)

    def tearDown(self):
        self.directory.cleanup()

    def args(self, *extra):
        return p.parser().parse_args([str(self.audio), "--output", str(self.output), "--mock", *extra])


class PipelineTests(FileFixture, unittest.TestCase):
    def test_sample_offsets_and_short_final_chunk(self):
        offsets, durations, samples = [], [], 0
        for path, offset, duration in p.audio_chunks(self.audio, self.root, 1):
            offsets.append(offset)
            durations.append(duration)
            with wave.open(str(path), "rb") as audio:
                samples += audio.getnframes()
        self.assertEqual(offsets, [0, 1, 2])
        self.assertEqual(durations, [1, 1, 0.5])
        self.assertEqual(samples, 40000)

    def test_global_timestamps(self):
        chunks = [(self.audio, 300.0, 2.5)]
        segments, languages = p.transcribe_chunks(p.MockModel(), chunks, "en")
        self.assertEqual((segments[0]["start"], segments[0]["end"]), (300, 302.5))
        self.assertEqual(languages, ["en"])

    def test_lazy_model_failure_is_caught(self):
        def failing_segments():
            raise RuntimeError("failed while iterating")
            yield
        model = SimpleNamespace(transcribe=lambda *a, **k: (failing_segments(), SimpleNamespace(language="en")))
        with self.assertRaises(p.PipelineError) as failure:
            p.transcribe_chunks(model, [(self.audio, 0, 2.5)], None)
        self.assertEqual(failure.exception.code, "transcription_failed")

    def test_invalid_model_timestamps(self):
        for start, end in [(2, 1), (-1, 1), (0, float("nan")), (100, 101)]:
            with self.subTest(start=start, end=end):
                model = SimpleNamespace(transcribe=lambda *a, **k: (
                    iter([SimpleNamespace(start=start, end=end, text="bad")]), SimpleNamespace(language="en")))
                with self.assertRaises(p.PipelineError) as failure:
                    p.transcribe_chunks(model, [(self.audio, 0, 2.5)], None)
                self.assertEqual(failure.exception.code, "invalid_timestamps")

    def test_model_padding_is_clamped_and_flagged(self):
        model = SimpleNamespace(transcribe=lambda *a, **k: (
            iter([SimpleNamespace(start=0, end=7, text="speech")]), SimpleNamespace(language="en")))
        segments, _ = p.transcribe_chunks(model, [(self.audio, 300, 2.5)], "en")
        self.assertEqual(segments[0]["end"], 302.5)
        self.assertTrue(segments[0]["timestamp_clamped"])

    def test_empty_recognition_is_valid(self):
        model = SimpleNamespace(transcribe=lambda *a, **k: (iter([]), SimpleNamespace(language="en")))
        segments, languages = p.transcribe_chunks(model, [(self.audio, 0, 2.5)], None)
        self.assertEqual(segments, [])
        self.assertEqual(languages, [])
        self.assertEqual(p.process_text(segments, ["hello"])["cleaned_text"], "")

    def test_cleanup_preserves_original_and_keyword_boundaries(self):
        text = "  Cafe\u0301\n update. UPDATE updated.  "
        segments = [{"id": 0, "start": 0, "end": 2, "text": text}]
        result = p.process_text(segments, ["update", "café", "update"])
        self.assertEqual(segments[0]["text"], text)
        self.assertEqual(result["cleaned_text"], "Café update. UPDATE updated.")
        self.assertEqual(result["token_count"], 4)
        self.assertEqual([hit["count"] for hit in result["keyword_hits"]], [2, 1])

    def test_bad_json_does_not_replace_previous_result(self):
        self.output.write_text("previous result", encoding="utf-8")
        with self.assertRaises(ValueError):
            p.atomic_json(self.output, {"number": float("nan")})
        self.assertEqual(self.output.read_text(), "previous result")
        self.assertEqual(list(self.root.glob(".transcript-*")), [])

    def test_missing_empty_and_oversized_input(self):
        missing = self.root / "missing.wav"
        empty = self.root / "empty.wav"
        empty.touch()
        for path, limit, expected in [(missing, 100, "invalid_input"), (empty, 100, "empty_input"),
                                      (self.audio, 100, "file_too_large")]:
            with self.subTest(expected=expected), self.assertRaises(p.PipelineError) as failure:
                p.validate_input(path, limit, 60)
            self.assertEqual(failure.exception.code, expected)

    def test_input_cannot_be_overwritten(self):
        args = self.args()
        args.output = self.audio
        before = self.audio.read_bytes()
        with self.assertRaises(p.PipelineError) as failure:
            p.run_pipeline(args)
        self.assertEqual(failure.exception.code, "invalid_output")
        self.assertEqual(self.audio.read_bytes(), before)

    def test_missing_decoder_and_timeout(self):
        for exception, code in [(FileNotFoundError(), "dependency_missing"),
                                 (subprocess.TimeoutExpired("ffprobe", 30), "decode_timeout")]:
            with self.subTest(code=code), patch("pipeline.subprocess.run", side_effect=exception):
                with self.assertRaises(p.PipelineError) as failure:
                    p.run_media(["ffprobe"], 30)
                self.assertEqual(failure.exception.code, code)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe required")
class DecoderIntegrationTests(FileFixture, unittest.TestCase):
    def test_wav_pipeline_multichunk(self):
        result = p.run_pipeline(self.args("--chunk-seconds", "1", "--keyword", "mock"))
        self.assertEqual(result["duration_seconds"], 2.5)
        self.assertTrue(result["mock"])
        self.assertEqual([s["start"] for s in result["segments"]], [0, 1, 2])
        self.assertEqual(result["segments"][-1]["end"], 2.5)
        self.assertEqual(json.loads(self.output.read_text()), result)

    def test_mp3_stereo_nonstandard_rate_and_misleading_extension(self):
        make_wav(self.audio, frames=44100, rate=44100, channels=2)
        disguised = self.root / "recording.bin"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.audio), "-f", "mp3", str(disguised)], check=True)
        args = self.args()
        args.audio = disguised
        result = p.run_pipeline(args)
        self.assertAlmostEqual(result["duration_seconds"], 1, places=2)
        self.assertEqual(result["source"]["audio_stream"]["codec_name"], "mp3")

    def test_corrupt_media_rejected(self):
        self.audio.write_bytes(b"This is not audio")
        with self.assertRaises(p.PipelineError) as failure:
            p.run_pipeline(self.args())
        self.assertEqual(failure.exception.code, "invalid_audio")
        self.assertFalse(self.output.exists())

    def test_metadata_and_decoded_duration_limits(self):
        with self.assertRaises(p.PipelineError) as failure:
            p.validate_input(self.audio, 1_000_000, 1)
        self.assertEqual(failure.exception.code, "audio_too_long")
        with self.assertRaises(p.PipelineError) as failure:
            p.normalize_audio(self.audio, self.root / "normalized.wav", 1)
        self.assertEqual(failure.exception.code, "audio_too_long")

    def test_failure_does_not_publish_partial_transcript(self):
        def fail(*a, **k):
            raise RuntimeError("inference error")
        with self.assertRaises(p.PipelineError):
            p.run_pipeline(self.args(), model=SimpleNamespace(transcribe=fail))
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
