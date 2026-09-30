# Validation record

Verified locally on 29 September 2026. These are observed results, not promised accuracy on other recordings.

## Automated tests

Command: `python3 -m unittest -v test_pipeline.py`

Result: **16 tests passed**, including five integration tests using real FFmpeg/ffprobe. No tests were skipped. Recognition edge cases use injected test doubles; they do not download a model.

Coverage includes WAV decoding, MP3/stereo resampling despite a misleading extension, sample-derived chunk offsets, a shorter final chunk, lazy inference failure, invalid timestamps, capped padded end times, empty recognition, deterministic post-processing, byte and duration limits, corrupt audio rejection, source overwrite protection, atomic writes, decoder timeout handling, and failure without a partial result.

## Actual ASR smoke tests

Model: `tiny.en`, CPU INT8, faster-whisper 1.2.1, English explicitly selected, beam size 5, VAD enabled.

Input: a synthetic spoken sentence generated locally with the operating system's speech synthesizer, encoded as both WAV and MP3. The normalized duration was 5.7669375 seconds.

Reference text:

> Hello. This is a test of the transcription pipeline. Please send the project update tomorrow.

Both runs completed with exit code 0 and `mock: false`. Both recognized:

> Hello, this is a test of the transcription pipeline. Please send the project update tomorrow.

The word sequence matched this short reference; punctuation differed. Each result contained a keyword hit for `update` and a segment covering 0.000 to 5.767 seconds after rounding. The model initially estimated an end of 7.0 seconds, beyond the audio duration. The implementation caps this at the audio end and records `timestamp_clamped: true`. A regression test covers that case.

Actual outputs are saved as `examples/transcript.wav.json` and `examples/transcript.mp3.json`. The audio files are included so this check can be repeated.

```sh
python pipeline.py examples/demo.wav --model tiny.en --language en --keyword update --output wav-result.json
python pipeline.py examples/demo.mp3 --model tiny.en --language en --keyword update --output mp3-result.json
```

## Environment

- Python 3.13.5 on macOS ARM64.
- FFmpeg / ffprobe 9.0.2.
- faster-whisper 1.2.1, CTranslate2 4.8.2.
- `pip check`: no broken requirements.
- Full installed Python versions are recorded in `requirements-tested.txt`; availability can differ by OS and Python version.
- For the downloaded smoke-test model, the Hugging Face cache snapshot revision is recorded in `examples/model-revision.txt`.

## Compatibility recheck: 30 September 2026

Reproduced `transcription_failed` with faster-whisper 1.2.1 and PyAV 19.0.0. The underlying exception was `TypeError: open() got an unexpected keyword argument 'metadata_errors'`. The project now pins PyAV 18.1.0, matching the original tested environment. The shell's `python` alias selected a global installation, so the recheck used the project's explicit `.venv/bin/python` path.

Installed with `.venv/bin/python -m pip install -r requirements.txt -c requirements-tested.txt`. All 16 tests passed with no skips, and `pip check` reported no broken requirements. A real `tiny.en` CPU INT8 run on `examples/demo.wav`, launched in VS Code's terminal, completed and wrote `transcript.json` with `mock: false`, the same text quoted above, and the `update` keyword hit. No Hugging Face token was required.

## Limits of this validation

This is a functional smoke test, not a benchmark or a guarantee of transcription accuracy. It does not establish performance on accents, noise, multiple speakers, other languages, or recordings near the two-hour limit. The default multilingual `base` model was not downloaded or benchmarked in this run; the real-model tests used `tiny.en` to verify the integration. Distributed uploads, persistence, retries, and API endpoints are design proposals, not running services.
