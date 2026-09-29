# Audio transcription pipeline

A Python CLI that accepts an audio file, transcribes speech, returns timestamps per segment, and prepares deterministic text features for downstream use. Part 1 is implemented. The distributed service described below is a design proposal for Part 2, not an implemented API.

## Quick start

Use Python 3.10+ and install FFmpeg, including `ffprobe`, through your operating system's package manager. Both commands must be on `PATH`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python pipeline.py examples/demo.wav --language en --keyword update --output transcript.json
```

On Windows, activate with `.venv\Scripts\activate` instead. A named model is downloaded on first use; subsequent runs reuse its cache. To control where models are stored, pass `--model-cache .model-cache`. To run offline after provisioning a model, use its local converted-model directory as `--model`.

The default is the multilingual `base` model on CPU with INT8 computation. A larger model may improve recognition on some data at greater compute cost; choose using measurements on representative recordings. The smaller `tiny.en` is useful for an English smoke test:

```sh
python pipeline.py examples/demo.mp3 --model tiny.en --language en --keyword update --output transcript.json
```

Mock mode exercises decoding, chunking, serialization, and post-processing without installing an ASR model. It deliberately returns fake text, including for silent input; `mock: true` identifies it in the result.

```sh
python pipeline.py examples/demo.wav --mock --chunk-seconds 3 --output mock.json
python -m unittest -v test_pipeline.py
```

## Implementation

```text
Local audio
  -> validate path and byte size
  -> inspect the first audio stream with ffprobe
  -> decode to 16 kHz mono 16-bit PCM WAV on temporary disk
  -> read bounded chunks, reuse one speech model
  -> add sample-derived offsets to ASR segment timestamps
  -> preserve raw text; derive cleaned text and keyword matches
  -> atomically publish a versioned JSON result
```

`pipeline.py` contains these stages as separate functions. `run_pipeline` coordinates them, while `main` handles arguments and structured errors. Tests inject a recognizer so failures and edge cases can be checked deterministically without a network download.

### Input and formats

The CLI accepts a local path, for example WAV or MP3. The file must exist, be a non-empty regular file, and fit the default 250 MiB limit. The decoder examines actual media, so changing a filename extension does not turn an invalid file into valid audio. Inputs with no audio stream are rejected.

FFmpeg normalizes sample rate, channel count, and sample representation. The prototype selects the first audio stream and downmixes its channels to mono. For a recording with one participant per channel, a production variant should preserve and transcribe channels separately. Resampling cannot restore information missing from the source.

External media processes receive argument lists without `shell=True`, have timeouts, and are restricted to local file/pipe protocols. This is not a complete hostile-media sandbox. An Internet-facing worker should run decoders in an isolated container with restricted filesystem access, no network, and CPU/memory/disk limits.

### Recognition and timestamps

The backend is [faster-whisper](https://github.com/SYSTRAN/faster-whisper), using its `WhisperModel.transcribe` API. The model is loaded once per run and reused across chunks. Transcription uses beam size 5, voice activity detection, and `task="transcribe"`. The segment generator is consumed inside error handling because inference occurs during iteration.

An explicit language avoids repeated detection. If omitted, the backend detects language independently for each chunk. `languages` records detected languages only for chunks producing text; it is not a guarantee of complete language identification. Silence may produce an empty transcript successfully. VAD reduces unwanted non-speech output but does not guarantee that hallucinations cannot occur.

Segment times are seconds relative to the beginning of the decoded recording. For a chunk beginning after `consumed_samples`:

```python
offset = consumed_samples / 16000
global_start = offset + segment.start
global_end = offset + segment.end
```

Offsets come from integer sample counts, including a shorter final chunk. Timestamp values are checked for finiteness and order, and starts must be within the chunk. A model can extend an estimated end into its padded audio window: the implementation caps that end at the chunk duration and sets `timestamp_clamped: true` for transparency. Rounding to three decimals is a serialization choice, not a claim of millisecond accuracy. These are estimated segment boundaries, not forced-aligned word times.

### Long recordings and resource limits

The default chunk length is 300 seconds. Decoded audio is stored on temporary disk, and only one chunk is passed to ASR at a time. The model and one chunk dominate inference memory; the list of output segments still grows with recording length. This is a bounded-file batch pipeline, not live streaming.

The maximum duration is two hours and can be reduced with `--max-seconds`. The decoder stops after the configured maximum plus one second, and the actual decoded sample count is checked. This prevents untrusted or absent duration metadata from producing unbounded PCM. At 16 kHz, mono, 16-bit, PCM uses roughly 115 MB per hour, so the two-hour cap needs roughly 230 MB of temporary normalized audio, plus a chunk and model files. Temporary files are removed on success and ordinary exception paths.

Fixed chunk boundaries can cut through a word, and independent chunks lose cross-boundary linguistic context. The prototype makes that tradeoff explicit. A higher-quality service should split at detected silence or use overlap with timestamp-aware duplicate reconciliation. Simply concatenating overlapping transcripts duplicates speech. No overlap reconciliation, diarization, chunk checkpoints, or inference deadline is implemented here.

### Downstream processing

Each segment's `text` retains the recognizer's output. The top-level `text` joins trimmed segment texts for convenience. `cleaned_text` applies Unicode NFC normalization and collapses whitespace. It does not rewrite names, remove repeated words, or invent punctuation.

`token_count` is a simple Unicode regex token count, not a language-aware word count. Optional repeated `--keyword` arguments produce case-insensitive boundary matches within each segment. Keyword hit timestamps refer to the containing segment; they are not word timestamps. Phrases spanning segment boundaries are not matched. These limitations keep processing predictable and auditable.

### Result schema

```json
{
  "schema_version": "1.0",
  "status": "completed",
  "mock": false,
  "duration_seconds": 4.0,
  "timestamp_unit": "seconds",
  "languages": ["en"],
  "segments": [
    {"id": 0, "chunk_index": 0, "start": 0.2, "end": 2.8, "text": " Project update."}
  ],
  "text": "Project update.",
  "cleaned_text": "Project update.",
  "token_count": 2,
  "keyword_hits": [
    {"keyword": "update", "segment_id": 0, "count": 1, "start": 0.2, "end": 2.8}
  ]
}
```

The values above are illustrative. Actual results also include source SHA-256, source audio metadata, and inference configuration. The checksum aids traceability; it is not a security credential or proof of ownership. For reproducible deployment, pin the model artifact/revision and the complete environment as well as the application version.

### Errors and output integrity

The CLI exits 0 on success, 1 on processing failure, and 2 on invalid command-line arguments. Processing errors are JSON on stderr with a stable code and a short message. Decoder diagnostics and transcript contents are not logged routinely. A result is written to a temporary file in the output directory, flushed, then replaced atomically after the whole job succeeds. A failed transcription does not publish a partial result. An existing output is replaced on success; a failure leaves any old output intact, so callers must check the exit status. The output cannot be the input audio or a hard link to it.

## Part 2: proposed service design

### Concurrent uploads

Use authenticated upload sessions and short-lived presigned URLs for a private object store. Clients upload directly, keeping audio traffic out of inference workers. Verify object existence, ownership, and limits before accepting a job. Persist a job and transactional outbox event together, then dispatch to a durable queue. A bounded worker pool reuses loaded models and limits concurrency according to measured CPU/GPU memory. Apply tenant quotas, idempotency keys, rate limits, queue limits, and backpressure. Scale on queue age and processing latency. An async route alone does not prevent inference from blocking a web server.

### Storage

Store original audio and large transcript JSON in private object storage. Keep job ownership, status, object references, attempts, timestamps, checksum, model/configuration version, and safe error codes in PostgreSQL. Index segments separately if search is needed. Keep raw and processed text separate. Authorize every access and issue short-lived download URLs. Use TLS, encryption at rest, retention periods, and deletion covering derived artifacts. Keep private speech out of routine logs.

### Retries and recovery

Retry transient timeouts, throttling, storage failures, and worker loss with capped exponential backoff and jitter. Honor server retry hints. Do not blindly retry invalid media or authorization failures. Use leases and heartbeats to reclaim abandoned jobs. Checkpoint chunks under job ID, input checksum, model/configuration version, and chunk index. Queue delivery can repeat, so writes must be idempotent. Conditional status transitions and fencing tokens stop stale workers from overwriting a newer attempt. Acknowledge messages only after durable writes. Exhausted retries produce a failed job and a dead-letter entry for investigation.

### API

| Endpoint | Behavior |
| --- | --- |
| `POST /v1/uploads` | Authenticate; create upload session and return object ID and presigned URL. |
| `POST /v1/transcriptions` | Validate owned, completed upload and options; accept `Idempotency-Key`; return `202`, job ID and `Location`. |
| `GET /v1/transcriptions/{id}` | Authorize owner; return queued/processing/completed/failed, progress and safe errors. |
| `GET /v1/transcriptions/{id}/result` | Authorize owner; return versioned text and timestamped segments when ready. |

Small-file multipart input is an optional alternative. Validate request schemas and size/duration limits. Use suitable 4xx responses for client errors, 429 for throttling, and 503 for temporary unavailability. Optional signed webhooks can report completion and must themselves tolerate retries and duplicates. These endpoints are design only.

## Validation and tradeoffs

Run `python -m unittest -v test_pipeline.py`. Tests cover sample offsets and final chunks, timestamp validation, lazy failures, no-speech results, raw-text preservation, keyword matching, atomic output, file/byte limits, source overwrite protection, decoder timeouts, real WAV/MP3 decoding, stereo resampling, corrupt input, duration limits, and no partial publication after inference failure. Decoder tests skip explicitly if FFmpeg tools are unavailable.

See `VALIDATION.md` for the actual test run and real-model smoke-test results. Mock tests validate pipeline behavior, not recognition accuracy. A smoke test on synthetic speech is not an accuracy benchmark. Before production, measure word error rate against human-reviewed reference transcripts, timestamp error, latency, real-time factor, peak memory, and failure rate across target languages, accents, silence, noise, and long recordings. Include speech crossing chunk boundaries. No model or implementation can promise perfect transcription for arbitrary audio.

## Repository contents

- `pipeline.py`: runnable implementation.
- `test_pipeline.py`: deterministic contracts and real decoder tests.
- `requirements.txt`: pinned ASR dependency; `requirements-tested.txt` records the tested environment.
- `examples/`: synthetic example audio and actual example output.
- `VALIDATION.md`: what was actually verified.

## References

- [faster-whisper API and installation](https://github.com/SYSTRAN/faster-whisper)
- [FFmpeg command-line documentation](https://ffmpeg.org/ffmpeg.html)
- [Python PCM WAV reader](https://docs.python.org/3/library/wave.html)
