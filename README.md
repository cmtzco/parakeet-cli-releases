# parakeet-cli

A lightweight CLI wrapper around [parakeet-rs](https://github.com/altunenes/parakeet-rs) for local speech-to-text transcription using NVIDIA's Parakeet TDT model.

## Features

- **Batch mode**: Transcribe WAV audio files
- **Streaming mode**: Read raw PCM audio from stdin
- **JSONL output**: Structured output with timestamps
- **Fast**: Runs on CPU, optimized for Apple Silicon

## Usage

```bash
# Batch: transcribe a WAV file
parakeet-cli --model-dir ./models/parakeet-tdt-0.6b-v2 --input recording.wav

# Streaming: pipe raw 16kHz mono s16le PCM from stdin
sox -q -d -r 16000 -c 1 -b 16 -e signed-integer -t raw - | \
  parakeet-cli --model-dir ./models/parakeet-tdt-0.6b-v2 --stdin

# Plain text output instead of JSON
parakeet-cli --model-dir ./models/parakeet-tdt-0.6b-v2 --input recording.wav --format text
```

## Output Format

JSONL (one JSON object per line):

```json
{"text":"Hello world.","is_final":true,"duration_secs":0.42,"timestamps":[{"word":"Hello world.","start":0.0,"end":0.4}]}
```

## Models

Download INT8 quantized ONNX models from HuggingFace:

- [parakeet-tdt-0.6b-v2](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx) (English, ~670 MB)
- [parakeet-tdt-0.6b-v3](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) (Multilingual, ~670 MB)

## Pre-built Binaries

Download from [Releases](https://github.com/cmtzco/parakeet-cli-releases/releases):

- `parakeet-cli-aarch64-apple-darwin.tar.gz` — Apple Silicon (M1/M2/M3/M4)

> **Note:** Intel Mac (x86_64) is not currently supported. ONNX Runtime does not provide prebuilt binaries for that platform.

## Building from Source

```bash
# Requires Rust toolchain
cargo build --release
```

## License

MIT
