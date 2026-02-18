/**
 * parakeet-cli
 *
 * Thin CLI wrapper around the parakeet-rs library for local
 * speech-to-text transcription using NVIDIA Parakeet TDT.
 *
 * Supports two modes:
 *   1. Batch: parakeet-cli --model-dir <path> --input <file.wav>
 *   2. Streaming (stdin): parakeet-cli --model-dir <path> --stdin
 *
 * Output is JSONL on stdout:
 *   {"text":"Hello world.","is_final":true}
 */
use clap::Parser;
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use serde::Serialize;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "parakeet-cli", version, about = "Speech-to-text using NVIDIA Parakeet TDT")]
struct Args {
    /// Path to the model directory containing encoder/decoder ONNX files and vocab.txt
    #[arg(long)]
    model_dir: PathBuf,

    /// Path to a WAV audio file to transcribe (batch mode)
    #[arg(long)]
    input: Option<PathBuf>,

    /// Read raw 16kHz mono s16le PCM audio from stdin (streaming mode)
    #[arg(long)]
    stdin: bool,

    /// Output format: "json" for JSONL, "text" for plain text
    #[arg(long, default_value = "json")]
    format: String,
}

#[derive(Serialize)]
struct TranscriptEvent {
    text: String,
    is_final: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_secs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamps: Option<Vec<TimedTokenOut>>,
}

#[derive(Serialize)]
struct TimedTokenOut {
    word: String,
    start: f32,
    end: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("Loading model from {:?}...", args.model_dir);
    let start = Instant::now();
    let mut model = ParakeetTDT::from_pretrained(&args.model_dir, None)?;
    eprintln!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    if let Some(input_path) = &args.input {
        // Batch mode: transcribe a WAV file
        let start = Instant::now();
        let result = model.transcribe_file(input_path, Some(TimestampMode::Sentences))?;
        let elapsed = start.elapsed().as_secs_f32();

        let timestamps: Vec<TimedTokenOut> = result
            .tokens
            .iter()
            .map(|t| TimedTokenOut {
                word: t.text.clone(),
                start: t.start,
                end: t.end,
            })
            .collect();

        if args.format == "text" {
            println!("{}", result.text);
        } else {
            let event = TranscriptEvent {
                text: result.text,
                is_final: true,
                duration_secs: Some(elapsed),
                timestamps: if timestamps.is_empty() {
                    None
                } else {
                    Some(timestamps)
                },
            };
            let json = serde_json::to_string(&event)?;
            println!("{}", json);
        }
    } else if args.stdin {
        // Streaming mode: read raw PCM from stdin in chunks
        // Input format: 16kHz, mono, signed 16-bit little-endian
        let start = Instant::now();
        let mut all_audio: Vec<f32> = Vec::new();
        let mut buf = [0u8; 32000]; // 1 second of 16kHz s16le audio

        let stdin = io::stdin();
        let mut handle = stdin.lock();

        loop {
            let bytes_read = match handle.read(&mut buf) {
                Ok(0) => break, // EOF
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            };

            // Convert s16le bytes to f32 samples
            let samples: Vec<f32> = buf[..bytes_read]
                .chunks_exact(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    sample as f32 / 32768.0
                })
                .collect();

            all_audio.extend_from_slice(&samples);
        }

        if all_audio.is_empty() {
            let event = TranscriptEvent {
                text: String::new(),
                is_final: true,
                duration_secs: Some(0.0),
                timestamps: None,
            };
            if args.format == "text" {
                println!();
            } else {
                println!("{}", serde_json::to_string(&event)?);
            }
            return Ok(());
        }

        // Transcribe the accumulated audio
        let result =
            model.transcribe_samples(all_audio, 16000, 1, Some(TimestampMode::Sentences))?;
        let elapsed = start.elapsed().as_secs_f32();

        let timestamps: Vec<TimedTokenOut> = result
            .tokens
            .iter()
            .map(|t| TimedTokenOut {
                word: t.text.clone(),
                start: t.start,
                end: t.end,
            })
            .collect();

        if args.format == "text" {
            println!("{}", result.text);
        } else {
            let event = TranscriptEvent {
                text: result.text,
                is_final: true,
                duration_secs: Some(elapsed),
                timestamps: if timestamps.is_empty() {
                    None
                } else {
                    Some(timestamps)
                },
            };
            let json = serde_json::to_string(&event)?;
            println!("{}", json);
        }
    } else {
        eprintln!("Error: specify either --input <file.wav> or --stdin");
        std::process::exit(1);
    }

    io::stdout().flush()?;
    Ok(())
}
