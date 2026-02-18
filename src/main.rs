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
 * In streaming mode, audio is read from stdin as raw 16kHz mono s16le PCM.
 * Partial transcription results are emitted every ~0.5 seconds of audio
 * as JSONL on stdout, enabling real-time live preview.
 *
 * Output is JSONL on stdout:
 *   {"text":"Hello","is_final":false,"audio_duration_secs":2.1}
 *   {"text":"Hello world.","is_final":false,"audio_duration_secs":4.3}
 *   {"text":"Hello world. How are you?","is_final":true,"duration_secs":0.12}
 */
use clap::Parser;
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use serde::Serialize;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Minimum audio duration (in samples at 16kHz) before first transcription.
/// ~1 second — shorter audio tends to produce garbage.
const MIN_SAMPLES_FOR_TRANSCRIPTION: usize = 16_000;

/// How often to run intermediate transcription (in samples at 16kHz).
/// ~0.5 seconds of audio between each partial result.
/// At ~40-80ms inference on M-series chips, this gives near real-time feel
/// with text updating roughly twice per second.
const CHUNK_INTERVAL_SAMPLES: usize = 8_000;

/// Maximum audio buffer size (in samples at 16kHz).
/// ~3 minutes — TDT models have a ~4-5 min hard limit.
const MAX_BUFFER_SAMPLES: usize = 16_000 * 180;

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
    audio_duration_secs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamps: Option<Vec<TimedTokenOut>>,
}

#[derive(Serialize)]
struct TimedTokenOut {
    word: String,
    start: f32,
    end: f32,
}

/// Emit a JSONL event to stdout and flush immediately.
fn emit_event(event: &TranscriptEvent, format: &str) {
    if format == "text" {
        println!("{}", event.text);
    } else {
        if let Ok(json) = serde_json::to_string(event) {
            println!("{}", json);
        }
    }
    // Flush immediately so the consumer sees partial results without delay
    let _ = io::stdout().flush();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("Loading model from {:?}...", args.model_dir);
    let start = Instant::now();
    let mut model = ParakeetTDT::from_pretrained(&args.model_dir, None)?;
    eprintln!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    if let Some(input_path) = &args.input {
        // ── Batch mode: transcribe a WAV file ──────────────────────
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

        emit_event(
            &TranscriptEvent {
                text: result.text,
                is_final: true,
                duration_secs: Some(elapsed),
                audio_duration_secs: None,
                timestamps: if timestamps.is_empty() {
                    None
                } else {
                    Some(timestamps)
                },
            },
            &args.format,
        );
    } else if args.stdin {
        // ── Streaming mode: read raw PCM from stdin ────────────────
        // Input format: 16kHz, mono, signed 16-bit little-endian (s16le)
        //
        // Strategy: accumulate audio in a growing buffer. Every ~0.5 seconds
        // of new audio, re-transcribe the entire buffer and emit a partial
        // JSONL result. When stdin closes (EOF), emit the final result.
        // ParakeetTDT is stateless so each transcribe call is independent.
        let _start = Instant::now();
        let mut all_audio: Vec<f32> = Vec::new();
        let mut buf = [0u8; 8000]; // Read in small chunks (0.25s) for responsiveness
        let mut samples_since_last_transcription: usize = 0;
        let mut last_text = String::new();

        let stdin_handle = io::stdin();
        let mut handle = stdin_handle.lock();

        // Emit a "ready" event so the consumer knows audio processing has started
        emit_event(
            &TranscriptEvent {
                text: String::new(),
                is_final: false,
                duration_secs: None,
                audio_duration_secs: Some(0.0),
                timestamps: None,
            },
            &args.format,
        );

        loop {
            let bytes_read = match handle.read(&mut buf) {
                Ok(0) => break, // EOF — sox stopped
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
            samples_since_last_transcription += samples.len();

            // Enforce max buffer size (~3 minutes) — trim from the front
            if all_audio.len() > MAX_BUFFER_SAMPLES {
                let excess = all_audio.len() - MAX_BUFFER_SAMPLES;
                all_audio.drain(..excess);
                eprintln!(
                    "Warning: audio buffer exceeded 3 minutes, trimmed oldest {} samples",
                    excess
                );
            }

            // Emit partial transcription every ~0.5 seconds of new audio,
            // but only after we have enough audio for a meaningful result
            if samples_since_last_transcription >= CHUNK_INTERVAL_SAMPLES
                && all_audio.len() >= MIN_SAMPLES_FOR_TRANSCRIPTION
            {
                samples_since_last_transcription = 0;
                let audio_duration = all_audio.len() as f32 / 16000.0;

                match model.transcribe_samples(
                    all_audio.clone(),
                    16000,
                    1,
                    None, // Skip timestamps for partial results (faster)
                ) {
                    Ok(result) => {
                        last_text = result.text.clone();
                        emit_event(
                            &TranscriptEvent {
                                text: result.text,
                                is_final: false,
                                duration_secs: None,
                                audio_duration_secs: Some(audio_duration),
                                timestamps: None,
                            },
                            &args.format,
                        );
                    }
                    Err(e) => {
                        eprintln!("Warning: partial transcription failed: {}", e);
                        // Continue accumulating audio — don't abort
                    }
                }
            }
        }

        // ── Final transcription after EOF ──────────────────────────
        if all_audio.is_empty() {
            emit_event(
                &TranscriptEvent {
                    text: String::new(),
                    is_final: true,
                    duration_secs: Some(0.0),
                    audio_duration_secs: Some(0.0),
                    timestamps: None,
                },
                &args.format,
            );
        } else {
            let final_start = Instant::now();
            match model.transcribe_samples(
                all_audio.clone(),
                16000,
                1,
                Some(TimestampMode::Sentences),
            ) {
                Ok(result) => {
                    let elapsed = final_start.elapsed().as_secs_f32();
                    let audio_duration = all_audio.len() as f32 / 16000.0;

                    let timestamps: Vec<TimedTokenOut> = result
                        .tokens
                        .iter()
                        .map(|t| TimedTokenOut {
                            word: t.text.clone(),
                            start: t.start,
                            end: t.end,
                        })
                        .collect();

                    emit_event(
                        &TranscriptEvent {
                            text: result.text,
                            is_final: true,
                            duration_secs: Some(elapsed),
                            audio_duration_secs: Some(audio_duration),
                            timestamps: if timestamps.is_empty() {
                                None
                            } else {
                                Some(timestamps)
                            },
                        },
                        &args.format,
                    );
                }
                Err(e) => {
                    // If final transcription fails, emit last known good text
                    eprintln!("Error: final transcription failed: {}", e);
                    if !last_text.is_empty() {
                        emit_event(
                            &TranscriptEvent {
                                text: last_text,
                                is_final: true,
                                duration_secs: None,
                                audio_duration_secs: None,
                                timestamps: None,
                            },
                            &args.format,
                        );
                    }
                }
            }
        }
    } else {
        eprintln!("Error: specify either --input <file.wav> or --stdin");
        std::process::exit(1);
    }

    io::stdout().flush()?;
    Ok(())
}
