use std::error::Error;
use std::path::PathBuf;

use log::{info, debug};
use ndarray::{Array, ArrayD, ArrayView1, Axis, Ix3};
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, TensorRTExecutionProvider,
};
use ort::session::{Session, builder::GraphOptimizationLevel};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, long_about = None)]
struct Args {
    /// The path to the input audio file
    #[arg(long)]
    audio_path: PathBuf,

    /// The path to the model file
    #[arg(long)]
    model_path: PathBuf,
}


#[derive(Debug, Clone)]
struct DiarizationConfig {
    offset: f32,
    step: f32,
    sampling_rate: f32,
}

#[derive(Debug, Clone)]
struct SegmentInternal {
    id: usize,    // Speaker ID (class index)
    start: usize, // Start frame index
    end: usize,   // End frame index (exclusive)
    score: f32,   // Accumulated probability score
}

#[derive(Debug, Clone)]
struct SpeakerSegment {
    id: usize,
    start: f32, // Start time in seconds
    end: f32,   // End time in seconds
    confidence: f32,
    // internal: SegmentInternal,
}

// Softmax implementation for a 1D ArrayView
// Applies stable softmax: subtract max before exp
fn softmax(x: ArrayView1<'_, f32>) -> Vec<f32> {
    let array: Vec<f32> = x.iter().map(|&v| v).collect();
    let mut softmax_array = array;

    for value in &mut softmax_array {
        *value = std::f32::consts::E.powf(*value);
    }

    let sum: f32 = softmax_array.iter().sum();

    for value in &mut softmax_array {
        *value /= sum;
    }

    softmax_array
}

// Find the maximum value and its index in a slice
fn find_max(probs: &[f32]) -> (f32, usize) {
    probs.iter().enumerate().fold(
        (f32::NEG_INFINITY, 0),
        |(max_prob, max_idx), (idx, &prob)| {
            if prob > max_prob {
                (prob, idx)
            } else {
                (max_prob, max_idx)
            }
        },
    )
}

// The main post-processing function, translated from JS
fn post_process_speaker_diarization(
    logits_tensor: &ArrayD<f32>, // Input tensor (dynamic dimensions)
    num_samples: usize,          // Original number of audio samples
    config: &DiarizationConfig,
) -> Result<Vec<Vec<SpeakerSegment>>, Box<dyn Error>> {
    let logits_3d = logits_tensor.view().into_dimensionality::<Ix3>()?; // Expects 3D

    let num_frames_float = (num_samples as f32 - config.offset) / config.step;
    if num_frames_float <= 0.0 {
        return Err("Calculated number of frames is zero or negative.".into());
    }
    let ratio = (num_samples as f32 / num_frames_float) / config.sampling_rate;

    let mut all_results: Vec<Vec<SpeakerSegment>> = Vec::new();

    for batch_item_logits in logits_3d.axis_iter(Axis(0)) {
        let mut accumulated_segments: Vec<SegmentInternal> = Vec::new();
        let mut current_speaker: Option<usize> = None;

        for (i, frame_scores_view) in batch_item_logits.axis_iter(Axis(0)).enumerate() {
            let probabilities = softmax(frame_scores_view);
            let (score, id) = find_max(&probabilities);
            let start_frame = i;
            let end_frame = i + 1;

            match current_speaker {
                Some(speaker_id) if speaker_id == id => {
                    if let Some(last_segment) = accumulated_segments.last_mut() {
                        last_segment.end = end_frame;
                        last_segment.score += score;
                    } else {
                        accumulated_segments.push(SegmentInternal {
                            id,
                            start: start_frame,
                            end: end_frame,
                            score,
                        });
                        current_speaker = Some(id);
                    }
                }
                _ => {
                    current_speaker = Some(id);
                    accumulated_segments.push(SegmentInternal {
                        id,
                        start: start_frame,
                        end: end_frame,
                        score,
                    });
                }
            }
        }

        let final_segments = accumulated_segments
            .into_iter()
            .map(|seg| {
                let num_frames_in_segment = seg.end - seg.start;
                let confidence = if num_frames_in_segment > 0 {
                    seg.score / num_frames_in_segment as f32
                } else {
                    0.0
                };

                SpeakerSegment {
                    // internal: seg.clone(),
                    id: seg.id,
                    start: seg.start as f32 * ratio,
                    end: seg.end as f32 * ratio,
                    confidence,
                }
            })
            // filter segments shorter than 0.1s
            .filter(|seg| seg.end - seg.start > 0.1)
            .collect();

        all_results.push(final_segments);
    }

    Ok(all_results)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    let cpu_count = num_cpus::get();

    info!("CPU count: {}", cpu_count);

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(cpu_count)?
        .with_inter_threads(cpu_count)?
        .with_execution_providers([
            TensorRTExecutionProvider::default().build(),
            CUDAExecutionProvider::default().build(),
            CoreMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])?
        .commit_from_file(args.model_path)?;

    let mut wav_reader = hound::WavReader::open(args.audio_path).unwrap();
    if wav_reader.spec().sample_format != hound::SampleFormat::Int {
        panic!("Unsupported sample format. Expect Int.");
    }
    let samples = wav_reader
        .samples()
        .filter_map(|x| x.ok())
        .collect::<Vec<i16>>();

    info!("Audio samples: {:?}", samples.len());

    // input: batch_size, num_channels, samples
    // output: batch_size, num_frames, num_classes

    let input_shape = [1, 1, samples.len()];
    let input_tensor = Array::from_shape_vec(
        input_shape,
        samples.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
    )?;
    let values = ort::inputs![input_tensor.into_dyn()]?;

    let start = std::time::Instant::now();

    let outputs = model.run(values)?;

    let elapsed = start.elapsed();
    info!("Inference time: {:?}", elapsed);

    // Process the outputs
    let logits = &outputs["logits"];

    // info!("Logits: {:?}", logits);

    let x: ndarray::ArrayD<f32> = logits.try_extract_tensor()?.to_owned();

    info!("x shape: {:?}", x.shape());
    // info!("x: {:?}", x);

    let num_samples = samples.len();

    let config = DiarizationConfig {
        offset: 990.0,
        step: 270.0,
        sampling_rate: 16000.0,
    };

    let diarization_results = post_process_speaker_diarization(&x, num_samples, &config)?;

    for (batch_index, segments) in diarization_results.iter().enumerate() {
        info!("Batch {}: ", batch_index);
        for (segment_id, segment) in segments.iter().enumerate() {
            debug!(
                "Speaker ID: {}, Start: {:.2}s, End: {:.2}s, Confidence: {:.2}",
                segment.id, segment.start, segment.end, segment.confidence
            );

            // Use hound to extract the audio segment using samples internal.start and internal.end
            let start_sample = (segment.start * config.sampling_rate) as usize;
            let mut end_sample = (segment.end * config.sampling_rate) as usize;

            if end_sample > samples.len() {
                info!("End sample exceeds audio length, adjusting to max.");
                end_sample = samples.len();
            }

            let total_samples = end_sample - start_sample;

            let segment_samples = &samples[start_sample..end_sample];

            info!(
                "Extracted segment samples: {} to {} ({} samples)",
                start_sample, end_sample, total_samples
            );

            // Here you can save the segment_samples to a file or process them further
            // For example, you can write them to a new WAV file
            let output_path = format!("samples/speaker_{}_segment_{}.wav", segment.id, segment_id);
            let mut writer = hound::WavWriter::create(&output_path, wav_reader.spec())?;
            for sample in segment_samples {
                writer.write_sample(*sample)?;
            }
            writer.finalize()?;

            debug!("Saved segment to: {}", output_path);
        }
    }

    Ok(())
}
