#pragma once

#include <vector>
#include <string>

// Audio preprocessing: WAV loading, resampling, mel spectrogram.
// All CPU-side C++ — no Python, no external FFmpeg.

struct WavFile {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;  // f32 samples interleaved if multi-channel
};

struct MelSpectrogram {
    int n_frames = 0;       // time dimension (padded to 3000)
    int n_mels = 128;       // mel bins
    int actual_frames = 0;  // actual frames (unpadded), for audio encoder
    std::vector<float> data;  // [n_mels * n_frames] row-major
};

// Load WAV file. Supports PCM f32le (format tag 0x0003) and s16le (format tag 0x0001).
bool load_wav(const std::string& path, WavFile& wav, std::string* error = nullptr);

// Resample from 32kHz to 16kHz via FIR low-pass + decimation by 2.
// Uses a precomputed 33-tap Kaiser-windowed sinc filter.
void resample_32k_to_16k(const std::vector<float>& input,
                         std::vector<float>& output);

// Compute Whisper-compatible mel spectrogram (128 bins, 16kHz, n_fft=400, hop=160).
// Input: mono 16kHz float samples.
bool compute_mel_spectrogram(const std::vector<float>& samples_16k,
                             MelSpectrogram& mel,
                             std::string* error = nullptr);

// Convenience: full pipeline from WAV path to mel spectrogram.
inline bool wav_to_mel(const std::string& wav_path, MelSpectrogram& mel,
                       std::string* error = nullptr) {
    WavFile wav;
    if (!load_wav(wav_path, wav, error)) return false;

    std::vector<float> resampled;
    if (wav.sample_rate == 32000) {
        resample_32k_to_16k(wav.samples, resampled);
    } else if (wav.sample_rate == 16000) {
        resampled = std::move(wav.samples);
    } else {
        if (error) *error = "Unsupported sample rate: " + std::to_string(wav.sample_rate);
        return false;
    }

    if (wav.channels > 1) {
        // Convert to mono by averaging
        std::vector<float> mono(resampled.size() / wav.channels);
        for (size_t i = 0; i < mono.size(); ++i) {
            float sum = 0.0f;
            for (int c = 0; c < wav.channels; ++c)
                sum += resampled[i * wav.channels + c];
            mono[i] = sum / static_cast<float>(wav.channels);
        }
        resampled = std::move(mono);
    }

    return compute_mel_spectrogram(resampled, mel, error);
}
