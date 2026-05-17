#include "AudioPreprocessor.hpp"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Pre-computed slaney mel filterbank (matches Python WhisperFeatureExtractor)
#include "mel_fb_data.inc"

namespace {

// ---- WAV header parsing ----

struct WavHeader {
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint32_t byte_rate = 0;
    uint16_t block_align = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_size = 0;
};

bool read_wav_header(std::ifstream& in, WavHeader& hdr) {
    char riff[5] = {};
    in.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0) return false;

    uint32_t file_size;
    in.read(reinterpret_cast<char*>(&file_size), 4);

    char wave[5] = {};
    in.read(wave, 4);
    if (std::strncmp(wave, "WAVE", 4) != 0) return false;

    // Scan chunks
    while (in.good()) {
        char chunk_id[5] = {};
        in.read(chunk_id, 4);
        uint32_t chunk_size = 0;
        in.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            in.read(reinterpret_cast<char*>(&hdr.audio_format), 2);
            in.read(reinterpret_cast<char*>(&hdr.num_channels), 2);
            in.read(reinterpret_cast<char*>(&hdr.sample_rate), 4);
            in.read(reinterpret_cast<char*>(&hdr.byte_rate), 4);
            in.read(reinterpret_cast<char*>(&hdr.block_align), 2);
            in.read(reinterpret_cast<char*>(&hdr.bits_per_sample), 2);
            // Skip extra fmt bytes if any
            if (chunk_size > 16) {
                in.ignore(chunk_size - 16);
            }
        } else if (std::strncmp(chunk_id, "data", 4) == 0) {
            hdr.data_size = chunk_size;
            return true;  // Found data chunk, done scanning
        } else {
            in.ignore(chunk_size);
        }
    }
    return false;
}

// ---- FIR low-pass filter for 32k -> 16k decimation ----

std::vector<float> build_decimation_filter() {
    // Kaiser windowed sinc, cutoff = 0.9 * Nyquist/2 = 0.9 * 8000/32000 = 0.225
    // (normalized freq), 33 taps
    const int ntaps = 33;
    const double cutoff = 0.225;  // normalized to Nyquist (16kHz for 32k signal)
    const double beta = 5.0;      // Kaiser beta

    std::vector<float> h(ntaps);
    int mid = ntaps / 2;

    // Kaiser window: I0(beta * sqrt(1 - (2n/N)^2)) / I0(beta)
    auto I0 = [](double x) {
        double sum = 1.0, term = 1.0;
        for (int k = 1; k < 50; ++k) {
            term *= (x * x) / (4.0 * k * k);
            sum += term;
            if (term < 1e-12) break;
        }
        return sum;
    };
    double I0_beta = I0(beta);

    for (int i = 0; i < ntaps; ++i) {
        int n = i - mid;
        double x = (n == 0) ? (2.0 * cutoff) : (std::sin(2.0 * M_PI * cutoff * n) / (M_PI * n));
        double w = I0(beta * std::sqrt(1.0 - std::pow(2.0 * n / ntaps, 2))) / I0_beta;
        h[i] = static_cast<float>(x * w);
    }

    // Normalize (unity gain at DC)
    float sum = 0.0f;
    for (float v : h) sum += v;
    for (float& v : h) v /= sum;

    return h;
}

// ---- Hann window ----

// NOTE: hann_window_400 is no longer used; Hann is computed inline.
// Keeping for reference but updated to periodic formula.

// ---- Mel filterbank ----

// Convert Hz to mel scale
inline float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

inline float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<float> build_mel_filterbank(int n_mels, int n_fft, int sample_rate) {
    // n_freq_bins = n_fft/2 + 1 = 201 for n_fft=400
    int n_freq = n_fft / 2 + 1;

    float mel_low = hz_to_mel(0.0f);
    float mel_high = hz_to_mel(static_cast<float>(sample_rate) / 2.0f);

    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        float mel = mel_low + (mel_high - mel_low) * i / (n_mels + 1);
        mel_points[i] = static_cast<float>(std::floor(0.5f + mel_to_hz(mel) * n_fft / sample_rate));
    }

    std::vector<float> filters(n_mels * n_freq, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        for (int f = 0; f < n_freq; ++f) {
            float f_start = mel_points[m];
            float f_center = mel_points[m + 1];
            float f_end = mel_points[m + 2];

            if (f >= f_start && f <= f_center && f_center > f_start) {
                filters[m * n_freq + f] = (f - f_start) / (f_center - f_start);
            } else if (f >= f_center && f <= f_end && f_end > f_center) {
                filters[m * n_freq + f] = (f_end - f) / (f_end - f_center);
            }
        }
    }
    return filters;
}

} // anonymous namespace

// ============================================================
// load_wav
// ============================================================

bool load_wav(const std::string& path, WavFile& wav, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (error) *error = "Cannot open: " + path;
        return false;
    }

    WavHeader hdr;
    if (!read_wav_header(in, hdr)) {
        if (error) *error = "Invalid WAV header: " + path;
        return false;
    }

    wav.sample_rate = hdr.sample_rate;
    wav.channels = hdr.num_channels;

    if (hdr.audio_format == 0x0003) {
        // IEEE float (f32le)
        size_t num_samples = hdr.data_size / 4;
        wav.samples.resize(num_samples);
        in.read(reinterpret_cast<char*>(wav.samples.data()), hdr.data_size);
    } else if (hdr.audio_format == 0x0001) {
        // PCM s16le
        size_t num_samples = hdr.data_size / 2;
        std::vector<int16_t> raw(num_samples);
        in.read(reinterpret_cast<char*>(raw.data()), hdr.data_size);
        wav.samples.resize(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            wav.samples[i] = static_cast<float>(raw[i]) / 32768.0f;
        }
    } else {
        if (error)
            *error = "Unsupported WAV format: " + std::to_string(hdr.audio_format);
        return false;
    }

    return true;
}

// ============================================================
// resample_32k_to_16k
// ============================================================

void resample_32k_to_16k(const std::vector<float>& input,
                         std::vector<float>& output) {
    static const std::vector<float> fir = build_decimation_filter();
    const int ntaps = static_cast<int>(fir.size());
    const int mid = ntaps / 2;
    const size_t n_in = input.size();

    // Each output sample at 16kHz = every other sample at 32kHz after filtering
    size_t n_out = n_in / 2;
    output.resize(n_out);

    for (size_t i = 0; i < n_out; ++i) {
        int center = static_cast<int>(i) * 2;  // center position in 32kHz domain
        float sum = 0.0f;
        for (int t = 0; t < ntaps; ++t) {
            int src_idx = center + t - mid;
            if (src_idx >= 0 && src_idx < static_cast<int>(n_in)) {
                sum += input[static_cast<size_t>(src_idx)] * fir[t];
            }
        }
        output[i] = sum;
    }
}

// ============================================================
// compute_mel_spectrogram
// ============================================================

bool compute_mel_spectrogram(const std::vector<float>& samples_16k,
                             MelSpectrogram& mel,
                             std::string* error) {
    const int n_fft = 400;
    const int hop_length = 160;
    const int n_mels = 128;
    const int n_freq = n_fft / 2 + 1;  // 201
    const size_t n_samples = samples_16k.size();

    // Pad audio to 30 seconds (480000 samples) to match WhisperFeatureExtractor
    const int target_samples = 480000;
    std::vector<float> padded_samples(target_samples, 0.0f);
    size_t copy_len = std::min(n_samples, (size_t)target_samples);
    std::copy(samples_16k.begin(), samples_16k.begin() + copy_len, padded_samples.begin());

    // Number of frames with center=True: (n_padded) / hop + 1 = 3001, trim to 3000
    int n_frames_raw = (target_samples) / hop_length + 1;  // 3001
    int n_frames = std::min(n_frames_raw, 3000);  // 3000
    // Actual frames: matches Python's feature_attention_mask.sum() = n_samples / hop
    int actual_frames = static_cast<int>(n_samples) / hop_length;
    if (actual_frames > n_frames) actual_frames = n_frames;

    // Hann window (PyTorch default: periodic, cos(2*pi*n/N))
    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; ++i)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / n_fft));

    // Mel filterbank: kMelFbData from embedded inc file (201 freqs x 128 mels)

    // Compute mel spectrogram
    std::vector<float> power_spec(n_freq);
    std::vector<float> mel_full(static_cast<size_t>(n_mels) * n_frames);
    // Reflect padding: mirrors PyTorch F.pad mode='reflect'
    // Left:  idx < 0        → signal[-idx]     (idx=-200→200, idx=-1→1)
    // Valid: 0 <= idx < N   → signal[idx]
    // Right: idx >= N       → signal[2N-idx-2]  (idx=N→N-2, idx=N+1→N-3,...)
    auto reflect_idx = [&](int idx) -> int {
        if (idx < 0) return -idx;
        if (idx >= target_samples) {
            int over = idx - target_samples;
            return target_samples - over - 2;
        }
        return idx;
    };

    for (int frm = 0; frm < n_frames; ++frm) {
        power_spec.assign(n_freq, 0.0f);
        // With center=True, frame center is at frm * hop
        int center = frm * hop_length;
        int start = center - n_fft / 2;

        for (int k = 0; k < n_freq; ++k) {
            float real_sum = 0.0f;
            float imag_sum = 0.0f;
            float angle_step = -2.0f * static_cast<float>(M_PI) * k / n_fft;
            for (int n = 0; n < n_fft; ++n) {
                int idx = reflect_idx(start + n);
                float sample = padded_samples[idx] * hann[n];
                float angle = angle_step * n;
                real_sum += sample * std::cos(angle);
                imag_sum += sample * std::sin(angle);
            }
            power_spec[k] = real_sum * real_sum + imag_sum * imag_sum;
        }

        for (int m = 0; m < n_mels; ++m) {
            float mel_energy = 0.0f;
            // kMelFbData is (201, 128) = (freq, mel), indexed [f * kMelFbMels + m]
            for (int f = 0; f < n_freq; ++f)
                mel_energy += power_spec[f] * kMelFbData[f * kMelFbMels + m];
            mel_energy = std::max(mel_energy, 1e-10f);
            mel_full[static_cast<size_t>(frm) * n_mels + m] = std::log10(mel_energy);
        }
    }

    // Whisper normalization: clamp to [max-8, max], then scale
    {
        float mel_max = -1e30f;
        for (size_t i = 0; i < mel_full.size(); ++i)
            mel_max = std::max(mel_max, mel_full[i]);
        float mel_min = mel_max - 8.0f;
        for (size_t i = 0; i < mel_full.size(); ++i) {
            mel_full[i] = std::max(mel_full[i], mel_min);
            mel_full[i] = (mel_full[i] + 4.0f) / 4.0f;
        }
    }

    mel.n_frames = n_frames;
    mel.n_mels = n_mels;
    mel.actual_frames = actual_frames;
    mel.data = std::move(mel_full);

    return true;
}
