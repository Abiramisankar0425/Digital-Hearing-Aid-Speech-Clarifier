import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


# ==========================================================
# LOAD RESULTS
# ==========================================================

data = np.load(
    "results/enhancement_data.npz"
)

frame_times = data["frame_times"]
speech_probability = data["speech_probability"]

time_gain_db = data["time_gain_db"]

frequencies = data["frequencies"]

input_audio = data["input_audio"]
enhanced_audio = data["enhanced_audio"]

sample_rate = int(
    data["sample_rate"]
)


# ==========================================================
# CREATE TIME AXES
# ==========================================================

input_time = (
    np.arange(len(input_audio))
    / sample_rate
)

enhanced_time = (
    np.arange(len(enhanced_audio))
    / sample_rate
)


# ==========================================================
# CREATE RESULTS FOLDER
# ==========================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ==========================================================
# PRINT NUMERICAL RESULTS
# ==========================================================

input_rms = np.sqrt(
    np.mean(input_audio ** 2)
)

enhanced_rms = np.sqrt(
    np.mean(enhanced_audio ** 2)
)

input_db = (
    20 *
    np.log10(
        input_rms + 1e-12
    )
)

enhanced_db = (
    20 *
    np.log10(
        enhanced_rms + 1e-12
    )
)

rms_change = (
    enhanced_db - input_db
)

speech_percentage = (
    np.mean(
        speech_probability >= 0.5
    ) * 100
)


print()
print("==========================================")
print("ML-ASSISTED DSP SPEECH ENHANCEMENT")
print("==========================================")
print()

print(
    f"Input RMS: {input_rms:.6f}"
)

print(
    f"Enhanced RMS: {enhanced_rms:.6f}"
)

print(
    f"Input level: {input_db:.2f} dB"
)

print(
    f"Enhanced level: {enhanced_db:.2f} dB"
)

print(
    f"Overall RMS change: {rms_change:.2f} dB"
)

print(
    f"Speech detected: {speech_percentage:.2f}%"
)

print(
    f"Minimum gain: {np.min(time_gain_db):.2f} dB"
)

print(
    f"Maximum gain: {np.max(time_gain_db):.2f} dB"
)

print(
    f"Average gain: {np.mean(time_gain_db):.2f} dB"
)


# ==========================================================
# IMAGE 1
# FOUR GRAPHS ON ONE PAGE
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(18, 10)
)

fig.suptitle(
    "ML-Assisted DSP Speech Clarity Enhancement",
    fontsize=20
)


# ==========================================================
# GRAPH 1
# ORIGINAL VS ENHANCED
# ==========================================================

axes[0, 0].plot(
    input_time,
    input_audio,
    label="Original",
    linewidth=1
)

axes[0, 0].plot(
    enhanced_time,
    enhanced_audio,
    label="Enhanced",
    linewidth=1
)

axes[0, 0].set_title(
    "Original vs Enhanced Waveform"
)

axes[0, 0].set_xlabel(
    "Time (seconds)"
)

axes[0, 0].set_ylabel(
    "Amplitude"
)

axes[0, 0].legend()

axes[0, 0].grid(
    True,
    alpha=0.3
)


# ==========================================================
# GRAPH 2
# ML SPEECH DETECTION
# ==========================================================

axes[0, 1].plot(
    frame_times,
    speech_probability,
    label="Speech probability",
    linewidth=1.5
)

axes[0, 1].axhline(
    0.5,
    linestyle="--",
    label="Speech threshold"
)

axes[0, 1].set_title(
    "ML Speech Detection"
)

axes[0, 1].set_xlabel(
    "Time (seconds)"
)

axes[0, 1].set_ylabel(
    "Speech probability"
)

axes[0, 1].set_ylim(
    0,
    1
)

axes[0, 1].legend()

axes[0, 1].grid(
    True,
    alpha=0.3
)


# ==========================================================
# GRAPH 3
# GAIN / SUPPRESSION
# ==========================================================

axes[1, 0].plot(
    frame_times,
    time_gain_db,
    linewidth=1.5
)

axes[1, 0].axhline(
    0,
    linestyle="--",
    label="0 dB"
)

axes[1, 0].set_title(
    "Speech Amplification / Noise Suppression"
)

axes[1, 0].set_xlabel(
    "Time (seconds)"
)

axes[1, 0].set_ylabel(
    "Gain (dB)"
)

axes[1, 0].legend()

axes[1, 0].grid(
    True,
    alpha=0.3
)


# ==========================================================
# GRAPH 4
# ENHANCED SPECTROGRAM
# ==========================================================

enhanced_stft = librosa.stft(
    enhanced_audio,
    n_fft=1024,
    hop_length=256
)

enhanced_db_spectrogram = librosa.amplitude_to_db(
    np.abs(enhanced_stft),
    ref=np.max
)

img = librosa.display.specshow(
    enhanced_db_spectrogram,
    sr=sample_rate,
    hop_length=256,
    x_axis="time",
    y_axis="hz",
    ax=axes[1, 1]
)

axes[1, 1].set_title(
    "Enhanced Speech Spectrogram"
)

fig.colorbar(
    img,
    ax=axes[1, 1],
    format="%+2.0f dB"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)


# ==========================================================
# SAVE IMAGE 1
# ==========================================================

four_graph_file = (
    "results/four_graphs_analysis.png"
)

plt.savefig(
    four_graph_file,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print()
print(
    "Saved:"
)

print(
    four_graph_file
)


# ==========================================================
# IMAGE 2
# ORIGINAL AND ENHANCED SEPARATELY
# SIDE-BY-SIDE
# ==========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 6)
)

fig.suptitle(
    "Original vs Enhanced Speech",
    fontsize=20
)


# ==========================================================
# ORIGINAL
# ==========================================================

axes[0].plot(
    input_time,
    input_audio,
    linewidth=1
)

axes[0].set_title(
    "ORIGINAL AUDIO"
)

axes[0].set_xlabel(
    "Time (seconds)"
)

axes[0].set_ylabel(
    "Amplitude"
)

axes[0].grid(
    True,
    alpha=0.3
)


# ==========================================================
# ENHANCED
# ==========================================================

axes[1].plot(
    enhanced_time,
    enhanced_audio,
    linewidth=1
)

axes[1].set_title(
    "ENHANCED AUDIO"
)

axes[1].set_xlabel(
    "Time (seconds)"
)

axes[1].set_ylabel(
    "Amplitude"
)

axes[1].grid(
    True,
    alpha=0.3
)


# Keep same Y-axis scale
# for a fair comparison

maximum_amplitude = max(
    np.max(np.abs(input_audio)),
    np.max(np.abs(enhanced_audio))
)

axes[0].set_ylim(
    -maximum_amplitude,
    maximum_amplitude
)

axes[1].set_ylim(
    -maximum_amplitude,
    maximum_amplitude
)


plt.tight_layout(
    rect=[0, 0, 1, 0.93]
)


# ==========================================================
# SAVE IMAGE 2
# ==========================================================

side_by_side_file = (
    "results/original_vs_enhanced_side_by_side.png"
)

plt.savefig(
    side_by_side_file,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print()
print(
    "Saved:"
)

print(
    side_by_side_file
)

print()
print(
    "=========================================="
)

print(
    "VISUALIZATION COMPLETED"
)

print(
    "=========================================="
)