import os
import numpy as np
import librosa
import soundfile as sf
import joblib

from scipy.ndimage import gaussian_filter1d
from features import extract_features


# ==========================================================
# SETTINGS
# ==========================================================

SAMPLE_RATE = 16000

INPUT_FILE = "input.wav"
OUTPUT_FILE = "enhanced.wav"

MODEL_FILE = "speech_noise_model.pkl"

RESULTS_FOLDER = "results"


# ==========================================================
# STFT SETTINGS
# ==========================================================

N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 320


# ==========================================================
# ENVIRONMENTAL NOISE SUPPRESSION
# ==========================================================

# Moderate subtraction.
# Do NOT make this extremely high because it can damage speech.
NOISE_SUBTRACTION = 1.45

# Minimum spectral gain.
# 0.08 = maximum suppression of about -22 dB.
MIN_NOISE_GAIN = 0.08


# ==========================================================
# SPEECH AMPLIFICATION
# ==========================================================

# Keep this moderate.
# Large amplification also amplifies remaining background speech.
MAX_SPEECH_GAIN = 1.35


# ==========================================================
# ML SETTINGS
# ==========================================================

# Frames below this are candidate noise-only frames.
NOISE_PROBABILITY_THRESHOLD = 0.30

# Frames above this are considered strong speech.
SPEECH_PROBABILITY_THRESHOLD = 0.50


# ==========================================================
# FREQUENCY SETTINGS
# ==========================================================

# Remove very low-frequency rumble and wind.
LOW_FREQ = 180

# Keep speech frequencies below this.
HIGH_FREQ = 7500


# ==========================================================
# EXTRA PROCESSING SETTINGS
# ==========================================================

# Controls how strongly speech probability protects a frequency.
#
# Lower value:
#     stronger noise suppression
#
# Higher value:
#     more speech protection
#
SPEECH_PROTECTION = 0.65


# Additional suppression in regions where ML strongly
# believes there is no speech.
NOISE_ONLY_SUPPRESSION = 0.45


# Smooth gain changes so the voice does not become
# choppy or robotic.
TIME_SMOOTHING_SIGMA = 2.5


# Frequency smoothing prevents isolated frequency bins
# from sounding metallic.
FREQUENCY_SMOOTHING_SIGMA = 1.2


# ==========================================================
# FUTURE TARGET-SPEAKER SETTINGS
# ==========================================================

# IMPORTANT:
#
# This is NOT yet a true target-speaker separator.
#
# The current Random Forest knows:
#
#     speech
#     vs
#     noise
#
# It does NOT know:
#
#     your voice
#     vs
#     another person's voice
#
# Therefore this switch remains False.
#
ENABLE_TARGET_SPEAKER_SUPPRESSION = False


# ==========================================================
# CREATE RESULTS FOLDER
# ==========================================================

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)


# ==========================================================
# LOAD ML MODEL
# ==========================================================

print()
print("======================================")
print("LOADING ML MODEL")
print("======================================")

model = joblib.load(
    MODEL_FILE
)

print("ML model loaded successfully.")


# ==========================================================
# LOAD INPUT AUDIO
# ==========================================================

print()
print("======================================")
print("LOADING INPUT AUDIO")
print("======================================")

audio, fs = librosa.load(
    INPUT_FILE,
    sr=SAMPLE_RATE,
    mono=True
)

audio = audio.astype(
    np.float32
)

# Remove DC offset
audio = audio - np.mean(audio)

print("Sampling frequency:", fs)

print(
    "Duration:",
    round(
        len(audio) / fs,
        2
    ),
    "seconds"
)


# ==========================================================
# INPUT STFT
# ==========================================================

print()
print("======================================")
print("CALCULATING STFT")
print("======================================")

audio_stft = librosa.stft(
    audio,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window="hann",
    center=True
)

audio_magnitude = np.abs(
    audio_stft
)

audio_power = (
    audio_magnitude ** 2
)

frequencies = librosa.fft_frequencies(
    sr=SAMPLE_RATE,
    n_fft=N_FFT
)

number_of_frames = (
    audio_stft.shape[1]
)

print(
    "Number of STFT frames:",
    number_of_frames
)


# ==========================================================
# ML SPEECH DETECTION
# ==========================================================

print()
print("======================================")
print("RUNNING ML SPEECH DETECTION")
print("======================================")

speech_probabilities = []


for frame_index in range(
    number_of_frames
):

    # ------------------------------------------------------
    # Find the corresponding 20 ms analysis frame
    # ------------------------------------------------------

    center_sample = (
        frame_index
        * HOP_LENGTH
    )

    start = (
        center_sample
        - WIN_LENGTH // 2
    )

    end = (
        start
        + WIN_LENGTH
    )


    # ------------------------------------------------------
    # Create zero-padded frame
    # ------------------------------------------------------

    frame = np.zeros(
        WIN_LENGTH,
        dtype=np.float32
    )


    source_start = max(
        0,
        start
    )

    source_end = min(
        len(audio),
        end
    )


    if source_end > source_start:

        destination_start = (
            source_start
            - start
        )

        destination_end = (
            destination_start
            +
            (
                source_end
                -
                source_start
            )
        )

        frame[
            destination_start:
            destination_end
        ] = audio[
            source_start:
            source_end
        ]


    # ------------------------------------------------------
    # Extract ML features
    # ------------------------------------------------------

    features = extract_features(
        frame,
        SAMPLE_RATE
    )

    features = np.array(
        features
    ).reshape(
        1,
        -1
    )


    # ------------------------------------------------------
    # Predict speech probability
    # ------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probability = model.predict_proba(
            features
        )[0, 1]

    else:

        prediction = model.predict(
            features
        )[0]

        probability = float(
            prediction
        )


    speech_probabilities.append(
        probability
    )


speech_probabilities = np.array(
    speech_probabilities
)


# ==========================================================
# SMOOTH ML PROBABILITY
# ==========================================================

speech_probabilities = gaussian_filter1d(
    speech_probabilities,
    sigma=2.0
)

speech_probabilities = np.clip(
    speech_probabilities,
    0,
    1
)


print(
    "Speech detection completed."
)

print(
    "Average speech probability:",
    round(
        np.mean(speech_probabilities),
        3
    )
)


# ==========================================================
# FIND NOISE-ONLY FRAMES
# ==========================================================

print()
print("======================================")
print("FINDING NOISE-ONLY REGIONS")
print("======================================")


noise_frame_indices = (
    speech_probabilities
    <
    NOISE_PROBABILITY_THRESHOLD
)


number_of_noise_frames = np.sum(
    noise_frame_indices
)


print(
    "Noise candidate frames:",
    number_of_noise_frames,
    "/",
    number_of_frames
)


# ==========================================================
# BUILD ADAPTIVE NOISE PROFILE
# ==========================================================

print()
print("======================================")
print("BUILDING NOISE PROFILE")
print("======================================")


if number_of_noise_frames >= 5:

    print(
        "Using detected noise-only regions."
    )

    adaptive_noise_power = np.median(
        audio_power[
            :,
            noise_frame_indices
        ],
        axis=1
    )

else:

    print(
        "Not enough noise-only frames."
    )

    print(
        "Using conservative low-percentile estimate."
    )

    adaptive_noise_power = np.percentile(
        audio_power,
        15,
        axis=1
    )


# Prevent zero values
adaptive_noise_power = np.maximum(
    adaptive_noise_power,
    1e-12
)


# ==========================================================
# SPECTRAL NOISE ESTIMATION
# ==========================================================

print()
print("======================================")
print("CALCULATING SPECTRAL MASK")
print("======================================")


estimated_speech_power = (
    audio_power
    -
    NOISE_SUBTRACTION
    *
    adaptive_noise_power[:, None]
)


# Do not allow negative power
estimated_speech_power = np.maximum(
    estimated_speech_power,
    0
)


# ==========================================================
# BASE SPECTRAL MASK
# ==========================================================

spectral_mask = (
    estimated_speech_power
    /
    (
        audio_power
        +
        1e-12
    )
)


# Square root gives softer amplitude suppression
spectral_mask = np.sqrt(
    spectral_mask
)


spectral_mask = np.clip(
    spectral_mask,
    0,
    1
)


# ==========================================================
# SPEECH PROTECTION
# ==========================================================

print()
print("Protecting target speech...")


speech_weight = (
    speech_probabilities ** 0.8
)


# ----------------------------------------------------------
# IMPORTANT CHANGE
#
# The old code moved the mask almost completely to 1
# whenever speech probability was high.
#
# That protects speech, but it also protects another
# person's speech.
#
# Here we use a controlled protection factor.
# ----------------------------------------------------------

protection_strength = (
    SPEECH_PROTECTION
    *
    speech_weight
)


protected_mask = (
    spectral_mask
    +
    protection_strength[None, :]
    *
    (
        1.0
        -
        spectral_mask
    )
)


protected_mask = np.clip(
    protected_mask,
    0,
    1
)


# ==========================================================
# NOISE-ONLY SUPPRESSION
# ==========================================================

noise_weight = (
    1.0
    -
    speech_probabilities
)


noise_suppression_factor = (
    1.0
    -
    NOISE_ONLY_SUPPRESSION
    *
    noise_weight
)


final_mask = (
    protected_mask
    *
    noise_suppression_factor[None, :]
)


# ==========================================================
# MINIMUM GAIN
# ==========================================================

final_mask = np.maximum(
    final_mask,
    MIN_NOISE_GAIN
)


# ==========================================================
# FREQUENCY WEIGHTING
# ==========================================================

print()
print("Applying frequency-dependent noise reduction...")


frequency_weight = np.ones(
    len(frequencies)
)


# ----------------------------------------------------------
# VERY LOW FREQUENCIES
#
# Wind, handling noise and low-frequency rumble often
# appear strongly here.
# ----------------------------------------------------------

low_frequency_mask = (
    frequencies < LOW_FREQ
)

frequency_weight[
    low_frequency_mask
] = 0.20


# ----------------------------------------------------------
# NORMAL SPEECH REGION
#
# 180 Hz - 7500 Hz remains mostly protected.
# ----------------------------------------------------------


# ----------------------------------------------------------
# VERY HIGH FREQUENCIES
#
# Reduce excessive hiss while keeping useful consonants.
# ----------------------------------------------------------

high_frequency_mask = (
    frequencies > HIGH_FREQ
)

frequency_weight[
    high_frequency_mask
] = 0.55


final_mask *= (
    frequency_weight[:, None]
)


# ==========================================================
# FREQUENCY SMOOTHING
# ==========================================================

final_mask = gaussian_filter1d(
    final_mask,
    sigma=FREQUENCY_SMOOTHING_SIGMA,
    axis=0
)


# ==========================================================
# SPEECH AMPLIFICATION
# ==========================================================

print()
print("Applying gentle speech amplification...")


speech_gain = (
    1.0
    +
    speech_weight
    *
    (
        MAX_SPEECH_GAIN
        -
        1.0
    )
)


# ==========================================================
# COMBINE SUPPRESSION + SPEECH GAIN
# ==========================================================

final_gain = (
    final_mask
    *
    speech_gain[None, :]
)


# ==========================================================
# TEMPORAL SMOOTHING
# ==========================================================

final_gain = gaussian_filter1d(
    final_gain,
    sigma=TIME_SMOOTHING_SIGMA,
    axis=1
)


# ==========================================================
# LIMIT FINAL GAIN
# ==========================================================

final_gain = np.clip(
    final_gain,
    MIN_NOISE_GAIN,
    MAX_SPEECH_GAIN
)


# ==========================================================
# FUTURE TARGET-SPEAKER HOOK
# ==========================================================

if ENABLE_TARGET_SPEAKER_SUPPRESSION:

    print()
    print(
        "Target-speaker suppression is enabled."
    )

    print(
        "A dedicated speaker-separation model is required."
    )

    print(
        "Current ML classifier alone cannot perform"
        " target-speaker separation."
    )


# ==========================================================
# APPLY GAIN TO STFT
# ==========================================================

print()
print("Applying enhancement...")


enhanced_stft = (
    audio_stft
    *
    final_gain
)


# ==========================================================
# RECONSTRUCT AUDIO
# ==========================================================

print()
print("Reconstructing enhanced audio...")


enhanced_audio = librosa.istft(
    enhanced_stft,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window="hann",
    center=True,
    length=len(audio)
)


# ==========================================================
# REMOVE DC OFFSET
# ==========================================================

enhanced_audio = (
    enhanced_audio
    -
    np.mean(enhanced_audio)
)


# ==========================================================
# FADE EDGES
# ==========================================================

fade_length = min(
    int(
        0.02
        *
        SAMPLE_RATE
    ),
    len(enhanced_audio) // 2
)


if fade_length > 0:

    fade_in = np.linspace(
        0,
        1,
        fade_length
    )

    fade_out = np.linspace(
        1,
        0,
        fade_length
    )

    enhanced_audio[
        :fade_length
    ] *= fade_in

    enhanced_audio[
        -fade_length:
    ] *= fade_out


# ==========================================================
# SAFE NORMALIZATION
# ==========================================================

peak = np.max(
    np.abs(enhanced_audio)
)


if peak > 0:

    if peak > 0.95:

        enhanced_audio *= (
            0.95
            /
            peak
        )


# ==========================================================
# SAVE AUDIO
# ==========================================================

print()
print("======================================")
print("SAVING ENHANCED AUDIO")
print("======================================")


sf.write(
    OUTPUT_FILE,
    enhanced_audio,
    SAMPLE_RATE
)


# ==========================================================
# AUDIO ANALYSIS
# ==========================================================

input_rms = np.sqrt(
    np.mean(
        audio ** 2
    )
)

output_rms = np.sqrt(
    np.mean(
        enhanced_audio ** 2
    )
)


input_db = (
    20
    *
    np.log10(
        input_rms
        +
        1e-12
    )
)


output_db = (
    20
    *
    np.log10(
        output_rms
        +
        1e-12
    )
)


rms_change = (
    output_db
    -
    input_db
)


gain_db = (
    20
    *
    np.log10(
        final_gain
        +
        1e-12
    )
)


# ==========================================================
# SAVE ANALYSIS DATA
# ==========================================================

np.savez(
    os.path.join(
        RESULTS_FOLDER,
        "enhancement_data.npz"
    ),
    speech_probabilities=speech_probabilities,
    frequencies=frequencies,
    final_gain=final_gain,
    adaptive_noise_power=adaptive_noise_power
)


# ==========================================================
# SAVE REPORT
# ==========================================================

report_file = os.path.join(
    RESULTS_FOLDER,
    "enhancement_report.txt"
)


with open(
    report_file,
    "w"
) as report:

    report.write(
        "ML-ASSISTED SPEECH ENHANCEMENT REPORT\n"
    )

    report.write(
        "=====================================\n\n"
    )

    report.write(
        f"Input file: {INPUT_FILE}\n"
    )

    report.write(
        f"Output file: {OUTPUT_FILE}\n"
    )

    report.write(
        f"Sampling rate: {SAMPLE_RATE} Hz\n"
    )

    report.write(
        f"Duration: {len(audio) / SAMPLE_RATE:.2f} seconds\n\n"
    )

    report.write(
        f"Input RMS: {input_rms:.6f}\n"
    )

    report.write(
        f"Output RMS: {output_rms:.6f}\n"
    )

    report.write(
        f"Input level: {input_db:.2f} dB\n"
    )

    report.write(
        f"Output level: {output_db:.2f} dB\n"
    )

    report.write(
        f"Overall RMS change: {rms_change:.2f} dB\n\n"
    )

    report.write(
        f"Average speech probability: "
        f"{np.mean(speech_probabilities):.3f}\n"
    )

    report.write(
        f"Maximum speech probability: "
        f"{np.max(speech_probabilities):.3f}\n"
    )

    report.write(
        f"Minimum speech probability: "
        f"{np.min(speech_probabilities):.3f}\n"
    )

    report.write(
        f"Noise candidate frames: "
        f"{number_of_noise_frames}\n"
    )

    report.write(
        f"Minimum gain: "
        f"{np.min(gain_db):.2f} dB\n"
    )

    report.write(
        f"Maximum gain: "
        f"{np.max(gain_db):.2f} dB\n"
    )

    report.write(
        f"Average gain: "
        f"{np.mean(gain_db):.2f} dB\n"
    )


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print()
print("======================================")
print("ENHANCEMENT COMPLETE")
print("======================================")

print()
print("Input file:")
print(INPUT_FILE)

print()
print("Enhanced file:")
print(OUTPUT_FILE)

print()
print("Input RMS:", round(input_rms, 6))
print("Output RMS:", round(output_rms, 6))

print(
    "Input level:",
    round(input_db, 2),
    "dB"
)

print(
    "Output level:",
    round(output_db, 2),
    "dB"
)

print(
    "Overall RMS change:",
    round(rms_change, 2),
    "dB"
)

print()
print(
    "Average speech probability:",
    round(
        np.mean(speech_probabilities),
        3
    )
)

print(
    "Noise candidate frames:",
    number_of_noise_frames
)

print()
print(
    "Minimum gain:",
    round(
        np.min(gain_db),
        2
    ),
    "dB"
)

print(
    "Maximum gain:",
    round(
        np.max(gain_db),
        2
    ),
    "dB"
)

print(
    "Average gain:",
    round(
        np.mean(gain_db),
        2
    ),
    "dB"
)

print()
print(
    "Saved:",
    OUTPUT_FILE
)

print(
    "Saved:",
    report_file
)

print()
print("======================================")