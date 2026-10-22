import numpy as np
import librosa
import soundfile as sf
import joblib

from scipy.ndimage import gaussian_filter1d
from features import extract_features


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

INPUT_FILE = "input.wav"
MODEL_FILE = "speech_noise_model.pkl"
OUTPUT_FILE = "dominant_enhanced.wav"

N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 320


# ============================================================
# NOISE SUPPRESSION SETTINGS
# ============================================================

NOISE_SUBTRACTION = 1.45

# Minimum amount retained from noise-dominated frequencies
MIN_NOISE_GAIN = 0.06

# Additional residual noise suppression
RESIDUAL_NOISE_SUPPRESSION = 0.55


# ============================================================
# SPEECH SETTINGS
# ============================================================

MAX_SPEECH_GAIN = 1.30

SPEECH_THRESHOLD = 0.50


# ============================================================
# FREQUENCY RANGE
# ============================================================

LOW_FREQ = 100
HIGH_FREQ = 7500


# ============================================================
# LOAD AUDIO
# ============================================================

audio, fs = librosa.load(
    INPUT_FILE,
    sr=SAMPLE_RATE,
    mono=True
)

print("Audio loaded")
print("Sample rate:", fs)
print(
    "Duration:",
    len(audio) / fs,
    "seconds"
)


# ============================================================
# LOAD ML MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)

print("ML model loaded")


# ============================================================
# STFT
# ============================================================

stft = librosa.stft(
    audio,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH
)

magnitude = np.abs(stft)

phase = np.angle(stft)

print("STFT created")
print(
    "Number of frames:",
    magnitude.shape[1]
)


# ============================================================
# FREQUENCY AXIS
# ============================================================

frequencies = np.linspace(
    0,
    fs / 2,
    magnitude.shape[0]
)


# ============================================================
# STEP 1
# ML SPEECH PROBABILITY
# ============================================================

speech_probabilities = []


for frame_index in range(
    magnitude.shape[1]
):

    start = (
        frame_index * HOP_LENGTH
    )

    end = (
        start + WIN_LENGTH
    )

    frame = audio[start:end]

    if len(frame) < WIN_LENGTH:

        frame = np.pad(
            frame,
            (
                0,
                WIN_LENGTH - len(frame)
            )
        )

    features = extract_features(
        frame,
        fs
    )

    features = np.array(
        features
    ).reshape(1, -1)

    probability = model.predict_proba(
        features
    )[0][1]

    speech_probabilities.append(
        probability
    )


speech_probabilities = np.array(
    speech_probabilities
)


# ============================================================
# STEP 2
# SMOOTH SPEECH PROBABILITY
# ============================================================

speech_probabilities = gaussian_filter1d(
    speech_probabilities,
    sigma=3
)

speech_probabilities = np.clip(
    speech_probabilities,
    0,
    1
)


# ============================================================
# STEP 3
# FIND NOISE FRAMES
# ============================================================

number_of_noise_frames = max(
    10,
    int(
        0.10 *
        len(speech_probabilities)
    )
)


noise_indices = np.argsort(
    speech_probabilities
)[
    :number_of_noise_frames
]


# ============================================================
# NOISE SPECTRUM
# ============================================================

noise_spectrum = np.median(
    magnitude[:, noise_indices],
    axis=1
)


print()
print(
    "Noise estimation completed"
)

print(
    "Noise frames used:",
    number_of_noise_frames
)


# ============================================================
# STEP 4
# SPECTRAL SUBTRACTION
# ============================================================

enhanced_magnitude = np.zeros_like(
    magnitude
)


for frame_index in range(
    magnitude.shape[1]
):

    current_spectrum = (
        magnitude[:, frame_index]
    )

    speech_probability = (
        speech_probabilities[
            frame_index
        ]
    )


    # --------------------------------------------------------
    # Estimated noise
    # --------------------------------------------------------

    estimated_noise = (
        noise_spectrum
        *
        NOISE_SUBTRACTION
    )


    # --------------------------------------------------------
    # Subtract noise
    # --------------------------------------------------------

    cleaned = (
        current_spectrum
        -
        estimated_noise
    )


    cleaned = np.maximum(
        cleaned,
        0
    )


    # --------------------------------------------------------
    # Speech protection
    # --------------------------------------------------------

    # Strong speech gets more protection.
    speech_protection = (
        0.45
        +
        0.55 * speech_probability
    )


    cleaned = (
        speech_protection
        * cleaned
        +
        (
            1
            -
            speech_protection
        )
        *
        current_spectrum
    )


    # --------------------------------------------------------
    # Frequency range
    # --------------------------------------------------------

    frequency_mask = (
        (frequencies >= LOW_FREQ)
        &
        (frequencies <= HIGH_FREQ)
    )


    cleaned[
        ~frequency_mask
    ] *= 0.20


    enhanced_magnitude[
        :, frame_index
    ] = cleaned


# ============================================================
# STEP 5
# RESIDUAL NOISE SUPPRESSION
# ============================================================

# Compare every frequency bin with the
# estimated background spectrum.

for frame_index in range(
    enhanced_magnitude.shape[1]
):

    current = (
        enhanced_magnitude[
            :, frame_index
        ]
    )

    original = (
        magnitude[
            :, frame_index
        ]
    )

    speech_probability = (
        speech_probabilities[
            frame_index
        ]
    )


    # --------------------------------------------------------
    # Estimate how much stronger the current
    # frequency is than the background
    # --------------------------------------------------------

    ratio = (
        current
        /
        (
            noise_spectrum
            + 1e-8
        )
    )


    # --------------------------------------------------------
    # Create soft frequency mask
    # --------------------------------------------------------

    frequency_mask = np.ones_like(
        ratio
    )


    # Clearly above noise
    strong_signal = ratio >= 3.0

    # Similar to background
    weak_signal = ratio < 1.5

    # Between background and speech
    middle_signal = (
        (ratio >= 1.5)
        &
        (ratio < 3.0)
    )


    # Strong frequencies are retained
    frequency_mask[
        strong_signal
    ] = 1.0


    # Weak frequencies are heavily suppressed
    frequency_mask[
        weak_signal
    ] = MIN_NOISE_GAIN


    # Intermediate frequencies
    frequency_mask[
        middle_signal
    ] = (
        0.20
        +
        0.40
        *
        (
            ratio[
                middle_signal
            ]
            - 1.5
        )
        /
        1.5
    )


    # --------------------------------------------------------
    # Protect speech frames
    # --------------------------------------------------------

    speech_protection = (
        0.45
        +
        0.55
        *
        speech_probability
    )


    frequency_mask = (
        frequency_mask
        *
        (
            1
            -
            speech_protection
            *
            0.35
        )
        +
        speech_protection
        *
        0.35
    )


    # --------------------------------------------------------
    # Apply mask
    # --------------------------------------------------------

    enhanced_magnitude[
        :, frame_index
    ] = (
        current
        *
        (
            1
            -
            RESIDUAL_NOISE_SUPPRESSION
        )
        +
        current
        *
        frequency_mask
        *
        RESIDUAL_NOISE_SUPPRESSION
    )


# ============================================================
# STEP 6
# FREQUENCY SMOOTHING
# ============================================================

enhanced_magnitude = gaussian_filter1d(
    enhanced_magnitude,
    sigma=1.0,
    axis=0
)


# ============================================================
# STEP 7
# TEMPORAL SMOOTHING
# ============================================================

enhanced_magnitude = gaussian_filter1d(
    enhanced_magnitude,
    sigma=0.8,
    axis=1
)


# ============================================================
# STEP 8
# SPEECH AMPLIFICATION
# ============================================================

speech_gain = (
    1.0
    +
    0.30
    *
    speech_probabilities
)


speech_gain = gaussian_filter1d(
    speech_gain,
    sigma=3
)


speech_gain = np.clip(
    speech_gain,
    1.0,
    MAX_SPEECH_GAIN
)


enhanced_magnitude *= (
    speech_gain[None, :]
)


# ============================================================
# STEP 9
# RECONSTRUCT AUDIO
# ============================================================

enhanced_stft = (
    enhanced_magnitude
    *
    np.exp(
        1j * phase
    )
)


enhanced_audio = librosa.istft(
    enhanced_stft,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH
)


# ============================================================
# STEP 10
# NORMALIZE
# ============================================================

peak = np.max(
    np.abs(enhanced_audio)
)

if peak > 0.98:

    enhanced_audio = (
        enhanced_audio
        /
        peak
        *
        0.98
    )


# ============================================================
# STEP 11
# SAVE
# ============================================================

sf.write(
    OUTPUT_FILE,
    enhanced_audio,
    fs
)


# ============================================================
# RESULTS
# ============================================================

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


print()
print(
    "==================================="
)

print(
    "ADAPTIVE NOISE SUPPRESSION"
)

print(
    "==================================="
)

print(
    "Average speech probability:",
    np.mean(
        speech_probabilities
    )
)

print(
    "Minimum speech probability:",
    np.min(
        speech_probabilities
    )
)

print(
    "Maximum speech probability:",
    np.max(
        speech_probabilities
    )
)

print(
    "Input RMS:",
    input_rms
)

print(
    "Output RMS:",
    output_rms
)

print()
print(
    "Enhanced file:",
    OUTPUT_FILE
)