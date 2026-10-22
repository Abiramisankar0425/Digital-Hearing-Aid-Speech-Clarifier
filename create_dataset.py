import os
import numpy as np
import librosa

from features import extract_features


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

FRAME_SIZE = int(0.020 * SAMPLE_RATE)   # 20 ms
HOP_SIZE = int(0.010 * SAMPLE_RATE)     # 10 ms

SPEECH_FOLDER = "data/speech"
NOISE_FOLDER = "data/noise"


# SNR values used to create noisy speech
SNR_VALUES = [
    0,
    5,
    10,
    15,
    20
]


# ============================================================
# FUNCTION: NORMALIZE
# ============================================================

def normalize_audio(audio):

    maximum = np.max(
        np.abs(audio)
    )

    if maximum > 0:

        audio = audio / maximum

    return audio


# ============================================================
# FUNCTION: ADD NOISE
# ============================================================

def mix_noise(
    speech,
    noise,
    snr_db
):

    # Repeat noise if necessary

    if len(noise) < len(speech):

        repetitions = int(
            np.ceil(
                len(speech) /
                len(noise)
            )
        )

        noise = np.tile(
            noise,
            repetitions
        )

    # Random starting position

    max_start = (
        len(noise) -
        len(speech)
    )

    if max_start > 0:

        start = np.random.randint(
            0,
            max_start
        )

        noise = noise[
            start:start + len(speech)
        ]

    else:

        noise = noise[:len(speech)]


    # Calculate speech power

    speech_power = np.mean(
        speech ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )


    if noise_power == 0:

        return speech


    # Desired noise power

    desired_noise_power = (
        speech_power /
        (10 ** (snr_db / 10))
    )


    # Scale noise

    noise_scale = np.sqrt(
        desired_noise_power /
        noise_power
    )


    noise = (
        noise *
        noise_scale
    )


    # Mix

    mixed = (
        speech +
        noise
    )


    return mixed


# ============================================================
# LOAD NOISE FILES
# ============================================================

noise_files = []

for filename in os.listdir(
    NOISE_FOLDER
):

    if filename.lower().endswith(".wav"):

        noise_files.append(
            os.path.join(
                NOISE_FOLDER,
                filename
            )
        )


print()
print("Noise files found:")
print("------------------")

for file in noise_files:

    print(
        os.path.basename(file)
    )


# ============================================================
# DATA ARRAYS
# ============================================================

X = []
y = []


# ============================================================
# PROCESS SPEECH FILES
# ============================================================

print()
print("Creating noisy speech dataset...")
print()


for speech_filename in os.listdir(
    SPEECH_FOLDER
):

    if not speech_filename.lower().endswith(".wav"):

        continue


    speech_path = os.path.join(
        SPEECH_FOLDER,
        speech_filename
    )


    speech, fs = librosa.load(
        speech_path,
        sr=SAMPLE_RATE,
        mono=True
    )


    speech = normalize_audio(
        speech
    )


    # --------------------------------------------------------
    # Create versions with different noises
    # --------------------------------------------------------

    for noise_path in noise_files:

        noise, fs = librosa.load(
            noise_path,
            sr=SAMPLE_RATE,
            mono=True
        )


        noise = normalize_audio(
            noise
        )


        # ----------------------------------------------------
        # Different noise levels
        # ----------------------------------------------------

        for snr_db in SNR_VALUES:

            print(
                "Processing:",
                speech_filename,
                "+",
                os.path.basename(noise_path),
                "SNR:",
                snr_db,
                "dB"
            )


            mixed = mix_noise(
                speech,
                noise,
                snr_db
            )


            # ------------------------------------------------
            # Extract frames
            # ------------------------------------------------

            for start in range(
                0,
                len(mixed) - FRAME_SIZE,
                HOP_SIZE
            ):

                mixed_frame = mixed[
                    start:start + FRAME_SIZE
                ]


                clean_frame = speech[
                    start:start + FRAME_SIZE
                ]


                # ------------------------------------------------
                # Determine whether this frame contains speech
                # ------------------------------------------------

                speech_rms = np.sqrt(
                    np.mean(
                        clean_frame ** 2
                    )
                )


                speech_db = (
                    20 *
                    np.log10(
                        speech_rms +
                        1e-12
                    )
                )


                # Speech activity threshold

                if speech_db > -35:

                    label = 1

                else:

                    label = 0


                # ------------------------------------------------
                # Extract features from NOISY frame
                # ------------------------------------------------

                feature_vector = extract_features(
                    mixed_frame,
                    SAMPLE_RATE
                )


                X.append(
                    feature_vector
                )

                y.append(
                    label
                )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(X)

y = np.array(y)


# ============================================================
# SAVE
# ============================================================

np.save(
    "X.npy",
    X
)

np.save(
    "y.npy",
    y
)


# ============================================================
# RESULTS
# ============================================================

print()
print("======================================")
print("NEW DATASET CREATED")
print("======================================")

print()

print(
    "Total samples:",
    len(X)
)

print(
    "Number of features:",
    X.shape[1]
)

print()

print(
    "Speech samples:",
    np.sum(y == 1)
)

print(
    "Noise samples:",
    np.sum(y == 0)
)

print()

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print()
print("Dataset saved.")
