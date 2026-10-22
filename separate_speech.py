import os
import numpy as np
import soundfile as sf
import torch

from speechbrain.inference.separation import SepformerSeparation


# ==================================================
# SETTINGS
# ==================================================

INPUT_FILE = "input.wav"

OUTPUT_FOLDER = "separated"

SOURCE1_FILE = os.path.join(
    OUTPUT_FOLDER,
    "source1.wav"
)

SOURCE2_FILE = os.path.join(
    OUTPUT_FOLDER,
    "source2.wav"
)

TARGET_SAMPLE_RATE = 8000


# ==================================================
# CREATE OUTPUT FOLDER
# ==================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==================================================
# LOAD INPUT AUDIO
# ==================================================

print("\nLoading input audio...")

audio, sample_rate = sf.read(
    INPUT_FILE
)

print("Original sample rate:", sample_rate)

print(
    "Audio duration:",
    len(audio) / sample_rate,
    "seconds"
)


# ==================================================
# CONVERT STEREO TO MONO
# ==================================================

if audio.ndim > 1:

    print("Stereo audio detected.")

    audio = np.mean(
        audio,
        axis=1
    )

    print("Converted to mono.")


# ==================================================
# CONVERT TO FLOAT32
# ==================================================

audio = audio.astype(
    np.float32
)


# ==================================================
# RESAMPLE IF NECESSARY
# ==================================================

if sample_rate != TARGET_SAMPLE_RATE:

    print(
        "\nResampling:",
        sample_rate,
        "Hz ->",
        TARGET_SAMPLE_RATE,
        "Hz"
    )

    import librosa

    audio = librosa.resample(
        audio,
        orig_sr=sample_rate,
        target_sr=TARGET_SAMPLE_RATE
    )

    sample_rate = TARGET_SAMPLE_RATE


# ==================================================
# SAVE TEMPORARY WAV
# ==================================================

TEMP_FILE = os.path.join(
    OUTPUT_FOLDER,
    "input_8k.wav"
)

sf.write(
    TEMP_FILE,
    audio,
    sample_rate
)

print(
    "\nTemporary file created:",
    TEMP_FILE
)


# ==================================================
# LOAD PRETRAINED SEPFORMER
# ==================================================

print("\nLoading pretrained SepFormer model...")

from speechbrain.utils.fetching import LocalStrategy

separator = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix",
    local_strategy=LocalStrategy.COPY
)
print("SepFormer model loaded successfully.")


# ==================================================
# SEPARATE SPEAKERS
# ==================================================

print("\nSeparating speech sources...")
print("Please wait...")

estimated_sources = separator.separate_file(
    path=TEMP_FILE
)

print("\nSeparation completed.")


# ==================================================
# CHECK OUTPUT
# ==================================================

print(
    "Separated tensor shape:",
    estimated_sources.shape
)


# ==================================================
# EXTRACT SOURCE 1 AND SOURCE 2
# ==================================================

source1 = (
    estimated_sources[:, :, 0]
    .detach()
    .cpu()
    .numpy()
    .squeeze()
)

source2 = (
    estimated_sources[:, :, 1]
    .detach()
    .cpu()
    .numpy()
    .squeeze()
)


# ==================================================
# NORMALIZE EACH SOURCE
# ==================================================

def normalize(audio):

    maximum = np.max(
        np.abs(audio)
    )

    if maximum > 0:

        audio = audio / maximum

    return audio


source1 = normalize(source1)

source2 = normalize(source2)


# ==================================================
# SAVE SOURCES
# ==================================================

sf.write(
    SOURCE1_FILE,
    source1,
    TARGET_SAMPLE_RATE
)

sf.write(
    SOURCE2_FILE,
    source2,
    TARGET_SAMPLE_RATE
)


# ==================================================
# COMPLETE
# ==================================================

print("\n========================================")
print("SPEECH SEPARATION COMPLETE")
print("========================================")

print("\nCreated files:")

print(
    "1.",
    SOURCE1_FILE
)

print(
    "2.",
    SOURCE2_FILE
)

print("\nNow listen to both files.")

print(
    "\nSource 1 should contain one separated speaker."
)

print(
    "Source 2 should contain the other separated speaker."
)