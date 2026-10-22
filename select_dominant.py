import numpy as np
import soundfile as sf
import os


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SOURCE1_FILE = "separated/source1.wav"
SOURCE2_FILE = "separated/source2.wav"

OUTPUT_FOLDER = "results"

DOMINANT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "dominant_speech.wav"
)

BACKGROUND_FILE = os.path.join(
    OUTPUT_FOLDER,
    "background_speaker.wav"
)


# --------------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --------------------------------------------------
# LOAD AUDIO
# --------------------------------------------------

print("\nLoading separated sources...")

source1, fs1 = sf.read(SOURCE1_FILE)
source2, fs2 = sf.read(SOURCE2_FILE)

print("Source 1 sample rate:", fs1)
print("Source 2 sample rate:", fs2)


# --------------------------------------------------
# MAKE BOTH SAME LENGTH
# --------------------------------------------------

minimum_length = min(
    len(source1),
    len(source2)
)

source1 = source1[:minimum_length]
source2 = source2[:minimum_length]


# --------------------------------------------------
# CALCULATE RMS ENERGY
# --------------------------------------------------

def calculate_rms(audio):

    return np.sqrt(
        np.mean(audio ** 2)
    )


rms1 = calculate_rms(source1)
rms2 = calculate_rms(source2)


print("\n====================================")
print("SOURCE ANALYSIS")
print("====================================")

print("Source 1 RMS:", rms1)
print("Source 2 RMS:", rms2)


# --------------------------------------------------
# SELECT DOMINANT SOURCE
# --------------------------------------------------

if rms1 >= rms2:

    dominant = source1
    background = source2

    dominant_number = 1

else:

    dominant = source2
    background = source1

    dominant_number = 2


# --------------------------------------------------
# SAVE
# --------------------------------------------------

sf.write(
    DOMINANT_FILE,
    dominant,
    fs1
)

sf.write(
    BACKGROUND_FILE,
    background,
    fs1
)


print("\nDominant source:", dominant_number)

print("\nCreated:")
print(DOMINANT_FILE)
print(BACKGROUND_FILE)

print("\n====================================")
print("DOMINANT SPEAKER SELECTED")
print("====================================")