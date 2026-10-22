import numpy as np
from scipy.signal import butter, sosfilt


def bandpass_filter(
    audio,
    fs,
    lowcut=300,
    highcut=6000,
    order=6
):
    """
    Apply a band-pass filter from
    lowcut to highcut.
    """

    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered_audio = sosfilt(
        sos,
        audio
    )

    return filtered_audio