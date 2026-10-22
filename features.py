import numpy as np


def calculate_rms(frame):
    return np.sqrt(np.mean(frame ** 2) + 1e-12)


def calculate_zcr(frame):
    return np.mean(
        np.abs(np.diff(np.sign(frame)))
    ) / 2


def calculate_spectral_centroid(frame, fs):

    spectrum = np.abs(
        np.fft.rfft(frame)
    )

    frequencies = np.fft.rfftfreq(
        len(frame),
        1 / fs
    )

    total_magnitude = np.sum(
        spectrum
    )

    if total_magnitude == 0:
        return 0

    return np.sum(
        frequencies * spectrum
    ) / total_magnitude


def calculate_hf_ratio(frame, fs):

    spectrum = (
        np.abs(
            np.fft.rfft(frame)
        ) ** 2
    )

    frequencies = np.fft.rfftfreq(
        len(frame),
        1 / fs
    )

    total_energy = np.sum(
        spectrum
    )

    if total_energy == 0:
        return 0

    hf_energy = np.sum(
        spectrum[
            (frequencies >= 2000)
            &
            (frequencies <= 6000)
        ]
    )

    return hf_energy / total_energy


# ==========================================================
# NEW FEATURE 5
# SPECTRAL BANDWIDTH
# ==========================================================

def calculate_spectral_bandwidth(frame, fs):

    spectrum = np.abs(
        np.fft.rfft(frame)
    )

    frequencies = np.fft.rfftfreq(
        len(frame),
        1 / fs
    )

    total = np.sum(spectrum)

    if total == 0:
        return 0

    centroid = np.sum(
        frequencies * spectrum
    ) / total

    bandwidth = np.sqrt(
        np.sum(
            ((frequencies - centroid) ** 2)
            * spectrum
        )
        / total
    )

    return bandwidth


# ==========================================================
# NEW FEATURE 6
# SPECTRAL ROLLOFF
# ==========================================================

def calculate_spectral_rolloff(
    frame,
    fs,
    roll_percent=0.85
):

    spectrum = np.abs(
        np.fft.rfft(frame)
    )

    frequencies = np.fft.rfftfreq(
        len(frame),
        1 / fs
    )

    total_energy = np.sum(
        spectrum
    )

    if total_energy == 0:
        return 0

    cumulative_energy = np.cumsum(
        spectrum
    )

    threshold = (
        roll_percent
        * total_energy
    )

    index = np.where(
        cumulative_energy >= threshold
    )[0]

    if len(index) == 0:
        return 0

    return frequencies[
        index[0]
    ]


# ==========================================================
# NEW FEATURE 7
# SPECTRAL FLATNESS
# ==========================================================

def calculate_spectral_flatness(
    frame,
    fs
):

    spectrum = np.abs(
        np.fft.rfft(frame)
    ) + 1e-12

    geometric_mean = np.exp(
        np.mean(
            np.log(spectrum)
        )
    )

    arithmetic_mean = np.mean(
        spectrum
    )

    if arithmetic_mean == 0:
        return 0

    return (
        geometric_mean
        / arithmetic_mean
    )


# ==========================================================
# FINAL FEATURE EXTRACTION
# ==========================================================

def extract_features(frame, fs):

    rms = calculate_rms(
        frame
    )

    zcr = calculate_zcr(
        frame
    )

    centroid = calculate_spectral_centroid(
        frame,
        fs
    )

    hf_ratio = calculate_hf_ratio(
        frame,
        fs
    )

    bandwidth = calculate_spectral_bandwidth(
        frame,
        fs
    )

    rolloff = calculate_spectral_rolloff(
        frame,
        fs
    )

    flatness = calculate_spectral_flatness(
        frame,
        fs
    )

    return [
        rms,
        zcr,
        centroid,
        hf_ratio,
        bandwidth,
        rolloff,
        flatness
    ]