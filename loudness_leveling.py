
import numpy as np
import soundfile as sf


def loudness_level(
    audio,
    sample_rate,
    target_db=-24.0,
    max_gain_db=12.0,
    max_attenuation_db=12.0,
    attack_time=0.05,
    release_time=0.30,
    frame_time=0.020,
    silence_threshold_db=-50.0
):
    """
    Automatic loudness leveling for speech audio.

    Quiet speech  -> increased gain
    Loud speech   -> reduced gain
    Silence       -> protected from excessive amplification

    Parameters:
        audio: Input mono audio array
        sample_rate: Audio sample rate in Hz
        target_db: Target short-term RMS level in dBFS
        max_gain_db: Maximum gain boost
        max_attenuation_db: Maximum gain reduction
        attack_time: Speed of gain reduction for loud audio
        release_time: Speed of gain recovery for quiet audio
        frame_time: Processing frame duration in seconds
        silence_threshold_db: Level below which audio is
                              treated as silence or very quiet

    Returns:
        Level-adjusted audio as a NumPy array
    """

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim != 1:
        raise ValueError("Input audio must be mono.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if audio.size == 0:
        return audio.copy()

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinite values.")

    frame_length = max(1, int(sample_rate * frame_time))

    # Convert gain limits from dB to linear amplitude.
    max_gain = 10 ** (max_gain_db / 20.0)
    min_gain = 10 ** (-max_attenuation_db / 20.0)

    # Attack and release smoothing coefficients.
    attack_coeff = np.exp(
        -1.0 / (sample_rate * attack_time)
    ) if attack_time > 0 else 0.0

    release_coeff = np.exp(
        -1.0 / (sample_rate * release_time)
    ) if release_time > 0 else 0.0

    output = np.zeros_like(audio)
    current_gain = 1.0

    # Process audio frame by frame.
    for start in range(0, len(audio), frame_length):

        end = min(start + frame_length, len(audio))
        frame = audio[start:end]

        # Calculate RMS loudness.
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        level_db = 20.0 * np.log10(rms)

        # Do not boost silence or extremely quiet regions.
        if level_db < silence_threshold_db:
            desired_gain = 1.0
        else:
            gain_db = target_db - level_db

            # Limit the maximum boost and reduction.
            gain_db = np.clip(
                gain_db,
                -max_attenuation_db,
                max_gain_db
            )

            desired_gain = 10 ** (gain_db / 20.0)

            desired_gain = np.clip(
                desired_gain,
                min_gain,
                max_gain
            )

        # Smooth the gain sample by sample to avoid abrupt jumps.
        for i in range(len(frame)):

            if desired_gain < current_gain:
                # Reduce gain relatively quickly for loud sounds.
                coeff = attack_coeff
            else:
                # Increase gain more gradually for quiet sounds.
                coeff = release_coeff

            current_gain = (
                coeff * current_gain
                + (1.0 - coeff) * desired_gain
            )

            output[start + i] = frame[i] * current_gain

    # Prevent digital clipping while preserving the waveform shape.
    peak = np.max(np.abs(output))

    if peak > 0.99:
        output = output * (0.99 / peak)

    return output


def process_file(
    input_file="dominant_enhanced.wav",
    output_file="loudness_leveled.wav"
):
    """Load, process and save a mono WAV file."""

    audio, sample_rate = sf.read(input_file)

    # Convert stereo to mono if necessary.
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    enhanced_audio = loudness_level(
        audio,
        sample_rate
    )

    sf.write(
        output_file,
        enhanced_audio,
        sample_rate,
        subtype="PCM_16"
    )

    print("Loudness leveling completed.")
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")


if __name__ == "__main__":
    process_file()
