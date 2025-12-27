import argparse
import logging
import numpy as np
import pathlib
import scipy
import scipy.io.wavfile as wavfile
from typing import Tuple

STANDARD_SAMPLE_RATE = 32000

def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example script with logging")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--audio-file",
        type=pathlib.Path,
        help="The audio file to process",
        required=True,
    )

    return parser.parse_args()

def read_audio_file(audio_file: pathlib.Path) -> Tuple[int, np.array]:
    if not audio_file.exists():
        logging.error(f"{audio_file.as_posix()} doesn't exists!")
        raise RuntimeError('Audio file doesn\'t exists')
    if not audio_file.is_file():
        logging.error(f"{audio_file.as_posix()} is not a file!")
        raise RuntimeError('Audio is not a file')

    original_sample_rate, raw_sample = wavfile.read(audio_file)

    if len(raw_sample.shape) == 2:
        channel_count = raw_sample.shape[1]
        assert channel_count in (1, 2), "Unexpected amount of audio channels"
        if channel_count == 2:
            raw_sample = raw_sample[:, 0]
            logging.debug(f"Truncating from {channel_count} to 1, choosing the first channel")
    else:
        assert len(raw_sample.shape) == 1, "Unexpected channel shape"

    assert len(raw_sample.shape) == 1, "Channel truncation failed"

    assert raw_sample.dtype == np.int16, "Unexpected data type"
    WAVFILE_INT16_MAX_RANGE = 32767.0 # Taken from wavfile.py
    raw_sample = raw_sample.astype(np.float32) / WAVFILE_INT16_MAX_RANGE

    return original_sample_rate, raw_sample

def convert_to_rate(original_sample_rate: int, raw_sample: np.array, new_sample_rate: int) -> np.array:
    fake_timestamps = 0, 1 / original_sample_rate
    num_samples = int(len(raw_sample) * new_sample_rate / original_sample_rate)
    resampled_audio, new_timestamps = scipy.signal.resample(raw_sample, num_samples, t=fake_timestamps)
    return resampled_audio

def main() -> None:
    args = parse_args()
    setup_logging(args.debug)

    original_sample_rate, raw_sample = read_audio_file(args.audio_file)
    logging.info(f"Original sampling frequency: {original_sample_rate} (samples/sec)")
    
    new_sample = convert_to_rate(original_sample_rate, raw_sample, STANDARD_SAMPLE_RATE)
    logging.debug(f'Resampled audio to: {STANDARD_SAMPLE_RATE} (samples/sec)')

if __name__ == "__main__":
    main()