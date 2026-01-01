import argparse
import librosa
import logging
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pyworld as pw
import scipy
import scipy.io.wavfile as wavfile
from typing import Tuple

STANDARD_SAMPLE_RATE = 32000
OUTPUT_DIR = pathlib.Path('outputs')

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
    parser.add_argument('--question',
                        type=str,
                        help='Which question we are doing now',
                        required=True,
                        choices=['a', 'b'])

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

def draw_resampled_plots(title: str, sample_rate: int, samples: np.array) -> None:
    sample_intervale = 1 / sample_rate
    fake_timestamps = [i * sample_intervale for i in range(len(samples))]

    fig, ((audio_plt, spectogram_plt), (rms_plt, melspectogram_plt)) = plt.subplots(2, 2)
    fig.suptitle(title)
    
    audio_plt.plot(fake_timestamps, samples)
    audio_plt.set_title('Audio as a function of time')
    audio_plt.set_ylabel('Amplitude')
    audio_plt.set_xlabel('Time (s)')

    window_size_ms = 20
    hop_size_ms = 10
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    hop_size_samples = int(sample_rate * hop_size_ms / 1000)

    S = librosa.stft(samples, win_length=window_size_samples, hop_length=hop_size_samples)
    S_db = librosa.power_to_db(np.abs(S), ref=np.max)

    spec_img = librosa.display.specshow(S_db, 
                             sr=sample_rate,
                             x_axis='time',
                             y_axis='log',
                             win_length=window_size_samples,
                             hop_length=hop_size_samples,
                             ax=spectogram_plt)
    spectogram_plt.set_title('Spectogram')
    spectogram_plt.set_xlabel("Time (s)")
    spectogram_plt.set_ylabel("Frequency (log(Hz))")
    f0, timestamps = pw.harvest(samples.copy(order='C').astype(np.double), sample_rate)
    spectogram_plt.plot(timestamps, f0, color='cyan', linewidth=2, label='Pitch Contour (F0)')
    spectogram_plt.legend(loc='upper right')
    plt.colorbar(spec_img, ax=spectogram_plt, format='%+2.0f dB')

    mel = librosa.feature.melspectrogram(y=samples,
                                         sr=sample_rate,
                                         win_length=window_size_samples,
                                         hop_length=hop_size_samples)
    mel_db = librosa.power_to_db(np.abs(mel), ref=np.max)

    mel_img = librosa.display.specshow(mel_db, 
                             sr=sample_rate,
                             x_axis='time',
                             y_axis='mel',
                             win_length=window_size_samples,
                             hop_length=hop_size_samples,
                             ax=melspectogram_plt)
    melspectogram_plt.set_title('Mel Spectogram')
    melspectogram_plt.set_xlabel("Time (s)")
    melspectogram_plt.set_ylabel("Frequency (Hz)")
    plt.colorbar(mel_img, ax=melspectogram_plt, format='%+2.0f dB')

    rms = librosa.feature.rms(y=samples, frame_length=window_size_samples, hop_length=hop_size_samples)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    times = librosa.times_like(rms, sr=sample_rate, hop_length=hop_size_samples)
    rms_plt.set_title('RMS Energy')
    rms_plt.plot(times, rms, label='')
    rms_plt.set_ylabel('RMS Energy (Root-Mean-Square)')
    rms_plt.set_xlabel('Time (s)')

    plt.show()

def question_a(samples: np.array) -> None:
    assert STANDARD_SAMPLE_RATE % 2 == 0
    new_sample_rate = STANDARD_SAMPLE_RATE // 2
    my_down_sample = samples[::2]
    scipy_down_sample = convert_to_rate(STANDARD_SAMPLE_RATE, samples, new_sample_rate)

    draw_resampled_plots('Trivial Down Sample', new_sample_rate, my_down_sample)
    draw_resampled_plots('Scipy Down Sample', new_sample_rate, scipy_down_sample)

    
    TRIVIAL_DOWN_SAMPLE_LOCATION = OUTPUT_DIR / pathlib.Path('1.c.trivial_down_sample.wav')
    SCIPY_DOWN_SAMPLE_LOCATION = OUTPUT_DIR / pathlib.Path('1.c.scipy_down_sample.wav')
    wavfile.write(TRIVIAL_DOWN_SAMPLE_LOCATION, new_sample_rate, my_down_sample)
    logging.info(f'Saved trivial down sample to: "{TRIVIAL_DOWN_SAMPLE_LOCATION.as_posix()}"')
    wavfile.write(SCIPY_DOWN_SAMPLE_LOCATION, new_sample_rate, scipy_down_sample)
    logging.info(f'Saved scipy down sample to: "{SCIPY_DOWN_SAMPLE_LOCATION.as_posix()}"')


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)

    original_sample_rate, raw_sample = read_audio_file(args.audio_file)
    logging.info(f"Original sampling frequency: {original_sample_rate} (samples/sec)")
    
    new_sample = convert_to_rate(original_sample_rate, raw_sample, STANDARD_SAMPLE_RATE)
    logging.debug(f'Resampled audio to: {STANDARD_SAMPLE_RATE} (samples/sec)')

    if args.question == 'a':
        logging.info('Answering question \'a\'')
        question_a(new_sample)
    elif args.question == 'b':
        logging.info('Answering question \'b\'')
        pass
    else:
        raise RuntimeError(f'Unknown question ({repr(args.question)})')

if __name__ == "__main__":
    main()