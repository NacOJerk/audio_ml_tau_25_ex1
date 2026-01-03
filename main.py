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
assert STANDARD_SAMPLE_RATE % 2 == 0
NEW_SAMPLE_RATE = STANDARD_SAMPLE_RATE // 2
WINDOW_SIZE_MS = 20
HOP_SIZE_MS = 10


OUTPUT_DIR = pathlib.Path('outputs')
ASSETS_DIR = pathlib.Path('assets')
NOISE_PATH = ASSETS_DIR / 'stationary_noise.wav'

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
    parser.add_argument(
        '--use-interpolate-for-spectracl',
        action='store_true',
        help='Use interpolation instead of average in spectral noise removal',
    )
    parser.add_argument(
        '--audio-noise-threshold',
        type=float,
        help='The threshold for the spectral substraction in db',
        default=-17.95,
    )
    parser.add_argument(
        '--target-amplified-rms',
        type=float,
        help='The target rms after amilification',
        default=-8,
    )
    parser.add_argument(
        '--speed-up-factor',
        type=float,
        help='The target rms after amilification',
        default=1.5,
    )
    parser.add_argument('--question',
                        type=str,
                        help='Which question we are doing now',
                        required=True,
                        choices=['a', 'b', 'c', 'd', 'e'])

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

    if raw_sample.dtype == np.int16:
        WAVFILE_INT16_MAX_RANGE = 32767.0 # Taken from wavfile.py
        raw_sample = raw_sample.astype(np.float32) / WAVFILE_INT16_MAX_RANGE

    assert raw_sample.dtype == np.float32, f"Unexpected data type ({repr(raw_sample.dtype)})"
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

    window_size_samples = int(sample_rate * WINDOW_SIZE_MS / 1000)
    hop_size_samples = int(sample_rate * HOP_SIZE_MS / 1000)

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

    # From librosa.feature.melspectrogram documentation
    f_max = float(sample_rate) / 2 
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
    melspectogram_plt.set_title('Mel Spectogram ($F_{max}$ = %.2f)' % (f_max))
    melspectogram_plt.set_xlabel("Time (s)")
    melspectogram_plt.set_ylabel("Frequency (Hz)")
    plt.colorbar(mel_img, ax=melspectogram_plt, format='%+2.0f dB')

    rms = librosa.feature.rms(y=samples, frame_length=window_size_samples, hop_length=hop_size_samples)[0]
    times = librosa.times_like(rms, sr=sample_rate, hop_length=hop_size_samples)
    rms_plt.set_title('RMS Energy')
    rms_plt.plot(times, rms, label='')
    rms_plt.set_ylabel('RMS Energy (Root-Mean-Square)')
    rms_plt.set_xlabel('Time (s)')

    plt.show()

def plot_question_a(my_down_sample: np.array, scipy_down_sample: np.array) -> None:
    draw_resampled_plots('Trivial Down Sample', NEW_SAMPLE_RATE, my_down_sample)
    draw_resampled_plots('Scipy Down Sample', NEW_SAMPLE_RATE, scipy_down_sample)

    TRIVIAL_DOWN_SAMPLE_LOCATION = OUTPUT_DIR / pathlib.Path('1.c.trivial_down_sample.wav')
    SCIPY_DOWN_SAMPLE_LOCATION = OUTPUT_DIR / pathlib.Path('1.c.scipy_down_sample.wav')
    wavfile.write(TRIVIAL_DOWN_SAMPLE_LOCATION, NEW_SAMPLE_RATE, my_down_sample)
    logging.info(f'Saved trivial down sample to: "{TRIVIAL_DOWN_SAMPLE_LOCATION.as_posix()}"')
    wavfile.write(SCIPY_DOWN_SAMPLE_LOCATION, NEW_SAMPLE_RATE, scipy_down_sample)
    logging.info(f'Saved scipy down sample to: "{SCIPY_DOWN_SAMPLE_LOCATION.as_posix()}"')

def load_audio_file_as_standard(path: pathlib.Path) -> np.array:
    logging.info(f'Loading "{path}"')
    original_sample_rate, sample = read_audio_file(path)
    logging.info(f"Original sampling frequency: {original_sample_rate} (samples/sec)")
    if original_sample_rate != STANDARD_SAMPLE_RATE:
        sample = convert_to_rate(original_sample_rate, sample, STANDARD_SAMPLE_RATE)
        logging.debug(f'Resampled audio to: {STANDARD_SAMPLE_RATE} (samples/sec)')
    return sample

def plot_question_b(sample_rate: int, noise: np.array, audio: np.array, noised_audio: np.array) -> None:
    assert len(noise) == len(audio) and \
        len(audio) == len(noised_audio), f'Length mismatch ({len(noise)} vs {len(audio)} vs {len(noised_audio)})'

    sample_intervale = 1 / sample_rate
    fake_timestamps = [i * sample_intervale for i in range(len(noise))]

    fig, (audio_plt, noised_audio_plt) = plt.subplots(2)
    fig.suptitle('Audio vs Noised Audio')
    
    audio_plt.plot(fake_timestamps, audio)
    audio_plt.set_title('Audio as a function of time')
    audio_plt.set_ylabel('Amplitude')
    audio_plt.set_xlabel('Time (s)')

    noised_audio_plt.plot(fake_timestamps, noised_audio)
    noised_audio_plt.set_title('Noised Audio as a function of time')
    noised_audio_plt.set_ylabel('Amplitude')
    noised_audio_plt.set_xlabel('Time (s)')

    plt.show()

    plt.plot(fake_timestamps, noise)
    plt.title('Noise as a function of time')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.show()

    draw_resampled_plots('Noised audio', sample_rate, noised_audio)
    
    NOISY_SAMPLE_LOCATION = OUTPUT_DIR / pathlib.Path('2.noisy_sample.wav')
    wavfile.write(NOISY_SAMPLE_LOCATION, NEW_SAMPLE_RATE, noised_audio)
    logging.info(f'Saved trivial down sample to: "{NOISY_SAMPLE_LOCATION.as_posix()}"')

def create_matching_audio_frames(y: np.ndarray,
                                 frame_length: int,
                                 hop_length: int,
                                 center: bool = True,
                                 pad_mode = "constant") -> np.ndarray:
    if center:
        padding = [(0, 0) for _ in range(y.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        y = np.pad(y, padding, mode=pad_mode)

    return librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

def plot_question_c(sample_rate: int,
                    hop_length: int,
                    audio_noise_threshold: float, 
                    rms: np.ndarray, 
                    cleaned_audio: np.ndarray) -> None:
    rms_db = librosa.power_to_db(rms)
    times = librosa.times_like(rms, sr=sample_rate, hop_length=hop_length)
    threshold_raw = [audio_noise_threshold] * len(times)
    threshold_raw_db = librosa.power_to_db(threshold_raw)
    plt.title('RMS Energy vs threshold')
    plt.plot(times, rms_db, label='RMS', color='blue')
    plt.plot(times, threshold_raw_db, label='threshold', color='red')
    plt.ylabel('RMS Energy (Root-Mean-Square) in dB')
    plt.xlabel('Time (s)')
    plt.legend('upper right')
    plt.show()

    draw_resampled_plots('Cleaned Audio', sample_rate, cleaned_audio)

    CLEANED_AUDIO_LOCATION = OUTPUT_DIR / pathlib.Path('3_cleaned_audio.wav')
    wavfile.write(CLEANED_AUDIO_LOCATION, NEW_SAMPLE_RATE, cleaned_audio)
    logging.info(f'Saved cleaned audio  to: "{CLEANED_AUDIO_LOCATION.as_posix()}"')

def plot_question_d(sample_rate: int,
                    amplified_audio: np.ndarray,
                    rms: np.ndarray,
                    target_rms: float,
                    noise_threshold_rms: float,
                    window_and_hop_size: int):
    draw_resampled_plots('Amplified Audio', NEW_SAMPLE_RATE, amplified_audio)
    CLEANED_AND_AMPLIFIED_AUDIO_LOCATION = OUTPUT_DIR / pathlib.Path('4_cleaned_and_amplified_audio.wav')
    wavfile.write(CLEANED_AND_AMPLIFIED_AUDIO_LOCATION, NEW_SAMPLE_RATE, amplified_audio)
    logging.info(f'Saved cleaned and amplified audio to: "{CLEANED_AND_AMPLIFIED_AUDIO_LOCATION.as_posix()}"')

    times = librosa.times_like(rms, sr=sample_rate, hop_length=window_and_hop_size)
    gain_factor = target_rms / rms
    amplification = np.where((rms > noise_threshold_rms) | (gain_factor < 1), gain_factor, 1)
    plt.plot(times, amplification)
    plt.title('Audio Amplification Factor as a function of time')
    plt.ylabel('Amplification') 
    plt.xlabel('Time (s)')
    plt.show()


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)

    ###################################
    #                                 #
    #       Loading Audio Files       #
    #                                 #
    ###################################
    audio_sample = load_audio_file_as_standard(args.audio_file)
    noise_sample = convert_to_rate(STANDARD_SAMPLE_RATE, load_audio_file_as_standard(NOISE_PATH), NEW_SAMPLE_RATE)

    my_down_sample = audio_sample[::2]
    scipy_down_sample = convert_to_rate(STANDARD_SAMPLE_RATE, audio_sample, NEW_SAMPLE_RATE)

    noise_sample = noise_sample[:len(scipy_down_sample)]
    noised_audio = scipy_down_sample + noise_sample

    ###################################
    #                                 #
    #           Question 3            #
    #                                 #
    ###################################
    window_size_samples = int(NEW_SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
    hop_size_samples = int(NEW_SAMPLE_RATE * HOP_SIZE_MS / 1000)
    audio_noise_threshold = librosa.db_to_power(args.audio_noise_threshold)
    f_noised_audio = librosa.stft(noised_audio, win_length=window_size_samples, hop_length=hop_size_samples)
    rms = librosa.feature.rms(y=noised_audio, frame_length=window_size_samples, hop_length=hop_size_samples)[0]
    frame_index_buffer = []
    noise_buffer = []
    for i in range(len(rms)):
        if rms[i] < audio_noise_threshold:
            noise_buffer.append(f_noised_audio[:, i])
            frame_index_buffer.append(i)

    frame_index_buffer = np.array(frame_index_buffer)
    noise_buffer = np.array(noise_buffer)
    
    f_clean_audio = f_noised_audio.copy()

    if args.use_interpolate_for_spectracl:
        interp_func = scipy.interpolate.interp1d(frame_index_buffer, noise_buffer, 
                axis=0, 
                kind='linear', 
                fill_value="extrapolate")
        for i in range(f_clean_audio.shape[1]):
            if rms[i] < audio_noise_threshold:
                f_clean_audio[:, i] -= f_noised_audio[:, i]
            else:
                f_clean_audio[:, i] -= interp_func(i)
    else:
        noise_average = np.average(noise_buffer, axis=0)
        for i in range(f_clean_audio.shape[1]):
            f_clean_audio[:, i] -= noise_average

    cleaned_audio = librosa.istft(f_clean_audio, win_length=window_size_samples, hop_length=hop_size_samples)

    ###################################
    #                                 #
    #           Question 4            #
    #                                 #
    ###################################
    amplified_audio = cleaned_audio.copy()
    windows_size_agc_ms = 1000
    windows_size_agc_samples = int(NEW_SAMPLE_RATE * windows_size_agc_ms / 1000)
    hop_size_agc_ms = 50
    hop_size_agc_samples = int(NEW_SAMPLE_RATE * hop_size_agc_ms / 1000)
    agc_rms = librosa.feature.rms(y=noised_audio,
                                  frame_length=windows_size_agc_samples, 
                                  hop_length=hop_size_agc_samples, 
                                  center=True)[0]
    target_rms = librosa.db_to_power(args.target_amplified_rms)
    
    for current_frame_idx in range(len(agc_rms)):
        current_rms = agc_rms[current_frame_idx]
        gain_factor = target_rms / current_rms
        if current_rms < audio_noise_threshold and gain_factor > 1:
            continue
        start_idx_in_audio = current_frame_idx * hop_size_agc_samples
        end_idx_in_audio = min(start_idx_in_audio + hop_size_agc_samples, len(amplified_audio))
        amplified_audio[start_idx_in_audio:end_idx_in_audio] *= gain_factor
    amplified_audio = np.tanh(amplified_audio)

    ###################################
    #                                 #
    #           Question 5            #
    #                                 #
    ###################################
    f_orig_audio = librosa.stft(scipy_down_sample, win_length=window_size_samples, hop_length=hop_size_samples)
    vo_coder_interp = scipy.interpolate.interp1d(
                list(range(f_orig_audio.shape[1])),
                f_orig_audio, 
                axis=1, 
                kind='linear')
    max_speed_up_frame = int(np.floor(f_orig_audio.shape[1] / args.speed_up_factor))
    f_speed_up_audio = []
    for i in range(max_speed_up_frame):
        input_in_orig = i * args.speed_up_factor
        f_speed_up_audio.append(vo_coder_interp(input_in_orig))
    f_speed_up_audio = np.array(f_speed_up_audio).T
    speed_up_audio = librosa.istft(f_speed_up_audio, win_length=window_size_samples, hop_length=hop_size_samples)

    logging.info(f"Answering question: '{args.question}'")
    if args.question == 'a':
        plot_question_a(my_down_sample, scipy_down_sample)
    elif args.question == 'b':
        plot_question_b(NEW_SAMPLE_RATE, noise_sample, scipy_down_sample, noised_audio)
    elif args.question == 'c':
        plot_question_c(NEW_SAMPLE_RATE, hop_size_samples, audio_noise_threshold, rms, cleaned_audio)
    elif args.question == 'd':
        plot_question_d(NEW_SAMPLE_RATE, amplified_audio, agc_rms, target_rms, audio_noise_threshold, hop_size_agc_samples)
    elif args.question == 'e':
        draw_resampled_plots(f'Audio speed up by {args.speed_up_factor}', NEW_SAMPLE_RATE, speed_up_audio)
        SPEED_UP_AUDIO_LOCATION = OUTPUT_DIR / pathlib.Path(f'5_speed_up_by_{args.speed_up_factor}_audio.wav')
        wavfile.write(SPEED_UP_AUDIO_LOCATION, NEW_SAMPLE_RATE,speed_up_audio)
        logging.info(f'Saved speed up audio to: "{SPEED_UP_AUDIO_LOCATION.as_posix()}"')
    else:
        raise RuntimeError(f'Unknown question ({repr(args.question)})')

if __name__ == "__main__":
    main()