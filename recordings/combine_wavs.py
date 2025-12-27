import scipy.io.wavfile as wavfile
import numpy as np

FAR_RECORDING_NAME = '3m.wav'
CLOSE_RECORDING_NAME = '20cm.wav'
NEW_NAME = 'combined.wav'
NEW_NAME_L = 'combined_l.wav'
NEW_NAME_R = 'combined_r.wav'

def main() -> None:
    far_recording_sample_rate, far_recording_raw_data = wavfile.read(FAR_RECORDING_NAME)
    close_recording_sample_rate, close_recording_raw_data = wavfile.read(CLOSE_RECORDING_NAME)

    assert far_recording_sample_rate == close_recording_sample_rate, 'Incompatible sample rates'
    sample_rate = far_recording_sample_rate

    combined_data = np.concatenate([close_recording_raw_data, far_recording_raw_data], axis=0)
    wavfile.write(NEW_NAME, sample_rate, combined_data)
    wavfile.write(NEW_NAME_L, sample_rate, combined_data[:, 0])
    wavfile.write(NEW_NAME_R, sample_rate, combined_data[:, 1])

if __name__ == "__main__":
    main()