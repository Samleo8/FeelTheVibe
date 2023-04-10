import base64
import io
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def soundDataToFloat(data):
    "Converts integer representation back into librosa-friendly floats, given a numpy array SD"
    INT_MAX = 32768
    return (np.array(data) / INT_MAX).astype(np.float32)


def generate_features(implementation_version, draw_graphs, raw_data, axes,
                      sampling_freq, lpc_order, num_lpcc):
    '''
    Generate series of features from raw data

    Features from:
    https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition?scriptVersionId=34958802&cellId=40
    '''
    # Convert raw WAV files in int16 form to float
    raw_data = soundDataToFloat(raw_data)
    sample_rate = sampling_freq

    assert lpc_order > 0, "LPC must have a strictly positive order number"
    assert num_lpcc >= lpc_order + 1, "Number of LPCC coefficients must be larger than LPC filter order + 1"

    # Initialize empty features
    features = None  # (nfeatures, nframes)
    graphs = []

    # TODO: might need to split into frames
    # https://stackoverflow.com/a/66921909
    # TODO: figure out frame length from sampling frequency?
    print("Sample Freq", sampling_freq)
    frame_len = 2048 # using default value from librosa, based on N_FFT
    hop_len = 492 # not sure what this should be
    frames = librosa.util.frame(raw_data, frame_length=frame_len, hop_length=hop_len).T
    windowed_frames = np.hanning(frame_len) * frames

    print("Frames shape: ", frames.shape)
    print("Windowed frames shape: ", windowed_frames.shape)

    # LPC
    lpc = librosa.lpc(windowed_frames, order=lpc_order)
    print("LPC shape: ", lpc.shape)

    # LPCC Calculation

    features = np.vstack((features, lpc)) \
        if features is not None else lpc

    # TODO: Calculate LPCC Coefficients, may need to apply lpc(c) on windows of raw data?
    # TODO: LPCC Coefficients require the error signal, which is not returned by librosa.lpc

    return {
        'features': features.flatten().tolist(),
        'graphs': graphs,
        # if you use FFTs then set the used FFTs here (this helps with memory optimization on MCUs)
        # NOTE: Unsure if this is correct
        'fft_used': [],  # if not use_chroma else stft.tolist(),
        'output_config': {
            # type can be 'flat', 'image' or 'spectrogram'
            # 'type': 'flat',
            'type': 'spectrogram',
            'shape': {
                # shape should be { width, height, channels } for image, { width, height } for spectrogram
                # 'width': len(features),
                'height': features.shape[0],
                'width': features.shape[1]
            }
        }
    }

if __name__ == "__main__":
    raw_data = np.loadtxt("./test_data.txt", dtype=np.int16)

    generate_features(
        implementation_version=1,
        draw_graphs=False,
        raw_data=raw_data,
        axes=[0,1,2],
        sampling_freq=16000,
        num_lpcc=20,
        lpc_order=16
    )