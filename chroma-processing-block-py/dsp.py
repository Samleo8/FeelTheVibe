import base64
import io
import numpy as np
import librosa
import matplotlib.pyplot as plt


def soundDataToFloat(SD):
    "Converts integer representation back into librosa-friendly floats, given a numpy array SD"
    # https://www.kaggle.com/general/213391
    return np.array([np.float32((s >> 2) / (32768.0)) for s in SD])


def generate_features(implementation_version, draw_graphs, raw_data, axes,
                      sampling_freq, scale_axes):
    raw_data = soundDataToFloat(raw_data)

    # features is a 1D array, reshape so we have a matrix
    # Chroma_stft
    # https://librosa.org/doc/main/generated/librosa.feature.chroma_stft.html
    sample_rate = sampling_freq
    stft = np.abs(librosa.stft(raw_data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                     axis=0)

    graphs = []
    if draw_graphs:
        graphs.append({
            'name': 'Chroma',
            'X': np.arange(0, len(chroma)).tolist(),
            'y': chroma.tolist(),
        })

    return {
        'features': chroma,
        'graphs': graphs,
        # if you use FFTs then set the used FFTs here (this helps with memory optimization on MCUs)
        # NOTE: Unsure if this is correct
        'fft_used': [stft.tolist()],
        'output_config': {
            # type can be 'flat', 'image' or 'spectrogram'
            'type': 'flat',
            'shape': {
                # shape should be { width, height, channels } for image, { width, height } for spectrogram
                'width': len(chroma),
            }
        }
    }
