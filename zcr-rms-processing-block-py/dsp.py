import base64
import io
import numpy as np
import librosa
import matplotlib.pyplot as plt


def soundDataToFloat(data):
    "Converts integer representation back into librosa-friendly floats, given a numpy array SD"
    INT_MAX = 32768
    return (np.array(data) / INT_MAX).astype(np.float32)


def generate_features(implementation_version, draw_graphs, raw_data, axes,
                      sampling_freq, scale_axes):
    raw_data = soundDataToFloat(raw_data)

    # Zero Crossing Rate and Root Mean Square Value
    sample_rate = sampling_freq
    zcr = np.mean(librosa.feature.zero_crossing_rate(raw_data).T, axis=0)
    rms = np.mean(librosa.feature.rms(raw_data).T, axis=0)

    zcr_rms = np.concatenate((zcr, rms), axis=0)

    print("ZCR: ", zcr.shape, "RMS: ", rms.shape)

    graphs = []
    if draw_graphs:
        graphs.append({
            'name': 'Zero Crossing Rate and RMS',
            'X': np.arange(0, len(zcr_rms)).tolist(),
            'y': zcr_rms.tolist(),
        })

    return {
        'features': zcr_rms,
        'graphs': graphs,
        # if you use FFTs then set the used FFTs here (this helps with memory optimization on MCUs)
        # NOTE: Unsure if this is correct
        'fft_used': [],
        'output_config': {
            # type can be 'flat', 'image' or 'spectrogram'
            'type': 'flat',
            'shape': {
                # shape should be { width, height, channels } for image, { width, height } for spectrogram
                'width': len(zcr_rms),
            }
        }
    }
