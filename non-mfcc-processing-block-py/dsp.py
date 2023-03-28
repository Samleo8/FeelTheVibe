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
                      sampling_freq, use_chroma, use_zcr, use_rms):
    '''
    Generate series of features from raw data

    Features from:
    https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition?scriptVersionId=34958802&cellId=40
    '''
    # Convert raw WAV files in int16 form to float
    raw_data = soundDataToFloat(raw_data)
    sample_rate = sampling_freq

    assert use_chroma or use_zcr or use_rms, "At least one feature must be selected"

    # Initialize empty features
    features = np.empty(0)
    graphs = []

    # Chroma STFT
    # https://librosa.org/doc/main/generated/librosa.feature.chroma_stft.html
    if use_chroma:
        stft = np.abs(librosa.stft(raw_data))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                         axis=0)
        
        chroma_std_dev = np.std(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                         axis=0)

        features = np.hstack((features, chroma, chroma_std_dev))

        print("Chroma:", chroma.shape)

        if draw_graphs:
            graphs.append({
                'name': 'Chroma',
                'X': {
                    axes[0]: np.arange(0, len(chroma)).tolist()
                },
                'y': chroma.tolist(),
            })

    # Zero Crossing Rate
    # https://librosa.org/doc/main/generated/librosa.feature.zero_crossing_rate.html
    if use_zcr:
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=raw_data).T, axis=0)
        features = np.hstack((features, zcr))

        print("ZCR:", zcr.shape)

        if draw_graphs:
            graphs.append({
                'name': 'Zero-Crossing-Rate',
                'X': {
                    axes[0]: np.arange(0, len(zcr)).tolist()
                },
                'y': zcr.tolist(),
            })

    # Root Mean Square
    # https://librosa.org/doc/main/generated/librosa.feature.rms.html
    if use_rms:
        rms = np.mean(librosa.feature.rms(y=raw_data).T, axis=0)
        features = np.hstack((features, rms))

        print("RMS:", rms.shape)

        if draw_graphs:
            graphs.append({
                'name': 'Root-Mean-Square',
                'X': {
                    axes[0]: np.arange(0, len(rms)).tolist()
                },
                'y': rms.tolist(),
            })

    print("Features:", features.shape)

    return {
        'features': features,
        'graphs': graphs,
        # if you use FFTs then set the used FFTs here (this helps with memory optimization on MCUs)
        # NOTE: Unsure if this is correct
        'fft_used': [] if not use_chroma else [stft.tolist()],
        'output_config': {
            # type can be 'flat', 'image' or 'spectrogram'
            'type': 'flat',
            'shape': {
                # shape should be { width, height, channels } for image, { width, height } for spectrogram
                'width': len(features),
            }
        }
    }
