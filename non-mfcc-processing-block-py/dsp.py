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
    features = None  # (nfeatures, nframes)
    graphs = []

    # Chroma STFT
    # https://librosa.org/doc/main/generated/librosa.feature.chroma_stft.html
    if use_chroma:
        stft = np.abs(librosa.stft(raw_data))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)

        features = np.vstack(
            (features, chroma)) if features is not None else chroma

        if draw_graphs:
            # Create image
            # https://github.com/edgeimpulse/processing-blocks/blob/master/mfcc/dsp.py
            fig, ax = plt.subplots()
            fig.set_size_inches(18.5, 20.5)
            ax.set_axis_off()
            # img = librosa.display.specshow(chroma,
            #                                y_axis='log',
            #                                x_axis='time',
            #                                ax=ax)
            # fig.colorbar(img, ax=ax)
            # ax.label_outer()
            cax = ax.imshow(chroma,
                            interpolation='nearest',
                            cmap=cm.coolwarm,
                            origin='lower')

            buf = io.BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            image = (base64.b64encode(buf.getvalue()).decode('ascii'))
            buf.close()

            graphs.append({
                'name': 'Chroma Spectrogram',
                'image': image,
                'imageMimeType': 'image/svg+xml',
                'type': 'image'
            })

    # Zero Crossing Rate
    # https://librosa.org/doc/main/generated/librosa.feature.zero_crossing_rate.html
    if use_zcr:
        zcr = librosa.feature.zero_crossing_rate(y=raw_data)
        # print("ZCR:", zcr.shape)

        features = np.vstack((features, zcr)) if features is not None else zcr

        if draw_graphs:
            graphs.append({
                'name': 'Zero-Crossing-Rate',
                'X': {
                    axes[0]: np.arange(0, zcr.shape[1]).tolist()
                },
                'y': zcr.flatten().tolist(),
            })

    # Root Mean Square
    # https://librosa.org/doc/main/generated/librosa.feature.rms.html
    if use_rms:
        rms = librosa.feature.rms(y=raw_data)
        # print("RMS:", rms.shape)

        features = np.vstack((features, rms)) if features is not None else rms

        if draw_graphs:
            graphs.append({
                'name': 'Root-Mean-Square',
                'X': {
                    axes[0]: np.arange(0, rms.shape[1]).tolist()
                },
                'y': rms.flatten().tolist(),
            })

    # print("Features:", features.shape)

    return {
        'features': features.flatten().tolist(),
        'graphs': graphs,
        # if you use FFTs then set the used FFTs here (this helps with memory optimization on MCUs)
        # NOTE: Unsure if this is correct
        'fft_used': [], # if not use_chroma else stft.tolist(),
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
