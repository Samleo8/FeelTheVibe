import base64
import io
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def soundDataToFloat(data):
    "Converts integer representation back into librosa-friendly floats, given a numpy array SD"
    INT_MAX = 32768
    return (np.array(data) / INT_MAX).astype(np.float32)

def get_lpc_error(signal, lpc_coeffcients):
    '''
    Calculate the prediction error signal of the LPC filter coefficients using autocorrlation
    Reference: 
        https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/120_lpc%20prediction%20error.pdf

    Inputs
    ------
    signal: input signal (n_frames, n_samples)
    lpc_coeffcients: Filter coefficients (n_frames, n_filter_order+1)
    '''
    n_frames, n_samples = signal.shape
    _, n_filter_order = lpc_coeffcients.shape

    # Calculate autocorrelation
    autocorr = librosa.autocorrelate(signal, max_size=n_filter_order)

    err = np.sum(lpc_coeffcients * autocorr, axis=1)
    err += np.finfo(float).eps

    # TODO: Probably need to square root, since we want RMS?
    return np.sqrt(err)
    # return err

def lpc_to_lpcc(lpc_coeffcients, error, num_lpcc):
    '''
    Calculate LPCC coefficients from LPC filter coefficients
    Algorithm from https://www.mathworks.com/help/dsp/ref/lpctofromcepstralcoefficients.html

    Inputs
    ------
    lpc_coeffcients: Filter coefficients (n_frames, n_filter_order+1)
    error: Error power (n_frames, )
    num_lpcc: number of LPCC coefficients to calculate (int)

    Returns
    -------
    lpcc: LPCC coefficients (n_frames, num_lpcc)
    '''
    n_frames, n_filter_order = lpc_coeffcients.shape
    lpcc = np.zeros((n_frames, num_lpcc))
    lpcc[:, 0] = np.log(error)

    # For extended LPCC, pad with zeros
    lpc_coeffcients_extend = np.hstack((lpc_coeffcients, np.zeros((n_frames, num_lpcc - n_filter_order))))
    print("lpc_coeffcients_extend shape: ", lpc_coeffcients_extend.shape)

    for m in range(1, num_lpcc):
        # NOTE: Take advantage of the fact that until next iteration, rest of lpcc part is 0
        a_m = -lpc_coeffcients_extend[:, m]

        m_minus_k = np.arange(m - 1, 0, -1)
        c_mminusk = lpcc[:, m_minus_k]

        # note that extended version includes 0s, which is important to cancel out unwanted parts
        a_k = lpc_coeffcients_extend[:, 1:m]
        sm_array = m_minus_k * a_k * c_mminusk

        # Vectorized operation :)
        c_m = -a_m - np.sum(sm_array, axis=1) / m
        lpcc[:, m] = c_m

    return lpcc

def generate_features(implementation_version, draw_graphs, raw_data, axes,
                      sampling_freq, lpc_order, num_lpcc, frame_len_ms, hop_len_ms):
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

    # Split into frames
    # https://stackoverflow.com/a/66921909
    # NOTE: Now a parameter. Default values based on paper

    # Convert frame length in ms to n_samples, using sampling frequency
    MS_TO_S = 1 / 1000
    frame_len = int(MS_TO_S * frame_len_ms * sampling_freq)
    hop_len = int(MS_TO_S * hop_len_ms * sampling_freq)

    print("Sample Freq", sampling_freq)
    print("Frame length: ", frame_len_ms, "ms,", frame_len, "samples")
    print("Hop length: ", hop_len_ms, "ms,", hop_len, "samples")

    # frame_len = 2048 # TODO: using default value from librosa, based on N_FFT
    # hop_len = 492 # TODO: not sure what this should be, this forces it to be the (default) 126 columns

    frames = librosa.util.frame(raw_data, frame_length=frame_len, hop_length=hop_len).T
    windowed_frames = np.hanning(frame_len) * frames

    print("Frames shape: ", frames.shape)
    print("Windowed frames shape: ", windowed_frames.shape)

    # LPC
    lpc = librosa.lpc(windowed_frames, order=lpc_order)

    # LPCC Calculation
    # TODO: Need to check error calculation
    error = get_lpc_error(windowed_frames, lpc)
    lpcc = lpc_to_lpcc(lpc, error, num_lpcc)

    # NOTE: Transpose to get into right shape for NN
    lpcc = lpcc.T

    # Display graphs
    if draw_graphs:
        # Create image
        # https://github.com/edgeimpulse/processing-blocks/blob/master/mfcc/dsp.py
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 20.5)
        ax.set_axis_off()
        cax = ax.imshow(lpcc,
                        interpolation='nearest',
                        cmap=cm.coolwarm,
                        origin='lower')

        buf = io.BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image = (base64.b64encode(buf.getvalue()).decode('ascii'))
        buf.close()

        graphs.append({
            'name': 'LPCC Features',
            'image': image,
            'imageMimeType': 'image/svg+xml',
            'type': 'image'
        })

    # Concat features
    features = np.vstack((features, lpcc)) \
        if features is not None else lpcc

    print("Features shape: ", features.shape)

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
    save_img = bool(int(sys.argv[1])) if len(sys.argv) > 1 else False

    info_dict = generate_features(
        implementation_version=1,
        draw_graphs=save_img,
        raw_data=raw_data,
        axes=[0,1,2],
        sampling_freq=16000,
        num_lpcc=20,
        lpc_order=16
    )

    if save_img:
        imgdata = base64.b64decode(info_dict['graphs'][0]['image'])
        filename = 'test_data_lpcc.jpg'
        with open(filename, 'wb') as f:
            f.write(imgdata)
        print("Saved image representation of LPCC features to ", filename)