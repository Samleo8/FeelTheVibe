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

    # print(n_frames, n_samples, n_filter_order)

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
    lpc_coeffcients_extend = np.hstack(
        (lpc_coeffcients, np.zeros((n_frames, num_lpcc - n_filter_order))))
    # print("lpc_coeffcients_extend shape: ", lpc_coeffcients_extend.shape)

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
                      sampling_freq, lpc_order, num_lpcc, num_mfcc,
                      use_mfcc_deltas, no_mean_mfcc, use_zcr, use_rms, use_spec_centroid,
                      use_spec_rolloff):
    '''
    Generate series of features from raw data

    Subset of Features from:
    https://www.mdpi.com/2079-9292/12/4/839
    '''
    # Convert raw WAV files in int16 form to float
    raw_data = soundDataToFloat(raw_data)
    sample_rate = sampling_freq

    assert lpc_order > 0, "LPC must have a strictly positive order number"
    assert num_lpcc >= lpc_order + 1, "Number of LPCC coefficients must be larger than LPC filter order + 1"
    assert num_mfcc > 0, "Number of MFCC coefficients must be larger than 0"

    ##======LPCC=====##
    # LPC
    lpc = librosa.lpc(raw_data, order=lpc_order)[:, np.newaxis].T

    # LPCC Calculation
    # TODO: Need to check error calculation
    error = get_lpc_error(raw_data[:, np.newaxis].T, lpc)
    lpcc = lpc_to_lpcc(lpc, error, num_lpcc).flatten()
    print("LPCC:", lpcc.shape)

    features = np.copy(lpcc)

    ##======MFCC=====##
    # MFCC
    # https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
    mfcc = librosa.feature.mfcc(y=raw_data, sr=sample_rate, n_mfcc=num_mfcc)
    
    print("MFCC:", mfcc.shape)
    
    # MFCC Delta
    # NOTE: Apparently important for emotion recognition
    # https://librosa.org/doc/main/generated/librosa.feature.delta.html
    if use_mfcc_deltas:
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        # print("MFCC Delta:", mfcc_delta.shape)
        # print("MFCC Delta2:", mfcc_delta2.shape)

    # TODO: Do we mean the MFCC before or after delta?
    if not no_mean_mfcc:
        mfcc = mfcc.mean(axis=1)
        if use_mfcc_deltas:
            mfcc_delta = mfcc_delta.mean(axis=1)
            mfcc_delta2 = mfcc_delta2.mean(axis=1)

    features = np.hstack((features, mfcc))
    if use_mfcc_deltas:
        features = np.hstack((features, mfcc_delta, mfcc_delta2))

    # Zero Crossing Rate
    # https://librosa.org/doc/main/generated/librosa.feature.zero_crossing_rate.html
    if use_zcr:
        zcr = librosa.feature.zero_crossing_rate(y=raw_data)
        # print("ZCR:", zcr.shape)

        features = np.hstack((features, zcr))

    # Root Mean Square
    # https://librosa.org/doc/main/generated/librosa.feature.rms.html
    if use_rms:
        rms = librosa.feature.rms(y=raw_data)
        # print("RMS:", rms.shape)

        features = np.hstack((features, rms))

    features = np.array(features)[:, np.newaxis]

    # Initialize graphs
    graphs = []

    # Display graphs
    if draw_graphs:
        # Create image
        # https://github.com/edgeimpulse/processing-blocks/blob/master/mfcc/dsp.py
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 20.5)
        ax.set_axis_off()
        cax = ax.imshow(features.T,
                        interpolation='nearest',
                        cmap=cm.coolwarm,
                        origin='lower')

        buf = io.BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image = (base64.b64encode(buf.getvalue()).decode('ascii'))
        buf.close()

        graphs.append({
            'name': 'Feature Vector',
            'image': image,
            'imageMimeType': 'image/svg+xml',
            'type': 'image'
        })

    print("Features:", features.shape)

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

    info_dict = generate_features(implementation_version=1,
                      draw_graphs=save_img,
                      raw_data=raw_data,
                      axes=[0, 1, 2],
                      sampling_freq=16000,
                      lpc_order=10,
                      num_lpcc=13,
                      num_mfcc=13,
                      use_mfcc_deltas=True,
                      no_mean_mfcc=False,
                      use_zcr=False,
                      use_rms=False,
                      use_spec_centroid=False,
                      use_spec_rolloff=False)

    if save_img:
        imgdata = base64.b64decode(info_dict['graphs'][0]['image'])
        filename = 'test_data_lpcc.jpg'
        with open(filename, 'wb') as f:
            f.write(imgdata)
        print("Saved image representation of Long Vector features to ", filename)
