import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import joblib


class EEGPreprocessor:

    def __init__(self, lowcut=4, highcut=45, fs=128):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.scaler = StandardScaler()


    # -------------------------
    # Bandpass Filter
    # -------------------------

    def bandpass_filter(self, data):

        nyquist = 0.5 * self.fs

        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        b, a = butter(
            N=4,
            Wn=[low, high],
            btype="band"
        )

        filtered = filtfilt(
            b,
            a,
            data,
            axis=-1
        )

        return filtered


    # -------------------------
    # Normalization
    # -------------------------

    def normalize(self, data):

        original_shape = data.shape

        # Convert:
        # trials, channels, samples
        # into:
        # samples, channels

        data = data.reshape(
            -1,
            original_shape[1]
        )

        data = self.scaler.fit_transform(data)


        data = data.reshape(
            original_shape
        )

        return data



    # -------------------------
    # Complete Pipeline
    # -------------------------

    def preprocess(self, eeg):

        print("Original Shape:")
        print(eeg.shape)


        print("\nApplying Bandpass Filter...")

        eeg = self.bandpass_filter(eeg)


        print("Normalizing EEG...")

        eeg = self.normalize(eeg)


        print("\nPreprocessing Completed")

        print("Final Shape:")
        print(eeg.shape)


        return eeg



    # Save scaler

    def save_scaler(self, path):

        joblib.dump(
            self.scaler,
            path
        )


    # Load scaler

    def load_scaler(self,path):

        self.scaler = joblib.load(path)



if __name__ == "__main__":

    from load_data import DEAPLoader


    loader = DEAPLoader()

    eeg, labels = loader.load_all_subjects()


    processor = EEGPreprocessor()


    clean_eeg = processor.preprocess(eeg)


    print("\nSample EEG Value:")
    print(clean_eeg[0][0][:10])