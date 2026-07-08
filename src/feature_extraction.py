import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis


class EEGFeatureExtractor:


    def __init__(self, fs=128):
        self.fs = fs



    # -------------------------
    # Time Domain Features
    # -------------------------

    def time_features(self, signal):

        features = []

        features.append(np.mean(signal))

        features.append(np.var(signal))

        features.append(np.std(signal))

        features.append(
            np.sqrt(np.mean(signal**2))
        )

        features.append(skew(signal))

        features.append(kurtosis(signal))


        return features



    # -------------------------
    # Frequency Features
    # -------------------------

    def band_power(self, signal):

        freqs, psd = welch(
            signal,
            fs=self.fs
        )


        bands = {

            "delta":(0.5,4),

            "theta":(4,8),

            "alpha":(8,13),

            "beta":(13,30),

            "gamma":(30,45)

        }


        features=[]


        for band in bands.values():

            low,high = band

            idx = np.logical_and(
                freqs>=low,
                freqs<=high
            )


            power = np.trapezoid(
    psd[idx],
    freqs[idx]
)


            features.append(power)


        return features




    # -------------------------
    # Extract All Features
    # -------------------------

    def extract(self,eeg):


        all_features=[]


        print("Extracting Features...")


        for trial in range(eeg.shape[0]):


            trial_features=[]


            for channel in range(eeg.shape[1]):


                signal=eeg[trial,channel]


                trial_features.extend(
                    self.time_features(signal)
                )


                trial_features.extend(
                    self.band_power(signal)
                )



            all_features.append(
                trial_features
            )


        features=np.array(all_features)


        print(
            "Feature Shape:",
            features.shape
        )


        return features




if __name__=="__main__":


    from load_data import DEAPLoader
    from preprocess import EEGPreprocessor


    loader=DEAPLoader()


    eeg,labels=loader.load_all_subjects()



    processor=EEGPreprocessor()


    eeg=processor.preprocess(eeg)



    extractor=EEGFeatureExtractor()


    X=extractor.extract(eeg)



    np.save(
        "outputs/features.npy",
        X
    )


    np.save(
        "outputs/labels.npy",
        labels
    )


    print("Features Saved")