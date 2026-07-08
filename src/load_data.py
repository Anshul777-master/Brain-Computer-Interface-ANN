import os
import sys
import pickle
import numpy as np

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class DEAPLoader:

    def __init__(self, dataset_path=config.DATASET_PATH):
        self.dataset_path = dataset_path

    def load_subject(self, filename):

        filepath = os.path.join(self.dataset_path, filename)

        with open(filepath, "rb") as file:
            data = pickle.load(file, encoding="latin1")

        eeg = data["data"][:, :32, :]
        labels = data["labels"]

        return eeg, labels

    def load_all_subjects(self):

        all_eeg = []
        all_labels = []

        files = sorted(os.listdir(self.dataset_path))

        for file in files:

            if file.endswith(".dat"):

                eeg, labels = self.load_subject(file)

                all_eeg.append(eeg)
                all_labels.append(labels)

                print(f"Loaded {file}")

        all_eeg = np.concatenate(all_eeg, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return all_eeg, all_labels


if __name__ == "__main__":

    loader = DEAPLoader()

    eeg, labels = loader.load_all_subjects()

    print()

    print("EEG Shape :", eeg.shape)

    print("Labels Shape :", labels.shape)

    print()

    print("Example Label")

    print(labels[0])