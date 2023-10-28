from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy import ndarray

from dataset_loader import Dataset, DatasetSplit

# Ignore division by zero warnings
np.seterr(divide='ignore')


class HiddenMarkovModel:
    states: ndarray  # States (Q)
    observations: ndarray  # Observations (O)
    transition_probabilities: ndarray  # Transition Probability Matrix (A); shape: (Q[t_{i-1}] x Q[t_{i}])
    emission_likelihoods: ndarray  # Emission Probability Matrix (B); shape (Q[ti] x O[wi])
    initial_probability_distribution: ndarray  # Initial Probability Distribution (π); shape: (Q)

    def __init__(
            self,
            dataset: Dataset,
    ):
        # Initialize the model parameters
        self.states = np.asarray(list(dataset.pos_tags))
        self.observations = np.asarray(list(dataset.vocabulary))

        self.transition_probabilities = np.zeros((len(self.states), len(self.states)))
        self.emission_likelihoods = np.zeros((len(self.states), len(self.observations)))
        self.initial_probability_distribution = np.zeros(len(self.states))

        # Fit the model to the dataset
        self.fit(dataset.train)

    def fit(self, dataset: DatasetSplit):
        # Iterate over sentences to calculate the frequencies for all model parameters
        for sentence in dataset.data:

            # Calculate the initial probability distribution
            self.initial_probability_distribution[self.states == sentence[0][1]] += 1

            # Calculate the transition probabilities and emission likelihoods
            for (token_1, tag_1), (token_2, tag_2) in pairwise(sentence):
                self.transition_probabilities[self.states == tag_1, self.states == tag_2] += 1
                self.emission_likelihoods[self.states == tag_1, self.observations == token_1] += 1
            else:
                # Add the last token to the emission likelihoods
                self.emission_likelihoods[self.states == sentence[-1][1], self.observations == sentence[-1][0]] += 1

        # Calculate log probabilities
        self.initial_probability_distribution = np.log2(self.initial_probability_distribution) - np.log2(np.sum(self.initial_probability_distribution))
        self.transition_probabilities = np.log2(self.transition_probabilities) - np.log2(np.sum(self.transition_probabilities, axis=1, keepdims=True))
        self.emission_likelihoods = np.log2(self.emission_likelihoods) - np.log2(np.sum(self.emission_likelihoods, axis=1, keepdims=True))

    def predict(self, sentence: list[str]) -> list[tuple[str, str]]:
        # Initialize the Viterbi matrix with zeros: viterbi[N, T] ← 0
        viterbi_matrix = np.zeros((len(self.states), len(sentence)))
        # Initialize the backpointers matrix with zeros: backpointers[N, T] ← 0
        backpointers = np.zeros((len(self.states), len(sentence)), dtype=int)


        # Calculate the initial probabilities: π_q ∗ b_q(o_1); as we are using log probabilities the multiplication becomes a sum
        viterbi_matrix[:, 0] = self.initial_probability_distribution + self.emission_likelihoods[:, self.observations == sentence[0]].squeeze()



if __name__ == '__main__':
    d = Dataset(
        dataset_name='UD_Basque-BDT',
        train_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-train.conllu'),
        dev_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-dev.conllu'),
        test_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-test.conllu'),
    )
    hmm = HiddenMarkovModel(d)
