from copy import copy
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy import ndarray

from dataset_loader import Dataset, DatasetSplit

# from evaluation_metrics import accuracy, conf_matrix, f1, precision, recall

# Ignore division by zero warnings
np.seterr(divide='ignore')
EOS = -1


class HiddenMarkovModel:
    # TODO: Add <UNK> token to the vocabulary and the states and define a minimum frequency threshold

    Q: dict[str, int]  # States (Q)
    O: dict[str, int]  # Observations (O)
    transition_probabilities: ndarray  # Transition Probability Matrix (A); shape: (Q[t_{i-1}] x Q[t_{i}])
    emission_likelihoods: ndarray  # Emission Probability Matrix (B); shape (Q[ti] x O[wi])
    initial_probability_distribution: ndarray  # Initial Probability Distribution (π); shape: (Q)

    def __init__(
            self,
            dataset: Dataset,
    ):
        # Initialize the model parameters
        self.states = list(dataset.pos_tags) + ['<EOS>']
        self.Q = {tag: index for index, tag in enumerate(self.states)}
        self.O = {token: index for index, token in enumerate(dataset.vocabulary)}

        self.transition_probabilities = np.zeros((len(self.Q), len(self.Q)))
        self.emission_likelihoods = np.zeros((len(self.Q), len(self.O)))
        self.initial_probability_distribution = np.zeros(len(self.Q))

        # Fit the model to the dataset
        self.fit(dataset.train)

    def fit(self, dataset: DatasetSplit):
        # Iterate over sentences to calculate the frequencies for all model parameters
        for sentence in dataset.data:

            # Calculate the initial probability distribution
            self.initial_probability_distribution[self.Q[sentence[0][1]]] += 1

            # Calculate the transition probabilities and emission likelihoods
            for (token_1, tag_1), (token_2, tag_2) in pairwise(sentence):
                self.transition_probabilities[self.Q[tag_1], self.Q[tag_2]] += 1
                self.emission_likelihoods[self.Q[tag_1], self.O[token_1]] += 1
            else:
                # Add the last token to the emission likelihoods
                self.emission_likelihoods[self.Q[sentence[-1][1]], self.O[sentence[-1][0]]] += 1

                # Add the transition probabilities to the <EOS> state
                self.transition_probabilities[self.Q[sentence[-1][1]], EOS] += 1

        # Calculate log probabilities
        self.initial_probability_distribution = np.log2(self.initial_probability_distribution) - np.log2(np.sum(self.initial_probability_distribution))
        self.transition_probabilities = np.log2(self.transition_probabilities) - np.log2(np.sum(self.transition_probabilities, axis=1, keepdims=True))
        self.emission_likelihoods = np.log2(self.emission_likelihoods) - np.log2(np.sum(self.emission_likelihoods, axis=1, keepdims=True))

        # Set the initial probability distribution likelihoods for the <EOS> token to -inf
        self.initial_probability_distribution[EOS] = -np.inf

        # Set the transition probabilities likelihoods for the <EOS> token to -inf
        self.transition_probabilities[EOS] = -np.inf
        self.emission_likelihoods[EOS] = -np.inf

    def predict(self, sentence: list[str]) -> tuple[list[tuple[str, str]], float]:
        """
        Predict the POS tags for a given sentence using the Viterbi algorithm
        :param sentence: The sentence to predict the POS tags for as a list of tokens
        :return: list of tuples with the tokens and their predicted POS tags, and the probability of the best path
        """
        assert len(sentence) > 0, 'The sentence must contain at least one token'
        # sentence = copy(sentence) + ['<EOS>']

        # -----------------------------------------------------------------------------------------------------------------------

        # Initialize the Viterbi matrix with zeros: viterbi[N, T] ← 0
        viterbi_matrix = np.zeros((len(self.Q), len(sentence) + 1))
        # Initialize the backpointers matrix with zeros: backpointers[N, T] ← 0
        backpointers = np.zeros((len(self.Q), len(sentence) + 1), dtype=int)

        # -----------------------------------------------------------------------------------------------------------------------

        # Calculate the initial probabilities: π_q ∗ b_q(o_1); as we are using log probabilities the multiplication becomes a sum
        viterbi_matrix[:, 0] = self.initial_probability_distribution + self.emission_likelihoods[:, self.O[sentence[0]]]

        # -----------------------------------------------------------------------------------------------------------------------

        # Calculate the probabilities for the remaining tokens
        for t, token in enumerate(sentence[1:], start=1):
            # for q, state in enumerate(self.Q):
            #     # viterbi[q, t] = max viterbi[q′, t − 1] ∗ A[q′,q] ∗ b_q (o_t)
            #     viterbi_matrix[q, t] = np.max(viterbi_matrix[:, t - 1] + self.transition_probabilities[:, q]) + self.emission_likelihoods[q, self.O[token]]
            #     # backpointers[q, t] = argmax viterbi[q′, t − 1] ∗ A[q′,q]
            #     backpointers[q, t] = np.argmax(viterbi_matrix[:, t - 1] + self.transition_probabilities[:, q])

            conditioned_transition_probabilities = viterbi_matrix[:, [t - 1]] + self.transition_probabilities[:, :]
            viterbi_matrix[:, t] = np.max(conditioned_transition_probabilities, axis=0) + self.emission_likelihoods[:, self.O[token]]
            backpointers[:, t] = np.argmax(conditioned_transition_probabilities, axis=0)
        else:
            # EOS probabilities
            viterbi_matrix[:, -1] = np.max(viterbi_matrix[:, -2] + self.transition_probabilities[:, EOS])
            backpointers[:, -1] = np.argmax(viterbi_matrix[:, -2] + self.transition_probabilities[:, EOS])

        # -----------------------------------------------------------------------------------------------------------------------

        # Termination step: calculate best path probability and best path pointer
        best_path_probability = np.max(viterbi_matrix[:, -1])
        best_path_pointer = np.argmax(viterbi_matrix[:, -1])

        # Backtrack to find the best path
        best_path = [best_path_pointer]
        for t in range(len(sentence), 0, -1):
            best_path.append(backpointers[best_path[-1], t])

        # best_path = best_path[1:]
        # Reverse the best path and return the predicted tags
        return [(token, self.states[tag]) for token, tag in zip(sentence, reversed(best_path))], best_path_probability


if __name__ == '__main__':
    d = Dataset(
        dataset_name='UD_Basque-BDT',
        train_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-train.conllu'),
        dev_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-dev.conllu'),
        test_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-test.conllu'),
    )
    hmm = HiddenMarkovModel(d)
    gold = d.train.data[0]
    pred = hmm.predict([token for token, _ in d.train.data[0]])

    y_gold = [tag for word, tag in gold]
    y_pred = [tag for word, tag in pred[0]]

    print('Pred:', pred)
    print('Gold:', gold)

    # print('accuracy: ', accuracy(y_gold, y_pred))
    # print('precision: ', precision(y_gold, y_pred, 'NOUN'))
    # print('recall: ', recall(y_gold, y_pred, 'NOUN'))
    # print('fscore: ', f1(y_gold, y_pred, 'NOUN'))
    #
    # conf_matrix(y_gold, y_pred, list(d.pos_tags))
