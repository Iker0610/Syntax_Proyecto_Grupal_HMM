from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy import ndarray

from dataset_loader import Dataset, DatasetSplit

# from evaluation_metrics import accuracy, conf_matrix, f1, precision, recall

# Ignore division by zero warnings
np.seterr(divide='ignore', invalid='ignore')
EOS = -1


class DefaultDict(dict):
    def __init__(self, default_value, seq=None, **kwargs):
        super().__init__(seq, **kwargs)
        self.default_value = default_value

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.default_value


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
            unknown_token_threshold: int | float = 0.000015,  # 0.00001 doesn't filter in basque dataset
    ):
        # Calculate which tokens are converted to <UNK> tokens
        assert isinstance(unknown_token_threshold, int) or unknown_token_threshold < 1
        _frequencies = dataset.train.statistics.token_frequencies
        if isinstance(unknown_token_threshold, float):
            _total_tokens = sum(_frequencies.values())
            final_vocab = [token for token, frequency in _frequencies.items() if frequency / _total_tokens >= unknown_token_threshold]
        else:
            final_vocab = [token for token, frequency in _frequencies.items() if frequency >= unknown_token_threshold]
        final_vocab.append('<UNK>')

        # Initialize the model parameters
        self.states = list(dataset.pos_tags) + ['<EOS>']
        self.vocab = final_vocab
        self.Q = {tag: index for index, tag in enumerate(self.states)}
        self.O = {token: index for index, token in enumerate(final_vocab)}
        self.O = DefaultDict(seq=self.O, default_value=self.O['<UNK>'])

        self.transition_probabilities = np.zeros((len(self.states), len(self.states)))
        self.emission_likelihoods = np.zeros((len(self.states), len(self.vocab)))
        self.initial_probability_distribution = np.zeros(len(self.states))

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

    def batch_predict(self, sentences: list[list[str]] | list[list[tuple[str | str]]] | DatasetSplit) -> tuple[tuple[list[tuple[str, str]]], tuple[float]]:
        """
        Predict the POS tags for a given list of sentences using the Viterbi algorithm
        :param sentences: The sentences to predict the POS tags for as a list of lists of tokens
        :return: list of lists of tuples with the tokens and their predicted POS tags and the probability of the best path
        """
        if isinstance(sentences, DatasetSplit):
            sentences = [[token for token, _ in sentence] for sentence in sentences.data]
        elif isinstance(sentences[0][0], tuple):
            sentences = [[token for token, _ in sentence] for sentence in sentences]

        predictions = [self.predict(sentence) for sentence in sentences]
        return tuple(zip(*predictions))


def optimize_unk_threshold(dataset: Dataset, metric_funct: callable, min_threshold: float = 0.00001, max_threshold: float = 0.001, num: int = 250) -> dict:
    search_space = np.geomspace(min_threshold, max_threshold, num)  # The geometric sequence is used to increase the number of points near the minimum threshold
    results = np.zeros(len(search_space), dtype=np.float32)

    for i, threshold in enumerate(search_space):
        hmm = HiddenMarkovModel(dataset, threshold)
        predictions = hmm.batch_predict(dataset.dev)
        y_gold = [tag for sentence in dataset.dev.data for _, tag in sentence]
        y_pred = [tag for sentence in predictions for _, tag in sentence]
        metric_funct(y_gold, y_pred)
        results[i] = metric_funct(y_gold, y_pred)

    # Obtain the best threshold
    best_threshold = np.argmax(results)

    return {
        'best_threshold': search_space[best_threshold],
        'best_metric': results[best_threshold],
        'search_space': search_space,
        'results': results,
    }


if __name__ == '__main__':
    d = Dataset(
        dataset_name='UD_Basque-BDT',
        train_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-train.conllu'),
        dev_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-dev.conllu'),
        test_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-test.conllu'),
    )
    hmm = HiddenMarkovModel(d)
    hmm.batch_predict(d.test)
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
