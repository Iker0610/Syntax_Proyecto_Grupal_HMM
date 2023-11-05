from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise
from pathlib import Path

import pandas as pd


@dataclass
class DatasetSplitStatistics:
    token_frequencies: dict[str, int]
    individual_tag_frequencies: dict
    tag_bigrams_frequencies: pd.DataFrame
    sentences_length: list[int]
    token_distribution_per_pos_tag: dict[str, dict[str, int]]
    pos_tag_distribution_per_token: dict[str, dict[str, int]]


@dataclass
class DatasetSplit:
    name: str
    path: Path | None
    data: list[list[tuple[str, str]]] = field(default_factory=list)

    @cached_property
    def statistics(self) -> DatasetSplitStatistics:
        pos_tags = set([tag for sentence in self.data for _, tag in sentence])

        individual_tag_frequencies = dict.fromkeys(pos_tags, 0)
        token_frequencies = defaultdict(int)
        token_distribution_per_pos_tag = {dataset_pos_tag: {} for dataset_pos_tag in pos_tags}
        pos_tag_distribution_per_token = defaultdict(lambda: defaultdict(int))

        tag_bigrams_frequencies = pd.DataFrame(
            data=0,
            index=['<BEGIN>'] + list(pos_tags),
            columns=list(pos_tags) + ['<END>'],
        )

        for sentence in self.data:
            for token, tag in sentence:
                # Calculate the frequency of each tag and token within each data split
                individual_tag_frequencies[tag] += 1
                token_frequencies[token] += 1

                # Calculate the distribution of tokens per tag within each data split
                if token not in token_distribution_per_pos_tag[tag]:
                    token_distribution_per_pos_tag[tag][token] = 0
                token_distribution_per_pos_tag[tag][token] += 1

                # Calculate the distribution PoS tags per token within each data split
                pos_tag_distribution_per_token[token][tag] += 1

            # Calculate the frequency of each tag pair within each data split
            for (_, tag_1), (_, tag_2) in pairwise([(None, '<BEGIN>')] + sentence + [(None, '<END>')]):
                tag_bigrams_frequencies[tag_2][tag_1] += 1

        # Calculate length of each sentence and the average length of the data split
        sentences_length = [len(sentence) for sentence in self.data]

        return DatasetSplitStatistics(
            token_frequencies=token_frequencies,
            individual_tag_frequencies=individual_tag_frequencies,
            tag_bigrams_frequencies=tag_bigrams_frequencies,
            sentences_length=sentences_length,
            token_distribution_per_pos_tag=token_distribution_per_pos_tag,
            pos_tag_distribution_per_token=pos_tag_distribution_per_token,
        )


class Dataset:
    pos_tags: set[str]
    vocabulary: set[str]

    train: DatasetSplit
    dev: DatasetSplit
    test: DatasetSplit

    def __init__(
            self,
            train_path: Path | str,
            dataset_name: str = 'POS Tagging Dataset',
            dev_path: Path | None = None,
            test_path: Path | None = None,
            lemmatized: bool = False,
    ):
        self.dataset_name = dataset_name
        self.pos_tags = set()
        self.vocabulary = set()
        splits = {
            'train': DatasetSplit('train', train_path),
            'dev': DatasetSplit('dev', dev_path),
            'test': DatasetSplit('test', test_path),
        }

        # Load splits
        for split_name, split in splits.items():
            if split.path is not None:
                self.__load_dataset__(split, split_name == 'train', lemmatized)

        # Add splits as attributes
        for split_name, split in splits.items():
            setattr(self, split_name, split)

    def __load_dataset__(self, split: DatasetSplit, is_train: bool = False, lemmatized: bool = False):
        token_column = 2 if lemmatized else 1
        with open(split.path, encoding='utf8') as f:
            lines = f.readlines()

        new_sentence = []
        for line in lines:
            if line[0] == '#':
                continue

            elif line == '\n' and new_sentence:
                split.data.append(new_sentence)
                new_sentence = []

            else:
                _line = line.split('\t', maxsplit=4)
                word, pos_tag = _line[token_column], _line[3]
                new_sentence.append((word, pos_tag))
                if is_train:
                    self.pos_tags.add(pos_tag)
                    self.vocabulary.add(word)


if __name__ == '__main__':
    d = Dataset(
        dataset_name='UD_Basque-BDT',
        train_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-train.conllu'),
        dev_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-dev.conllu'),
        test_path=Path('../data/UD_Basque-BDT/eu_bdt-ud-test.conllu'),
    )

    for ind, s in enumerate(d.train.data, start=1):
        print(f"{ind} --> {s}")

    print(f"pos_tags ({len(d.pos_tags)}) = {d.pos_tags}")

    print(f"Individual tag frequencies (train) --> {d.train.statistics.individual_tag_frequencies}")
    print(f"Sentences length (train) --> {d.train.statistics.sentences_length}")
    # print(f"Sentences average length (train) --> {d.train.statistics.sentences_average_length} tokens")

    print("Token distribution per PoS tag (train)")
    for pos_tag, token_distribution in d.train.statistics.token_distribution_per_pos_tag.items():
        if len(token_distribution) < 100:  # Just to try values are generated correctly (PROVISIONAL)
            print(f"{pos_tag} --> {token_distribution}")

    print("PoS tag distribution per token (train)")
    for token, pos_tag_distribution in d.train.statistics.pos_tag_distribution_per_token.items():
        if len(pos_tag_distribution) > 1:  # Just to try values are generated correctly (PROVISIONAL)
            print(f"{token} --> {pos_tag_distribution}")
