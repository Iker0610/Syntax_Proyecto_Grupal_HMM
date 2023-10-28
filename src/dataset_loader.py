from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetSplit:
    name: str
    path: Path | None
    data: list[list[tuple[str, str]]] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)


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
                self.__load_dataset__(split, split_name == 'train')

        # Add splits as attributes
        for split_name, split in splits.items():
            setattr(self, split_name, split)

    def __load_dataset__(self, split: DatasetSplit, is_train: bool = False):
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
                word, _, pos_tag = line.split('\t')[1:4]
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