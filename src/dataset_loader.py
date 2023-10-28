class Dataset:

    def __init__(self, dataset_name_or_path):
        self.dataset_name_or_path = dataset_name_or_path
        self.sentences = []

    def load_dataset(self):
        with open(self.dataset_name_or_path) as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == '#':
                    pass
                elif line == '\n':
                    self.sentences.append(new_sentence)
                else:
                    parsed_line = line.split('\t')
                    if parsed_line[0] == '1':
                        new_sentence = []
                    word, tag = (parsed_line[2], parsed_line[3])
                    new_sentence.append({word: tag})
            print(f"Sentences = {self.sentences}")


if __name__ == '__main__':
    d = Dataset('../data/UD_Basque-BDT/eu_bdt-ud-train.conllu')
    d.load_dataset()
    print(len(d.sentences))
    ind = 1
    for s in d.sentences:
        print(f"{ind} --> {s}")
        ind += 1