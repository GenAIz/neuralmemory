# A simple script to calculate a baseline for POS tagging, if
# POS tagging was dependent on only the current token.
# Scores above this one are evidence of working memory.


from sklearn.metrics import precision_recall_fscore_support

from nm import posdata


class BaselineModel:

    def __init__(self):
        self.mapping: dict[str, dict[str, int]] = dict()

    def count(self, word: str, label: str) -> None:
        if word not in posdata.GLOVE_840B_300:
            return
        if word not in self.mapping:
            self.mapping[word] = dict()
        if label not in self.mapping[word]:
            self.mapping[word][label] = 0
        self.mapping[word][label] += 1

    def predict(self, word: str) -> str:
        if word not in self.mapping:
            return "NOT_KNOWN"
        best = None
        best_count = 0
        for label in self.mapping[word]:
            if best is None or best_count < self.mapping[word][label]:
                best = label
                best_count = self.mapping[word][label]
        return best


def train() -> BaselineModel:
    baseline = BaselineModel()
    for words, _, tags in posdata.UDPOS_DATASETS[0]:
        for i in range(len(words)):
            baseline.count(words[i], tags[i])
    return baseline


def evaluate(model: BaselineModel, dataset) -> tuple[float, float, float]:
    all_predictions = []
    all_targets = []
    for words, _, tags in dataset:
        for i in range(len(words)):
            predicted = model.predict(words[i])
            target = tags[i]
            all_predictions.append(predicted)
            all_targets.append(target)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted', zero_division=0)
    return p, r, f1


def main():
    print("train baseline")
    baseline_model = train()
    p, r, f1 = evaluate(baseline_model, posdata.UDPOS_DATASETS[0])
    print("TRAIN Precision {}, Recall {}, F1 {}".format(p, r, f1))
    p, r, f1 = evaluate(baseline_model, posdata.UDPOS_DATASETS[1])
    print("VALIDATION Precision {}, Recall {}, F1 {}".format(p, r, f1))


if __name__ == "__main__":
    main()
