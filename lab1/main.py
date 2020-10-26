import math
import pandas
import functools
import time
import seaborn
from matplotlib import pyplot
from sklearn.svm import SVC


def get_xs_ys(data):
    xs = data.drop('price_range', axis=1).values.tolist()
    ys = data['price_range'].values.tolist()
    return xs, ys


def preprocess_logistic(data):
    data = data.copy()
    for col_name in data.columns:
        if col_name != 'price_range':
            s = data[col_name].max()
            data[col_name] = data[col_name].apply(lambda x: x / s)
    return data


def preprocess_bayesian(data):
    data = data.copy()
    data['battery_power'] = pandas.cut(data['battery_power'], 10, labels=False)
    data['clock_speed'] = pandas.cut(data['clock_speed'], 10, labels=False)
    data['int_memory'] = pandas.cut(data['int_memory'], 10, labels=False)
    data['mobile_wt'] = pandas.cut(data['mobile_wt'], 10, labels=False)
    data['px_height'] = pandas.cut(data['px_height'], 10, labels=False)
    data['px_width'] = pandas.cut(data['px_width'], 10, labels=False)
    data['ram'] = pandas.cut(data['ram'], 10, labels=False)
    return data


def preprocess_svm(data):
    data = data.copy()
    return data


def split(data, train, valid, test):
    size = len(data)
    size_train = int(size * train / (train + valid + test))
    size_valid = int(size * valid / (train + valid + test))

    data_train = data[0:size_train].copy()
    data_valid = data[size_train:size_train + size_valid].copy()
    data_test = data[size_train + size_valid:size].copy()

    return data_train, data_valid, data_test


def score(model, xs, ys):
    predicted_ys = model.predict(xs)
    return sum([y == predicted_y for y, predicted_y in zip(ys, predicted_ys)]) / len(xs)


def run(preprocessor, model_class, data):
    data = preprocessor(data)
    data_train, data_valid, data_test = split(data, 8, 1, 1)
    xs_train, ys_train = get_xs_ys(data_train)
    xs_valid, ys_valid = get_xs_ys(data_valid)
    xs_test, ys_test = get_xs_ys(data_test)

    model = model_class()
    time_start = time.time()
    model.fit(xs_train, ys_train)
    time_finish = time.time()

    score_test = score(model, xs_test, ys_test)
    score_train = score(model, xs_train, ys_train)
    elapsed_time = time_finish - time_start

    return score_test, score_train, elapsed_time


def compare(n):
    names = ['logistic', 'bayesian', 'svm']

    preprocessors = {
        'logistic': preprocess_logistic,
        'bayesian': preprocess_bayesian,
        'svm': preprocess_svm
    }
    model_classes = {
        'logistic': LogisticRegression,
        'bayesian': NaiveBayesianClassifier,
        'svm': SVM
    }

    avg_scores_test = {'logistic': 0.0, 'bayesian': 0.0, 'svm': 0.0}
    avg_scores_train = {'logistic': 0.0, 'bayesian': 0.0, 'svm': 0.0}
    avg_elapsed_times = {'logistic': 0.0, 'bayesian': 0.0, 'svm': 0.0}

    data = pandas.read_csv('train.csv')

    data['price_range'].replace([0, 1, 2, 3], [0, 0, 1, 1], True)

    for _ in range(n):
        data = data.sample(frac=1)
        for name in names:
            score_test, score_train, elapsed_time = run(preprocessors[name], model_classes[name], data)
            avg_scores_test[name] += score_test / n
            avg_scores_train[name] += score_train / n
            avg_elapsed_times[name] += elapsed_time / n

    print('Method    Accuracy (Test)  Accuracy (Training)  Time (sec)')
    fmt_str = '{:<8}  {:>15.2%}  {:>19.2%}  {:>10.3f}'
    for name in names:
        print(fmt_str.format(name, avg_scores_test[name], avg_scores_train[name], avg_elapsed_times[name]))

    pyplot.figure()
    seaborn.barplot(
        names * 2,
        [avg_scores_test[name] for name in names] + [avg_scores_train[name] for name in names],
        ['Test'] * 3 + ['Training'] * 3
    )
    pyplot.xlabel('Method')
    pyplot.ylabel('Accuracy')
    pyplot.show()


class LogisticRegression:
    def __init__(self):
        self.ws = []

    def fit(self, xs, ys):
        eta = 0.001
        conv_bound = 0.005
        attr_num = len(xs[0]) + 1
        self.ws = [0.0] * attr_num
        while True:
            dws = [0.0] * attr_num
            for x, y in zip(xs, ys):
                post_prob = self._calc_post_prob(x)
                dws = [dw + (y - post_prob) * attr for dw, attr in zip(dws, [1.0, *x])]
            if all([abs(eta * dw) < conv_bound for dw in dws]):
                break
            self.ws = [w + eta * dw for w, dw in zip(self.ws, dws)]

    def predict(self, xs):
        return [int(self._calc_post_prob(x) > 0.5) for x in xs]

    def _calc_post_prob(self, x):
        a = sum([w * attr for w, attr in zip(self.ws, [1.0, *x])])
        return 1 / (1 + math.exp(-a))


class NaiveBayesianClassifier:
    def __init__(self):
        self.tables = [[], []]
        self.ns = [0, 0]
        self.total = 0

    def fit(self, xs, ys):
        attr_num = len(xs[0])
        self.total = len(xs)
        for c in [0, 1]:
            self.ns[c] = ys.count(c)
            self.tables[c] = [dict() for _ in range(attr_num)]
            for x in [x for x, y in zip(xs, ys) if y == c]:
                for attr, row in zip(x, self.tables[c]):
                    row.setdefault(attr, 0)
                    row[attr] += 1

    def predict(self, xs):
        attr_num = len(xs[0])
        total = sum(self.ns)
        prior_probs = [self.ns[c] / total for c in [0, 1]]
        ys = []
        for x in xs:
            f = [0.0, 0.0]
            for c in [0, 1]:
                likelihoods = [self.tables[c][i].get(x[i], 0) / self.ns[c] for i in range(attr_num)]
                f[c] = functools.reduce(lambda v, e: v * e, [prior_probs[c], *likelihoods])
            ys.append(f.index(max(f)))
        return ys


class SVM:
    def __init__(self):
        self.svc = SVC()

    def fit(self, xs, ys):
        self.svc.fit(xs, ys)

    def predict(self, xs):
        return self.svc.predict(xs)


def main():
    compare(3)


if __name__ == '__main__':
    main()
