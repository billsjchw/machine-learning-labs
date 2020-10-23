import math
import pandas
import functools
from sklearn.svm import SVC


def get_xs_ys(data):
    xs = data.drop('price_range', axis=1).values.tolist()
    ys = data['price_range'].values.tolist()
    return xs, ys


def preprocess_logistic(data):
    data = data.copy()
    data['price_range'].replace([0, 1, 2, 3], [0, 0, 1, 1], True)
    for col_name in data.columns:
        if col_name != 'price_range':
            s = data[col_name].max()
            data[col_name] = data[col_name].apply(lambda x: x / s)
    return data


def preprocess_bayesian(data):
    data = data.copy()
    data['price_range'].replace([0, 1, 2, 3], [0, 0, 1, 1], True)
    data['battery_power'] = pandas.cut(data['battery_power'], 10, labels=False)
    data['clock_speed'] = pandas.cut(data['clock_speed'], 10, labels=False)
    data['fc'] = pandas.cut(data['fc'], 10, labels=False)
    data['int_memory'] = pandas.cut(data['int_memory'], 10, labels=False)
    data['mobile_wt'] = pandas.cut(data['mobile_wt'], 10, labels=False)
    data['px_height'] = pandas.cut(data['px_height'], 10, labels=False)
    data['px_width'] = pandas.cut(data['px_width'], 10, labels=False)
    data['ram'] = pandas.cut(data['ram'], 10, labels=False)
    return data


def preprocess_svm(data):
    data = data.copy()
    data['price_range'].replace([0, 1, 2, 3], [0, 0, 1, 1], True)
    return data


def split(data, train, valid, test):
    size = len(data)
    size_train = int(size * train / (train + valid + test))
    size_valid = int(size * valid / (train + valid + test))

    data_train = data[0:size_train]
    data_valid = data[size_train:size_train + size_valid]
    data_test = data[size_train + size_valid:size]

    return data_train, data_valid, data_test


def score(model, xs, ys):
    predicted_ys = model.predict(xs)
    return sum([y != predicted_y for y, predicted_y in zip(ys, predicted_ys)]) / len(xs)


def run(preprocessor, model_class, data):
    data = preprocessor(data)
    data_train, data_valid, data_test = split(data, 8, 1, 1)
    xs_train, ys_train = get_xs_ys(data_train)
    xs_valid, ys_valid = get_xs_ys(data_valid)
    xs_test, ys_test = get_xs_ys(data_test)

    model = model_class()
    model.fit(xs_train, ys_train)

    print(score(model, xs_test, ys_test))


class LogisticRegression:
    def __init__(self):
        self.ws = []

    def fit(self, xs, ys):
        eta = 0.001
        conv_bound = 0.003
        attr_num = len(xs[0]) + 1
        self.ws = [0.0] * attr_num
        while True:
            dws = [0.0] * attr_num
            for x, y in zip(xs, ys):
                prob = self._calc_post_prob(x)
                dws = [dw + (y - prob) * attr for dw, attr in zip(dws, [1.0, *x])]
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
        ys = []
        for x in xs:
            f = [0.0, 0.0]
            for c in [0, 1]:
                p = self.ns[c] / self.total
                ps = [self.tables[c][i].get(x[i], 0) / self.ns[c] for i in range(attr_num)]
                f[c] = functools.reduce(lambda v, e: v * e, [p, *ps])
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
    data = pandas.read_csv('train.csv')

    run(preprocess_logistic, LogisticRegression, data)
    run(preprocess_bayesian, NaiveBayesianClassifier, data)
    run(preprocess_svm, SVM, data)


if __name__ == '__main__':
    main()
