import math
import pandas


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


def split(data, train, valid, test):
    size = len(data)
    size_train = int(size * train / (train + valid + test))
    size_valid = int(size * valid / (train + valid + test))

    data_train = data[0:size_train]
    data_valid = data[size_train:size_train + size_valid]
    data_test = data[size_train + size_valid:size]

    return data_train, data_valid, data_test


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

    def predict(self, x):
        return 1 if self._calc_post_prob(x) > 0.5 else 0

    def calc_error_rate(self, xs, ys):
        return sum([y != self.predict(x) for x, y in zip(xs, ys)]) / len(xs)

    def _calc_post_prob(self, x):
        a = sum([w * attr for w, attr in zip(self.ws, [1.0, *x])])
        return 1 / (1 + math.exp(-a))


def main():
    data = pandas.read_csv('train.csv')

    data = preprocess_logistic(data)
    data_train, data_valid, data_test = split(data, 8, 1, 1)
    xs_train, ys_train = get_xs_ys(data_train)
    xs_valid, ys_valid = get_xs_ys(data_valid)
    xs_test, ys_test = get_xs_ys(data_test)

    logistic_regression = LogisticRegression()
    logistic_regression.fit(xs_train, ys_train)

    print(logistic_regression.calc_error_rate(xs_test, ys_test))


if __name__ == '__main__':
    main()
