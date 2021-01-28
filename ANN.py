import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
import mlrose
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
from collections import defaultdict


# Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(title, train_sizes, train_scores, test_scores, multi_run=True, x_label='Training examples', x_scale='linear', y_label='Score', y_scale='linear'):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()
    ax = plt.gca()
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    train_points = train_scores
    test_points = test_scores

    if multi_run:
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        train_points = train_scores_mean
        test_points = test_scores_mean

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2)

    plt.plot(train_sizes, train_points, 'o-', linewidth=1, markersize=4, label="Training score")
    plt.plot(train_sizes, test_points, 'o-', linewidth=1, markersize=4, label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def get_score(y_true, y_pred):
    weights = compute_sample_weight('balanced', y_true)
    return f1_score(y_true, y_pred, average='binary', sample_weight=weights)


class ANN:
    _test_size = 0.2
    _seed = 7641

    def __init__(self, file, output_path):
        abs_dir = os.path.dirname(__file__)
        self._file = Path(os.path.join(abs_dir, file))

        self._data = pd.DataFrame()
        self.output_path = Path(os.path.join(abs_dir, output_path))
        self.X = None
        self.Y = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self.num_records = None
        self.num_features = None
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    def load_data(self):
        self._data = pd.read_csv(self._file, header=None)
        self.X = np.array(self._data.iloc[:, :-1])
        self.Y = np.array(self._data.iloc[:, -1])
        self.num_records, self.num_features = self.X.shape
        self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
            self.X, self.Y, test_size=self._test_size, random_state=self._seed, stratify=self.Y
        )

    def scale_data(self):
        self.X = StandardScaler().fit_transform(self.X)
        self.training_x = StandardScaler().fit_transform(self.training_x)
        self.testing_x = StandardScaler().fit_transform(self.testing_x)

    def rhc(self, max_iter=10000, restarts=0):
        learner = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu', algorithm='random_hill_climb',
                                       max_iters=max_iter, bias=True, learning_rate=3e-03, early_stopping=True,
                                       max_attempts=1000, restarts=restarts)
        return learner

    def sa(self, max_iter=10000, schedule=None):
        if schedule is None:
            schedule = mlrose.ExpDecay()
        learner = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu', algorithm='simulated_annealing',
                                       max_iters=max_iter, bias=True, learning_rate=3e-03, early_stopping=True,
                                       max_attempts=1000, schedule=schedule)
        return learner

    def ga(self, max_iter=10000, mutation_prob=0.1, pop_size=200):
        learner = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu', algorithm='genetic_alg',
                                       max_iters=max_iter, bias=True, learning_rate=3e-03, early_stopping=True,
                                       max_attempts=100, mutation_prob=mutation_prob, pop_size=pop_size)
        return learner

    def learning_curve(self):
        # draw learning curve
        for nn_model, model_name in zip([self.rhc(), self.sa(), self.ga()], ['RHC', 'SA', 'GA']):
            train_sizes, train_scores, test_scores = ms.learning_curve(
                nn_model,
                self.training_x,
                self.training_y,
                cv=5,
                train_sizes=np.linspace(0.1, 1, 20),
                random_state=self._seed)

            lc_plt = plot_learning_curve('Learning Curve: {}'.format(model_name), train_sizes, train_scores, test_scores)
            lc_plt.savefig(os.path.join(self.output_path, '{}_learning_curve.png'.format(model_name)), dpi=150)

    def iter_learn_curve(self):
        output_path = self.output_path
        np.random.seed(self._seed)

        res = defaultdict(list)
        clf_name = 'RHC'
        param_name = 'Number of Iteration'
        for value in range(1000, 50000, 2000):
            clf = self.rhc(max_iter=value)
            res['param_{}'.format(param_name)].append(value)
            clf.fit(self.training_x, self.training_y)
            pred_y = clf.predict(self.training_x)
            res['train acc'].append(get_score(self.training_y, pred_y))
            pred_y = clf.predict(self.testing_x)
            res['test acc'].append(get_score(self.testing_y, pred_y))

        res = pd.DataFrame(res)
        res.to_csv(os.path.join(output_path, '{}_iter_result.csv'.format(clf_name)), index=False)
        this_plt = plot_learning_curve('Learning Curve: {}'.format(clf_name, ),
                                       res['param_{}'.format(param_name)], res['train acc'], res['test acc'],
                                       multi_run=False, x_label=clf_name, x_scale='log')
        this_plt.savefig(os.path.join(output_path, '{}_iter_learning_curve.png'.format(clf_name)), dpi=150)

        res = defaultdict(list)
        clf_name = 'SA'
        param_name = 'Number of Iteration'
        for value in range(1000, 50000, 2000):
            clf = self.sa(max_iter=value)
            res['param_{}'.format(param_name)].append(value)
            clf.fit(self.training_x, self.training_y)
            pred_y = clf.predict(self.training_x)
            res['train acc'].append(get_score(self.training_y, pred_y))
            pred_y = clf.predict(self.testing_x)
            res['test acc'].append(get_score(self.testing_y, pred_y))

        res = pd.DataFrame(res)
        res.to_csv(os.path.join(output_path, '{}_iter_result.csv'.format(clf_name)), index=False)
        this_plt = plot_learning_curve('Learning Curve: {}'.format(clf_name, ),
                                       res['param_{}'.format(param_name)], res['train acc'], res['test acc'],
                                       multi_run=False, x_label=clf_name, x_scale='log')
        this_plt.savefig(os.path.join(output_path, '{}_iter_learning_curve.png'.format(clf_name)), dpi=150)

        res = defaultdict(list)
        clf_name = 'GA'
        param_name = 'Number of Iteration'
        for value in range(1000, 50000, 2000):
            clf = self.ga(max_iter=value)
            res['param_{}'.format(param_name)].append(value)
            clf.fit(self.training_x, self.training_y)
            pred_y = clf.predict(self.training_x)
            res['train acc'].append(get_score(self.training_y, pred_y))
            pred_y = clf.predict(self.testing_x)
            res['test acc'].append(get_score(self.testing_y, pred_y))

        res = pd.DataFrame(res)
        res.to_csv(os.path.join(output_path, '{}_iter_result.csv'.format(clf_name)), index=False)
        this_plt = plot_learning_curve('Learning Curve: {}'.format(clf_name, ),
                                       res['param_{}'.format(param_name)], res['train acc'], res['test acc'],
                                       multi_run=False, x_label=clf_name, x_scale='log')
        this_plt.savefig(os.path.join(output_path, '{}_iter_learning_curve.png'.format(clf_name)), dpi=150)

    def perform_iteration(self):
        train_acc_rhc = []
        test_acc_rhc = []
        train_acc_sa = []
        test_acc_sa = []
        train_acc_ga = []
        test_acc_ga = []
        time_rhc = []
        time_sa = []
        time_ga = []
        for i in range(1000, 50000, 2000):
            print(i)
            nn_model_rhc = self.rhc(i)

            st = time.time()
            nn_model_rhc.fit(self.training_x, self.training_y)
            end = time.time()
            ft = end - st

            # Predict labels for train set and assess accuracy
            training_y_pred_rhc = nn_model_rhc.predict(self.training_x)
            training_y_accuracy_rhc = accuracy_score(self.training_y, training_y_pred_rhc)
            train_acc_rhc.append(training_y_accuracy_rhc)

            # Predict labels for test set and assess accuracy
            y_test_pred_rhc = nn_model_rhc.predict(self.testing_x)
            y_test_accuracy_rhc = accuracy_score(self.testing_y, y_test_pred_rhc)
            test_acc_rhc.append(y_test_accuracy_rhc)
            time_rhc.append(ft)
            print('RHC Completed!')

            nn_model_sa = self.sa(i)

            st = time.time()
            nn_model_sa.fit(self.training_x, self.training_y)
            end = time.time()
            ft = end - st

            # Predict labels for train set and assess accuracy
            training_y_pred_sa = nn_model_sa.predict(self.training_x)
            training_y_accuracy_sa = accuracy_score(self.training_y, training_y_pred_sa)
            train_acc_sa.append(training_y_accuracy_sa)

            # Predict labels for test set and assess accuracy
            y_test_pred_sa = nn_model_sa.predict(self.testing_x)
            y_test_accuracy_sa = accuracy_score(self.testing_y, y_test_pred_sa)
            test_acc_sa.append(y_test_accuracy_sa)
            time_sa.append(ft)
            print('SA completed!')

            nn_model_ga = self.ga(i)

            st = time.time()
            nn_model_ga.fit(self.training_x, self.training_y)
            end = time.time()
            ft = end - st

            # Predict labels for train set and assess accuracy
            training_y_pred_ga = nn_model_ga.predict(self.training_x)
            training_y_accuracy_ga = accuracy_score(self.training_y, training_y_pred_ga)
            train_acc_ga.append(training_y_accuracy_ga)

            # Predict labels for test set and assess accuracy
            y_test_pred_ga = nn_model_ga.predict(self.testing_x)
            y_test_accuracy_ga = accuracy_score(self.testing_y, y_test_pred_ga)
            test_acc_ga.append(y_test_accuracy_ga)
            time_ga.append(ft)
            print('GA completed!')

        plt.close()
        plt.figure()
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_rhc), label='RHC')
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_sa), label='SA')
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_ga), label='GA')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs. Iterations')
        plt.legend(loc="best")
        plt.savefig(os.path.join(self.output_path, 'NN_test_iterations.png'))

        plt.close()
        plt.figure()
        plt.plot(np.arange(1000, 50000, 2000), np.array(train_acc_rhc), label='RHC')
        plt.plot(np.arange(1000, 50000, 2000), np.array(train_acc_sa), label='SA')
        plt.plot(np.arange(1000, 50000, 2000), np.array(train_acc_ga), label='GA')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy vs. Iterations')
        plt.legend(loc="best")
        plt.savefig(os.path.join(self.output_path, 'NN_train_iterations.png'))

        plt.close()
        plt.figure()
        plt.plot(np.arange(1000, 50000, 2000), np.array(time_rhc), label='RHC')
        plt.plot(np.arange(1000, 50000, 2000), np.array(time_sa), label='SA')
        plt.plot(np.arange(1000, 50000, 2000), np.array(time_ga), label='GA')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Training Time')
        plt.title('Computation Time vs. Iterations')
        plt.legend(loc="best")
        plt.savefig(os.path.join(self.output_path, 'NN_time.png'))

        plt.close()
        plt.figure()
        plt.plot(np.arange(1000, 50000, 2000), np.array(time_rhc), label='RHC')
        plt.plot(np.arange(1000, 50000, 2000), np.array(time_sa), label='SA')
        plt.plot(np.arange(1000, 50000, 2000), np.array(time_ga), label='GA')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Training Time')
        plt.title('Computation Time vs. Iterations')
        plt.legend(loc="best")
        plt.savefig(os.path.join(self.output_path, 'NN_time.png'))

    def cv_analysis(self):

        test_acc_sa_1 = []
        for i in range(1000, 50000, 2000):
            print(i)
            nn_model_sa = self.sa(i, mlrose.GeomDecay())

            nn_model_sa.fit(self.training_x, self.training_y)
            # Predict labels for test set and assess accuracy
            y_test_pred_sa = nn_model_sa.predict(self.testing_x)
            y_test_accuracy_sa = accuracy_score(self.testing_y, y_test_pred_sa)
            test_acc_sa_1.append(y_test_accuracy_sa)

        test_acc_sa_2 = []
        for i in range(1000, 50000, 2000):
            nn_model_sa = self.sa(i, mlrose.ExpDecay())

            nn_model_sa.fit(self.training_x, self.training_y)

            # Predict labels for test set and assess accuracy
            y_test_pred_sa = nn_model_sa.predict(self.testing_x)
            y_test_accuracy_sa = accuracy_score(self.testing_y, y_test_pred_sa)
            test_acc_sa_2.append(y_test_accuracy_sa)

        test_acc_sa_3 = []
        for i in range(1000, 50000, 2000):
            nn_model_sa = self.sa(i, mlrose.ArithDecay())

            nn_model_sa.fit(self.training_x, self.training_y)

            # Predict labels for test set and assess accuracy
            y_test_pred_sa = nn_model_sa.predict(self.testing_x)
            y_test_accuracy_sa = accuracy_score(self.testing_y, y_test_pred_sa)
            test_acc_sa_3.append(y_test_accuracy_sa)

        plt.close()
        plt.figure()
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_sa_1), label='Geometric Decay')
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_sa_2), label='Exponential Decay')
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_sa_3), label='Arithmetic Decay')
        plt.title('Neural Network SA Analysis')
        plt.legend(loc="best")
        plt.xlabel('Number of Iterations')
        plt.ylabel('Testing Accuracy')
        plt.savefig(os.path.join(self.output_path, 'NN_SA_analysis.png'))

        test_acc_ga_1 = []
        for i in range(1000, 50000, 2000):
            print(i)
            nn_model_ga = self.ga(i, pop_size=100, mutation_prob=0.1)

            nn_model_ga.fit(self.training_x, self.training_y)

            # Predict labels for test set and assess accuracy
            y_test_pred_ga = nn_model_ga.predict(self.testing_x)
            y_test_accuracy_ga = accuracy_score(self.testing_y, y_test_pred_ga)
            test_acc_ga_1.append(y_test_accuracy_ga)

        test_acc_ga_2 = []
        for i in range(1000, 50000, 2000):
            nn_model_ga = self.ga(i, pop_size=200, mutation_prob=0.1)

            nn_model_ga.fit(self.training_x, self.training_y)

            # Predict labels for test set and assess accuracy
            y_test_pred_ga = nn_model_ga.predict(self.testing_x)
            y_test_accuracy_ga = accuracy_score(self.testing_y, y_test_pred_ga)
            test_acc_ga_2.append(y_test_accuracy_ga)

        test_acc_ga_3 = []
        for i in range(1000, 50000, 2000):
            nn_model_ga = self.ga(i, pop_size=500, mutation_prob=0.1)

            nn_model_ga.fit(self.training_x, self.training_y)

            # Predict labels for test set and assess accuracy
            y_test_pred_ga = nn_model_ga.predict(self.testing_x)
            y_test_accuracy_ga = accuracy_score(self.testing_y, y_test_pred_ga)
            test_acc_ga_3.append(y_test_accuracy_ga)

        test_acc_ga_4 = []
        for i in range(1000, 50000, 2000):
            nn_model_ga = self.ga(i, pop_size=100, mutation_prob=0.5)

            nn_model_ga.fit(self.training_x, self.training_y)

            # Predict labels for test set and assess accuracy
            y_test_pred_ga = nn_model_ga.predict(self.testing_x)
            y_test_accuracy_ga = accuracy_score(self.testing_y, y_test_pred_ga)
            test_acc_ga_4.append(y_test_accuracy_ga)

        test_acc_ga_5 = []
        for i in range(1000, 50000, 2000):
            nn_model_ga = self.ga(i, pop_size=200, mutation_prob=0.5)

            nn_model_ga.fit(self.training_x, self.training_y)

            # Predict labels for test set and assess accuracy
            y_test_pred_ga = nn_model_ga.predict(self.testing_x)
            y_test_accuracy_ga = accuracy_score(self.testing_y, y_test_pred_ga)
            test_acc_ga_5.append(y_test_accuracy_ga)

        test_acc_ga_6 = []
        for i in range(1000, 50000, 2000):
            nn_model_ga = self.ga(i, pop_size=500, mutation_prob=0.5)

            nn_model_ga.fit(self.training_x, self.training_y)

            # Predict labels for test set and assess accuracy
            y_test_pred_ga = nn_model_ga.predict(self.testing_x)
            y_test_accuracy_ga = accuracy_score(self.testing_y, y_test_pred_ga)
            test_acc_ga_6.append(y_test_accuracy_ga)

        plt.close()
        plt.figure()
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_ga_1), label='0.1 and 100')
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_ga_2), label='0.1 and 200')
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_ga_3), label='0.1 and 500')
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_ga_4), label='0.5 and 100')
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_ga_5), label='0.5 and 200')
        plt.plot(np.arange(1000, 50000, 2000), np.array(test_acc_ga_6), label='0.5 and 500')
        plt.title('Neural Network GA Analysis')
        plt.legend(loc="best")
        plt.xlabel('Number of Iterations')
        plt.ylabel('Testing Accuracy')
        plt.savefig(os.path.join(self.output_path, 'NN_GA_analysis.png'))
