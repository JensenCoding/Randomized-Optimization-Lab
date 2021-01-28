import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
from time import clock

from sklearn.utils import compute_sample_weight
from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import sklearn.model_selection as ms


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


# Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(c_matrix, y, title='Confusion matrix'):
    plt.close()
    plt.figure()
    plt.imshow(c_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(y))
    plt.xticks(tick_marks, y, rotation=45)
    plt.yticks(tick_marks, y)
    plt.tight_layout()

    for i, j in product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
        plt.text(j, i, c_matrix[i, j], horizontalalignment="center")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    return plt


# Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
def plot_roc_curve(fpr, tpr, auc, title='ROC Curve'):
    plt.close()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    return plt


def plot_time_curve(title, data_sizes, fit_time, predict_time, ylim=None):
    plt.close()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Data Size (% of total)")
    plt.ylabel("Time (s)")
    fit_time_mean = np.mean(fit_time, axis=1)
    fit_time_std = np.std(fit_time, axis=1)
    predict_time_mean = np.mean(predict_time, axis=1)
    predict_time_std = np.std(predict_time, axis=1)
    plt.grid()
    plt.tight_layout()

    plt.fill_between(data_sizes, fit_time_mean - fit_time_std, fit_time_mean + fit_time_std, alpha=0.2)
    plt.fill_between(data_sizes, predict_time_mean - predict_time_std, predict_time_mean + predict_time_std, alpha=0.2)
    plt.plot(data_sizes, fit_time_mean, 'o-', linewidth=1, markersize=4,
             label="Fit time")
    plt.plot(data_sizes, predict_time_mean, 'o-', linewidth=1, markersize=4,
             label="Predict time")

    plt.legend(loc="best")
    return plt


def gen_time_curve(data, clf, clf_name, scaled=True):
    data_name = data.data_name
    output_path = data.output_path
    if scaled:
        X, Y, _, _, _, _ = data.get_scaled_data()
    else:
        X, Y = data.get_data()

    fracs = np.linspace(0.1, 0.9, num=9)
    res = dict()
    res['train'] = np.zeros((len(fracs), 5))
    res['test'] = np.zeros((len(fracs), 5))
    for i, frac in enumerate(fracs):
        for j in range(5):
            np.random.seed(data.seed)
            x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=1 - frac, random_state=data.seed)
            st = clock()
            clf.fit(x_train, y_train)
            res['train'][i, j] = (clock() - st)
            st = clock()
            clf.predict(x_test)
            res['test'][i, j] = (clock() - st)

    train_df = pd.DataFrame(res['train'], index=fracs)
    test_df = pd.DataFrame(res['test'], index=fracs)
    this_plt = plot_time_curve('{} - {}'.format(data_name, clf_name), np.array(fracs) * 100, train_df, test_df)
    this_plt.savefig(os.path.join(output_path, '{}_{}_time_curve.png'.format(data_name, clf_name)), format='png', dpi=150)

    res = pd.DataFrame(index=fracs)
    res['train'] = np.mean(train_df, axis=1)
    res['test'] = np.mean(test_df, axis=1)
    res.to_csv(os.path.join(output_path, '{}_{}_timing.csv'.format(data_name, clf_name)))


# Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
def plot_validation_curve(title, v_param, v_value, train_scores, test_scores):

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(v_param)
    plt.ylabel("Score")
    plt.xscale("linear")
    plt.semilogx(v_value, train_scores_mean, label="Training score")
    plt.fill_between(v_value, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
    plt.semilogx(v_value, test_scores_mean, label="Cross-validation score")
    plt.fill_between(v_value, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2)
    plt.legend(loc="best")

    return plt


def get_score(y_true, y_pred):
    weights = compute_sample_weight('balanced', y_true)
    return f1_score(y_true, y_pred, average='binary', sample_weight=weights)


def cross_validate(data, clf, clf_name, params, scaled=True, best_params=None, validation_params=None):
    data_name = data.data_name
    output_path = data.output_path
    if scaled:
        X, Y, training_x, training_y, testing_x, testing_y = data.get_scaled_data()
    else:
        training_x, training_y, testing_x, testing_y = data.get_split_data()
        _, Y = data.get_data()
    np.random.seed(data.seed)

    scorer = make_scorer(get_score)
    if best_params:
        clf.set_params(**best_params)
        clf.fit(training_x, training_y)
        train_score = clf.score(testing_x, testing_y)
        test_score = clf.score(testing_x, testing_y)
        pred_y = clf.predict(testing_x)
        cv = clf
    else:
        cv = ms.GridSearchCV(clf, param_grid=params, refit=True, verbose=10, scoring=scorer)
        cv.fit(training_x, training_y)
        train_score = cv.score(training_x, training_y)
        test_score = cv.score(testing_x, testing_y)

        # best parameters from the cross validation
        best_clf = cv.best_estimator_
        best_clf.fit(training_x, training_y)
        cv_best_params = best_clf.get_params()
        pred_y = best_clf.predict(testing_x)

        # print the cross validation results
        pd.DataFrame([cv_best_params])\
            .to_csv(os.path.join(output_path, '{}_{}_best_parameter.csv'.format(data_name, clf_name)), index=False)
        pd.DataFrame(cv.cv_results_) \
            .to_csv(os.path.join(output_path, '{}_{}_cv_result.csv'.format(data_name, clf_name)), index=False)

    with open(os.path.join(output_path, 'test_result.csv'), 'a') as f:
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        f.write('{},{},{},{},{},{}\n'.format(now_, data_name, clf_name, train_score, test_score, cv_best_params))

    # draw normalized confusion matrix
    c_matrix = confusion_matrix(pred_y, testing_y)
    c_matrix = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]
    c_plt = plot_confusion_matrix(c_matrix, np.unique(Y),  title='Confusion Matrix: {}, {}'.format(data_name, clf_name))
    c_plt.savefig(os.path.join(output_path, '{}_{}_c_matrix.png'.format(data_name, clf_name)), dpi=150, bbox_inches='tight')

    # draw ROC curve
    fpr, tpr, _ = roc_curve(pred_y, testing_y)
    roc_auc = auc(fpr, tpr)
    roc_plot = plot_roc_curve(fpr, tpr, roc_auc, title='ROC Curve: {}, {}'.format(data_name, clf_name))
    roc_plot.savefig(os.path.join(output_path, '{}_{}_roc_curve.png'.format(data_name, clf_name)), dpi=150, bbox_inches='tight')

    # draw learning curve
    train_sizes, train_scores, test_scores = ms.learning_curve(
        clf if best_params is not None else cv.best_estimator_,
        training_x,
        training_y,
        cv=5,
        train_sizes=np.linspace(0.1, 1, 20),
        scoring=scorer,
        random_state=data.seed)

    lc_plt = plot_learning_curve('Learning Curve: {}, {}'.format(data_name, clf_name), train_sizes, train_scores, test_scores)
    lc_plt.savefig(os.path.join(output_path, '{}_{}_learning_curve.png'.format(data_name, clf_name)), dpi=150)

    # draw validation curve
    for v_param, v_value in validation_params.items():
        train_scores, test_scores = ms.validation_curve(
            clf if best_params is not None else cv.best_estimator_,
            training_x, training_y, cv=5, param_name=v_param, param_range=v_value, scoring=scorer)

        validate_plt = plot_validation_curve('Validation Curve: {}, {} ({})'.format(data_name, clf_name, v_param), v_param, v_value, train_scores, test_scores)
        validate_plt.savefig(os.path.join(output_path, '{}_{}_{}_validation_curve.png'.format(data_name, clf_name, v_param)), dpi=150, bbox_inches='tight')

    return cv


def iter_learn_curve(data, clf, clf_name, lc_params, scaled=True):

    data_name = data.data_name
    output_path = data.output_path
    np.random.seed(data.seed)
    if scaled:
        X, Y, training_x, training_y, testing_x, testing_y = data.get_scaled_data()
    else:
        training_x, training_y, testing_x, testing_y = data.get_split_data()
        _, Y = data.get_data()

    res = defaultdict(list)
    param_name = list(lc_params.keys())[0]
    for value in list(lc_params.values())[0]:
        res['param_{}'.format(param_name)].append(value)
        clf.set_params(**{param_name: value})
        clf.fit(training_x, training_y)
        pred_y = clf.predict(training_x)
        res['train acc'].append(get_score(training_y, pred_y))
        pred_y = clf.predict(testing_x)
        res['test acc'].append(get_score(testing_y, pred_y))

    res = pd.DataFrame(res)
    res.to_csv(os.path.join(output_path, '{}_{}_iter_result.csv'.format(data_name, clf_name)), index=False)
    this_plt = plot_learning_curve('Learning Curve: {}, {}'.format(data_name, clf_name, ),
                                   res['param_{}'.format(param_name)], res['train acc'], res['test acc'],
                                   multi_run=False, x_label=param_name, x_scale='log')
    this_plt.savefig(os.path.join(output_path, '{}_{}_iter_learning_curve.png'.format(data_name, clf_name)), dpi=150)


def exp_perform(data, clf, clf_name, params, scaled=True, best_params=None, timing_params=None, validation_params=None, lc_params=None):

    print("performing cross validation....")
    ds_clf = cross_validate(data, clf, clf_name, params, scaled=scaled, best_params=best_params, validation_params=validation_params)

    if best_params is not None:
        final_params = best_params
    else:
        final_params = ds_clf.best_params_
        clf.set_params(**final_params)

    if timing_params is not None:
        clf.set_params(**timing_params)
    print("generating time curve....")
    gen_time_curve(data, clf, clf_name, scaled=scaled)

    if lc_params is not None:
        print("generating iteration learning curve....")
        iter_learn_curve(data, clf, clf_name, lc_params)

    return final_params
