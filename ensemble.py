from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from math import exp
from numpy.random import uniform, random_integers, dirichlet
from random import sample
from sklearn import datasets
from numpy import zeros, round, mean, ones, vstack

names = ["Logistic Regression", "Random Forest", "Linear SVM", "Nearest Neighbors", "RBF SVM", "Decision Tree", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis", "Quadratic Discriminant Analysis"]

### This framework uses hierarchial Tree objects for book-keeping.  In the future it should probably be re-written using TreeDict https://github.com/hoytak/treedict

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None


def initialize_hp():

    """
    creates the default hyperparameter tree
    :return: hp:  the hyperparameter tree
    """

    hp = Tree()   # hp is the hyper parameter tree

    hp.n_deltas = 3

    hp.LogisticRegression = Tree()
    hp.RandomForestClassifier = Tree()
    hp.LinearSVC = Tree()
    hp.KNeighborsClassifier = Tree()
    hp.SVC = Tree()
    hp.DecisionTreeClassifier = Tree()
    hp.AdaBoostClassifier = Tree()
    hp.LinearDiscriminantAnalysis = Tree()
    hp.QuadraticDiscriminantAnalysis = Tree()

    hp.LogisticRegression.C = Tree()
    hp.RandomForestClassifier.n_estimators = Tree()
    hp.LinearSVC.C = Tree()
    hp.KNeighborsClassifier.n_neighbors = Tree()
    hp.SVC.C = Tree()
    hp.DecisionTreeClassifier.max_depth = Tree()
    hp.AdaBoostClassifier.n_estimators = Tree()
    hp.LinearDiscriminantAnalysis.priors = Tree()
    hp.QuadraticDiscriminantAnalysis.priors = Tree()

    hp.LogisticRegression.C.value = 1
    hp.RandomForestClassifier.n_estimators.value = 10
    hp.RandomForestClassifier.n_estimators.floor = 1
    hp.LinearSVC.C.value = 1
    hp.KNeighborsClassifier.n_neighbors.value = 3
    hp.KNeighborsClassifier.n_neighbors.floor = 1
    hp.SVC.C.value = 1
    hp.DecisionTreeClassifier.max_depth.value = 5
    hp.DecisionTreeClassifier.max_depth.floor = 1
    hp.AdaBoostClassifier.n_estimators.value = 50
    hp.AdaBoostClassifier.n_estimators.floor = 1
    hp.LinearDiscriminantAnalysis.priors.value = (0.5, 0.5)
    hp.QuadraticDiscriminantAnalysis.priors.value = (0.5, 0.5)

    hp.LogisticRegression.C.delta = True
    hp.RandomForestClassifier.n_estimators.delta = True
    hp.LinearSVC.C.delta = True
    hp.KNeighborsClassifier.n_neighbors.delta = True
    hp.SVC.C.delta = True
    hp.DecisionTreeClassifier.max_depth.delta = True
    hp.AdaBoostClassifier.n_estimators.delta = True
    hp.LinearDiscriminantAnalysis.priors.delta = True
    hp.QuadraticDiscriminantAnalysis.priors.delta = True

    hp.LogisticRegression.C.search_distance = 1
    hp.RandomForestClassifier.n_estimators.search_distance = 1
    hp.LinearSVC.C.search_distance = 1
    hp.KNeighborsClassifier.n_neighbors.search_distance = 1
    hp.SVC.C.search_distance = 1
    hp.DecisionTreeClassifier.max_depth.search_distance = 1
    hp.AdaBoostClassifier.n_estimators.search_distance = 1
    hp.LinearDiscriminantAnalysis.priors.search_distance = 1
    hp.QuadraticDiscriminantAnalysis.priors.search_distance = 1

    hp.LogisticRegression.enable = True
    hp.RandomForestClassifier.enable = True
    hp.LinearSVC.enable = True
    hp.KNeighborsClassifier.enable = True
    hp.SVC.enable = True
    hp.DecisionTreeClassifier.enable = True
    hp.AdaBoostClassifier.enable = True
    hp.LinearDiscriminantAnalysis.enable = True
    hp.QuadraticDiscriminantAnalysis.enable = True

    hp.num_models = hp.LogisticRegression.enable + hp.RandomForestClassifier.enable + hp.LinearSVC.enable + hp.KNeighborsClassifier.enable + hp.SVC.enable + hp.DecisionTreeClassifier.enable + hp.AdaBoostClassifier.enable + hp.LinearDiscriminantAnalysis.enable + hp.QuadraticDiscriminantAnalysis.enable

    return hp


def neighborhood(hp):

    """
    searches the neighborhood of the hyperparameter space and returns a permutations
    :param hp: the hyperparameter tree
    :return: hp:  permuted set of hyperparamter values
    """

    def neighborhood_continuous(value,search_distance):
        value *= exp(uniform(low=-search_distance, high=search_distance))
        return value

    def neighborhood_discrete(value,search_distance,floor):
        proposed_change = random_integers(low=-search_distance, high=search_distance)
        while value + proposed_change < floor:
            proposed_change = random_integers(low=-search_distance, high=search_distance)
        value += proposed_change
        return value

    def neighborhood_priors(value,search_distance):
        proposed_change = uniform(low=-search_distance*0.01, high=search_distance*0.01)
        while (value[0] + proposed_change) > 1 or (value[1] - proposed_change) > 1 or (value[0] + proposed_change) < 0 or (value[1] - proposed_change) < 0:
            proposed_change = uniform(low=-search_distance*0.01, high=search_distance*0.01)
        new_prior_change = (+proposed_change,-proposed_change)
        value = tuple(map(sum, zip(value, new_prior_change)))
        return value

    choose_list = sample(xrange(9), hp.n_deltas)  # this needs to be fixed to be general if some models are not enabled

    if 0 in choose_list:
        hp.LogisticRegression.C.delta = True
    else:
        hp.LogisticRegression.C.delta = False

    if 1 in choose_list:
        hp.RandomForestClassifier.n_estimators.delta = True
    else:
        hp.RandomForestClassifier.n_estimators.delta = False

    if 2 in choose_list:
        hp.LinearSVC.C.delta = True
    else:
        hp.LinearSVC.C.delta = False

    if 3 in choose_list:
        hp.KNeighborsClassifier.n_neighbors.delta = True
    else:
        hp.KNeighborsClassifier.n_neighbors.delta = False

    if 4 in choose_list:
        hp.SVC.C.delta = True
    else:
        hp.SVC.C.delta = False

    if 5 in choose_list:
        hp.DecisionTreeClassifier.max_depth.delta = True
    else:
        hp.DecisionTreeClassifier.max_depth.delta = False

    if 6 in choose_list:
        hp.AdaBoostClassifier.n_estimators.delta = True
    else:
        hp.AdaBoostClassifier.n_estimators.delta = False

    if 7 in choose_list:
        hp.LinearDiscriminantAnalysis.priors.delta = True
    else:
        hp.LinearDiscriminantAnalysis.priors.delta = False

    if 8 in choose_list:
        hp.QuadraticDiscriminantAnalysis.priors.delta = True
    else:
        hp.QuadraticDiscriminantAnalysis.priors.delta = False

    if hp.LogisticRegression.C.delta:
        hp.LogisticRegression.C.value = neighborhood_continuous(value=hp.LogisticRegression.C.value, search_distance=hp.LogisticRegression.C.search_distance)

    if hp.RandomForestClassifier.n_estimators.delta:
        hp.RandomForestClassifier.n_estimators.value = neighborhood_discrete(value=hp.RandomForestClassifier.n_estimators.value, search_distance=hp.RandomForestClassifier.n_estimators.search_distance, floor=hp.RandomForestClassifier.n_estimators.floor)

    if hp.LinearSVC.C.delta:
        hp.LinearSVC.C.value = neighborhood_continuous(value=hp.LinearSVC.C.value, search_distance=hp.LinearSVC.C.search_distance)

    if hp.KNeighborsClassifier.n_neighbors.delta:
        hp.KNeighborsClassifier.n_neighbors.value = neighborhood_discrete(value=hp.KNeighborsClassifier.n_neighbors.value, search_distance=hp.KNeighborsClassifier.n_neighbors.search_distance, floor=hp.KNeighborsClassifier.n_neighbors.floor)

    if hp.SVC.C.delta:
        hp.SVC.C.value = neighborhood_continuous(value=hp.SVC.C.value, search_distance=hp.SVC.C.search_distance)

    if hp.DecisionTreeClassifier.max_depth.delta:
        hp.DecisionTreeClassifier.max_depth.value = neighborhood_discrete(value=hp.DecisionTreeClassifier.max_depth.value, search_distance=hp.DecisionTreeClassifier.max_depth.search_distance, floor=hp.DecisionTreeClassifier.max_depth.floor)

    if hp.AdaBoostClassifier.n_estimators.delta:
        hp.AdaBoostClassifier.n_estimators.value = neighborhood_discrete(value=hp.AdaBoostClassifier.n_estimators.value, search_distance=hp.AdaBoostClassifier.n_estimators.search_distance, floor=hp.AdaBoostClassifier.n_estimators.floor)

    if hp.LinearDiscriminantAnalysis.priors.delta:
        hp.LinearDiscriminantAnalysis.priors.value = neighborhood_priors(value=hp.LinearDiscriminantAnalysis.priors.value, search_distance=hp.LinearDiscriminantAnalysis.priors.search_distance)

    if hp.QuadraticDiscriminantAnalysis.priors.delta:
        hp.QuadraticDiscriminantAnalysis.priors.value = neighborhood_priors(value=hp.QuadraticDiscriminantAnalysis.priors.value, search_distance=hp.QuadraticDiscriminantAnalysis.priors.search_distance)

    return hp


def initialize_dt(X, y):
    """
    splits a given data set into train, validate, and test and returns the data tree
    :param X: features
    :param y: classes
    :return: dt: the data tree
    """

    n = X.shape[0]

    train_n = round(n/2)
    val_n = round(n/4)
    test_n = round(n/4)

    while train_n + val_n + test_n != n:
        if train_n + val_n + test_n < n:
            train_n += 1
        elif train_n + val_n + test_n > n:
            train_n -= 1

    X_train, y_train = X[:train_n, :], y[:train_n]
    X_val, y_val = X[train_n:(train_n+val_n), :], y[train_n:(train_n+val_n)]
    X_test, y_test = X[(train_n+val_n):(train_n+val_n+test_n), :], y[(train_n+val_n):(train_n+val_n+test_n)]

    dt = Tree()

    dt.X_train = X_train
    dt.y_train = y_train
    dt.X_val = X_val
    dt.y_val = y_val
    dt.X_test = X_test
    dt.y_test = y_test

    return dt


def results(clf, dt):
    """
    Runs a classification model and returns a results tree
    :param clf: an sklearn classification model object
    :param X_train: training features
    :param y_train: training classes
    :param X_val: validation features
    :param y_val: validation classes
    :param X_test: test features
    :param y_test: test classes
    :return: results tree branch for the classification model
    """
    X_train = dt.X_train
    y_train = dt.y_train
    X_val = dt.X_val
    y_val = dt.y_val
    X_test = dt.X_test
    y_test = dt.y_test

    rs_clf = Tree()
    clf.fit(X_train, y_train)
    rs_clf.train_pred = clf.predict(X_train)
    rs_clf.val_pred = clf.predict(X_val)
    rs_clf.test_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        rs_clf.train_prob = clf.predict_proba(X_train)[:, 1]
        rs_clf.val_prob = clf.predict_proba(X_val)[:, 1]
        rs_clf.test_prob = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_train)
        rs_clf.train_prob = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        prob_pos = clf.decision_function(X_val)
        rs_clf.val_prob = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        prob_pos = clf.decision_function(X_test)
        rs_clf.test_prob = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    rs_clf.train_correct = 1 - abs(y_train - rs_clf.train_pred)
    rs_clf.val_correct = 1 - abs(y_val - rs_clf.val_pred)
    rs_clf.test_correct = 1 - abs(y_test - rs_clf.test_pred)
    return rs_clf


def fit_models(hp, dt):
    """
    Takes hyperparameter settings and a data set and fits up to nine classification models
    :param hp: The hyperparameter tree
    :param dt: The data tree
    :return: rs:  The results tree
    """

    rs = Tree()
    rs.y_train, rs.y_val, rs.y_test = dt.y_train, dt.y_val, dt.y_test

    if hp.LogisticRegression.enable:
        clf = LogisticRegression(C=hp.LogisticRegression.C.value)
        rs.LogisticRegression = results(clf, dt)
        rs.LogisticRegression.C = hp.LogisticRegression.C.value
        rs.LogisticRegression.enable = True

    if hp.RandomForestClassifier.enable:
        clf = RandomForestClassifier(n_estimators=hp.RandomForestClassifier.n_estimators.value)
        rs.RandomForestClassifier = results(clf, dt)
        rs.RandomForestClassifier.n_estimators = hp.RandomForestClassifier.n_estimators.value
        rs.RandomForestClassifier.enable = True

    if hp.LinearSVC.enable:
        clf = LinearSVC(C=hp.LinearSVC.C.value)
        rs.LinearSVC = results(clf, dt)
        rs.LinearSVC.C = hp.LinearSVC.C.value
        rs.LinearSVC.enable = True

    if hp.KNeighborsClassifier.enable:
        clf = KNeighborsClassifier(n_neighbors=hp.KNeighborsClassifier.n_neighbors.value)
        rs.KNeighborsClassifier = results(clf, dt)
        rs.KNeighborsClassifier.n_neighbors = hp.KNeighborsClassifier.n_neighbors.value
        rs.KNeighborsClassifier.enable = True

    if hp.SVC.enable:
        clf = SVC(C=hp.SVC.C.value, kernel='rbf')
        rs.SVC = results(clf, dt)
        rs.SVC.C = hp.SVC.C.value
        rs.SVC.enable = True

    if hp.DecisionTreeClassifier.enable:
        clf = DecisionTreeClassifier(max_depth=hp.DecisionTreeClassifier.max_depth.value)
        rs.DecisionTreeClassifier = results(clf, dt)
        rs.DecisionTreeClassifier.max_depth = hp.DecisionTreeClassifier.max_depth.value
        rs.DecisionTreeClassifier.enable = True

    if hp.AdaBoostClassifier.enable:
        clf = AdaBoostClassifier(n_estimators=hp.AdaBoostClassifier.n_estimators.value)
        rs.AdaBoostClassifier = results(clf, dt)
        rs.AdaBoostClassifier.n_estimators = hp.AdaBoostClassifier.n_estimators.value
        rs.AdaBoostClassifier.enable = True

    if hp.LinearDiscriminantAnalysis.enable:
        clf = LinearDiscriminantAnalysis(priors=hp.LinearDiscriminantAnalysis.priors.value)
        rs.LinearDiscriminantAnalysis = results(clf, dt)
        rs.LinearDiscriminantAnalysis.priors = hp.LinearDiscriminantAnalysis.priors.value
        rs.LinearDiscriminantAnalysis.enable = True

    if hp.QuadraticDiscriminantAnalysis.enable:
        clf = QuadraticDiscriminantAnalysis(priors=hp.QuadraticDiscriminantAnalysis.priors.value)
        rs.QuadraticDiscriminantAnalysis = results(clf, dt)
        rs.QuadraticDiscriminantAnalysis.priors = hp.QuadraticDiscriminantAnalysis.priors.value
        rs.QuadraticDiscriminantAnalysis.enable = True

    return rs


def build_ensemble(rs, weights):
    """
    builds an ensemble of up to 9 classifiers, and runs a meta-classification
    :param rs: the results tree
    :param weights: a set of normalized weights for each of the classifiers
    :return: en:  the ensemble tree
    """


    en = Tree()
    en.votes_train = zeros(len(rs.y_train))
    en.votes_val = zeros(len(rs.y_val))
    en.votes_test = zeros(len(rs.y_test))
    en.prob_train = zeros(len(rs.y_train))
    en.prob_val = zeros(len(rs.y_val))
    en.prob_test = zeros(len(rs.y_test))
    num_models = rs.LogisticRegression.enable + rs.RandomForestClassifier.enable + rs.LinearSVC.enable + rs.KNeighborsClassifier.enable + rs.SVC.enable + rs.DecisionTreeClassifier.enable + rs.AdaBoostClassifier.enable + rs.LinearDiscriminantAnalysis.enable + rs.QuadraticDiscriminantAnalysis.enable

    def update_ensemble(en, rs_clf, weight):
        en.votes_train += rs_clf.train_pred
        en.votes_val += rs_clf.val_pred
        en.votes_test += rs_clf.test_pred
        en.prob_train += rs_clf.train_prob * weight
        en.prob_val += rs_clf.val_prob * weight
        en.prob_test += rs_clf.test_prob * weight
        return en

    i = 0

    if rs.LogisticRegression.enable:
        en = update_ensemble(en, rs.LogisticRegression, weight=weights[i])
        i += 1

    if rs.RandomForestClassifier.enable:
        en = update_ensemble(en, rs.RandomForestClassifier, weight=weights[i])
        i += 1

    if rs.LinearSVC.enable:
        en = update_ensemble(en, rs.LinearSVC, weight=weights[i])
        i += 1

    if rs.KNeighborsClassifier.enable:
        en = update_ensemble(en, rs.KNeighborsClassifier, weight=weights[i])
        i += 1

    if rs.SVC.enable:
        en = update_ensemble(en, rs.SVC, weight=weights[i])
        i += 1

    if rs.DecisionTreeClassifier.enable:
        en = update_ensemble(en, rs.DecisionTreeClassifier, weight=weights[i])
        i += 1

    if rs.AdaBoostClassifier.enable:
        en = update_ensemble(en, rs.AdaBoostClassifier, weight=weights[i])
        i += 1

    if rs.LinearDiscriminantAnalysis.enable:
        en = update_ensemble(en, rs.LinearDiscriminantAnalysis, weight=weights[i])
        i += 1

    if rs.QuadraticDiscriminantAnalysis.enable:
        en = update_ensemble(en, rs.QuadraticDiscriminantAnalysis, weight=weights[i])
        i += 1

    en.prob_train_correct = 1 - abs(rs.y_train - round(en.prob_train))
    en.prob_val_correct = 1 - abs(rs.y_val - round(en.prob_val))
    en.prob_test_correct = 1 - abs(rs.y_test - round(en.prob_test))

    en.votes_train_correct = 1 - abs(rs.y_train - round(en.votes_train/num_models))
    en.votes_val_correct = 1 - abs(rs.y_val - round(en.votes_val/num_models))
    en.votes_test_correct = 1 - abs(rs.y_test - round(en.votes_test/num_models))

    en.LogisticRegression = Tree()
    en.RandomForestClassifier = Tree()
    en.LinearSVC = Tree()
    en.KNeighborsClassifier = Tree()
    en.SVC = Tree()
    en.DecisionTreeClassifier = Tree()
    en.AdaBoostClassifier = Tree()
    en.LinearDiscriminantAnalysis = Tree()
    en.QuadraticDiscriminantAnalysis = Tree()

    i = 0
    if rs.LogisticRegression.enable:
        en.LogisticRegression.enable = True
        en.LogisticRegression.weight = weights[i]
        en.LogisticRegression.C = rs.LogisticRegression.C
        i += 1
    else:
        en.LogisticRegression.enable = False

    if rs.RandomForestClassifier.enable:
        en.RandomForestClassifier.enable = True
        en.RandomForestClassifier.weight = weights[i]
        en.RandomForestClassifier.n_estimators = rs.RandomForestClassifier.n_estimators
        i += 1
    else:
        en.RandomForestClassifier.enable = False

    if rs.LinearSVC.enable:
        en.LinearSVC.enable = True
        en.LinearSVC.weight = weights[i]
        en.LinearSVC.C = rs.LinearSVC.C
        i += 1
    else:
        en.LinearSVC.enable = False

    if rs.KNeighborsClassifier.enable:
        en.KNeighborsClassifier.enable = True
        en.KNeighborsClassifier.weight = weights[i]
        en.KNeighborsClassifier.n_neighbors = rs.KNeighborsClassifier.n_neighbors
        i += 1
    else:
        en.KNeighborsClassifier.enable = False

    if rs.SVC.enable:
        en.SVC.enable = True
        en.SVC.weight = weights[i]
        en.SVC.C = rs.SVC.C
        i += 1
    else:
        en.SVC.enable = False

    if rs.DecisionTreeClassifier.enable:
        en.DecisionTreeClassifier.enable = True
        en.DecisionTreeClassifier.weight = weights[i]
        en.DecisionTreeClassifier.max_depth = rs.DecisionTreeClassifier.max_depth
        i += 1
    else:
        en.DecisionTreeClassifier.enable = False

    if rs.AdaBoostClassifier.enable:
        en.AdaBoostClassifier.enable = True
        en.AdaBoostClassifier.weight = weights[i]
        en.AdaBoostClassifier.n_estimators = rs.AdaBoostClassifier.n_estimators
        i += 1
    else:
        en.AdaBoostClassifier.enable = False

    if rs.LinearDiscriminantAnalysis.enable:
        en.LinearDiscriminantAnalysis.enable = True
        en.LinearDiscriminantAnalysis.weight = weights[i]
        en.LinearDiscriminantAnalysis.priors = rs.LinearDiscriminantAnalysis.priors
        i += 1
    else:
        en.LinearDiscriminantAnalysis.enable = False

    if rs.QuadraticDiscriminantAnalysis.enable:
        en.QuadraticDiscriminantAnalysis.enable = True
        en.QuadraticDiscriminantAnalysis.weight = weights[i]
        en.QuadraticDiscriminantAnalysis.priors = rs.QuadraticDiscriminantAnalysis.priors
        i += 1
    else:
        en.QuadraticDiscriminantAnalysis.enable = False

    return en


def random_weights(en):
    """
    creates a set of random weights
    :param en: the ensebmle tree
    :return: a random vector sampled from a diriclet distribution
    """

    num_models = en.LogisticRegression.enable + en.RandomForestClassifier.enable + en.LinearSVC.enable + en.KNeighborsClassifier.enable + en.SVC.enable + en.DecisionTreeClassifier.enable + en.AdaBoostClassifier.enable + en.LinearDiscriminantAnalysis.enable + en.QuadraticDiscriminantAnalysis.enable
    weights = dirichlet(ones(num_models), 1)[0]

    return weights


def neighborhood_weights(en):
    """
    searches the neighborhood of current weights and find new ones
    :param en:  the ensemble tree
    :return: a new set of feasible weights near the previous one
    """

    num_models = en.LogisticRegression.enable + en.RandomForestClassifier.enable + en.LinearSVC.enable + en.KNeighborsClassifier.enable + en.SVC.enable + en.DecisionTreeClassifier.enable + en.AdaBoostClassifier.enable + en.LinearDiscriminantAnalysis.enable + en.QuadraticDiscriminantAnalysis.enable
    weights = zeros(num_models)
    i = 0
    if en.LogisticRegression.enable:
        while weights[i] <= 0:
            weights[i] = en.LogisticRegression.weight + uniform(-0.1, 0.1)
        i += 1

    if en.RandomForestClassifier.enable:
        while weights[i] <= 0:
            weights[i] = en.RandomForestClassifier.weight + uniform(-0.1, 0.1)
        i += 1

    if en.LinearSVC.enable:
        while weights[i] <= 0:
            weights[i] = en.LinearSVC.weight + uniform(-0.1, 0.1)
        i += 1

    if en.KNeighborsClassifier.enable:
        while weights[i] <= 0:
            weights[i] = en.KNeighborsClassifier.weight + uniform(-0.1, 0.1)
        i += 1

    if en.SVC.enable:
        while weights[i] <= 0:
            weights[i] = en.SVC.weight + uniform(-0.1, 0.1)
        i += 1

    if en.DecisionTreeClassifier.enable:
        while weights[i] <= 0:
            weights[i] = en.DecisionTreeClassifier.weight + uniform(-0.1, 0.1)
        i += 1

    if en.AdaBoostClassifier.enable:
        while weights[i] <= 0:
            weights[i] = en.AdaBoostClassifier.weight + uniform(-0.1, 0.1)
        i += 1

    if en.LinearDiscriminantAnalysis.enable:
        while weights[i] <= 0:
            weights[i] = en.LinearDiscriminantAnalysis.weight + uniform(-0.1, 0.1)
        i += 1

    if en.QuadraticDiscriminantAnalysis.enable:
        while weights[i] <= 0:
            weights[i] = en.QuadraticDiscriminantAnalysis.weight + uniform(-0.1, 0.1)
        i += 1

    weights = weights/sum(weights)

    return weights


def hp_search(hp_init, rs_init, en_init, dt, weights, K=30, max_iter=1000):
    """
    performs a simmulated annealing search with linear cooling to attempt to optimize hyperparameter settings
    :param hp_init: initial state of the hp tree
    :param rs_init: initial state of the rs tree
    :param en_init: initial state of the en tree
    :param dt: the data tree
    :param weights:  current weights for the ensemble
    :param K:  cooling parameter for simmulated annealing
    :param max_iter:  max iterations for simmulated annealing
    :return:  the new hp, rs, and en trees ans a tracker of train, validate, and test performance by iteration
    """
    hp_current = hp_init
    rs_current = rs_init
    en_current = en_init

    hp_best = hp_init
    rs_best = rs_init
    en_best = en_init

    tracker = (mean(en_init.prob_train_correct), mean(en_init.prob_val_correct), mean(en_init.prob_test_correct))

    i = 0
    i_moves = 0
    p_moves = 0
    n_moves = 0
    print "Hyperparameter Search"
    while i < max_iter:
        # print i
        hp_candidate = neighborhood(hp_current)
        rs_candidate = fit_models(hp_candidate, dt)
        # weights = neighborhood_weights(en_current)
        en_candidate = build_ensemble(rs_candidate, weights)
        temperature = (float(max_iter - i) / max_iter) / K
        print temperature
        acceptance_probability = min(
            exp(- (mean(en_current.prob_val_correct) - mean(en_candidate.prob_val_correct)) / temperature), 1)
        print "Acceptance Probability %.4f" % acceptance_probability
        if mean(en_candidate.prob_val_correct) > mean(en_current.prob_val_correct):
            hp_current = hp_candidate
            rs_current = rs_candidate
            en_current = en_candidate
            i_moves += 1
            print "%d improvement moves" % i_moves

        elif exp(- (mean(en_current.prob_val_correct) - mean(en_candidate.prob_val_correct)) / temperature) > uniform(0,
                                                                                                                      1):
            hp_current = hp_candidate
            rs_current = rs_candidate
            en_current = en_candidate
            p_moves += 1
            print "%d positioning moves" % p_moves

        else:
            n_moves += 1
            print "%d no moves" % n_moves

        if mean(en_candidate.prob_val_correct) > mean(en_best.prob_val_correct):
            hp_best = hp_candidate
            rs_best = rs_candidate
            en_best = en_candidate

        tracker = vstack((tracker, (
        mean(en_current.prob_train_correct), mean(en_current.prob_val_correct), mean(en_current.prob_test_correct))))
        i += 1

    output_list = (tracker, hp_best, rs_best, en_best)

    return output_list


def greedy_hp(hp, dt):
    """
    A greedy construction for all hyperparameters performed using a simple grid search
    :param hp:  the current hp tree
    :param dt:  the data tree
    :return: a new hp tree fit to maximize validation set performance
    """

    if hp.LogisticRegression.enable:

        C_ln = -5
        C_ln_max = 5
        step = 0.1

        def val_LogisticRegression(Cval):
            clf = LogisticRegression(C=Cval)
            rs_LogisticRegression = results(clf, dt)
            val_score = mean(rs_LogisticRegression.val_correct)
            return val_score

        val_score_best = val_LogisticRegression(Cval=exp(C_ln))
        C_ln_best = C_ln

        while C_ln < C_ln_max:
            C_ln += step
            val_score_test = val_LogisticRegression(Cval=exp(C_ln))
            if val_score_test > val_score_best:
                val_score_best = val_score_test
                C_ln_best = C_ln

        hp.LogisticRegression.C.value = exp(C_ln_best)

    if hp.RandomForestClassifier.enable:

        n_estimators = 1
        n_estimators_max = 30
        step = 1

        def val_RandomForestClassifier(n_estimators_val):
            clf = RandomForestClassifier(n_estimators=n_estimators_val)
            rs_RandomForestClassifier = results(clf, dt)
            val_score = mean(rs_RandomForestClassifier.val_correct)
            return val_score

        val_score_best = val_RandomForestClassifier(n_estimators)
        n_estimators_best = n_estimators

        while n_estimators < n_estimators_max:
            n_estimators += step
            val_score_test = val_RandomForestClassifier(n_estimators)
            if val_score_test > val_score_best:
                val_score_best = val_score_test
                n_estimators_best = n_estimators

        hp.RandomForestClassifier.n_estimators.value = n_estimators_best

    if hp.LinearSVC.enable:

        C_ln = -5
        C_ln_max = 5
        step = 0.1

        def val_LinearSVC(Cval):
            clf = LinearSVC(C=Cval)
            rs_LinearSVC = results(clf, dt)
            val_score = mean(rs_LinearSVC.val_correct)
            return val_score

        val_score_best = val_LinearSVC(Cval=exp(C_ln))
        C_ln_best = C_ln

        while C_ln < C_ln_max:
            C_ln += step
            val_score_test = val_LinearSVC(Cval=exp(C_ln))
            if val_score_test > val_score_best:
                val_score_best = val_score_test
                C_ln_best = C_ln

        hp.LinearSVC.C.value = exp(C_ln_best)

    if hp.KNeighborsClassifier.enable:

        n_neighbors = 1
        n_neighbors_max = 30
        step = 1

        def val_KNeighborsClassifier(n_neighbors_val):
            clf = KNeighborsClassifier(n_neighbors=n_neighbors_val)
            rs_KNeighborsClassifier = results(clf, dt)
            val_score = mean(rs_KNeighborsClassifier.val_correct)
            return val_score

        val_score_best = val_KNeighborsClassifier(n_neighbors)
        n_neighbors_best = n_neighbors

        while n_neighbors < n_neighbors_max:
            n_neighbors += step
            val_score_test = val_KNeighborsClassifier(n_neighbors)
            if val_score_test > val_score_best:
                val_score_best = val_score_test
                n_neighbors_best = n_neighbors

        hp.KNeighborsClassifier.n_neighbors.value = n_neighbors_best

    if hp.SVC.enable:

        C_ln = -5
        C_ln_max = 5
        step = 0.1

        def val_SVC(Cval):
            clf = SVC(C=Cval, kernel='rbf')
            rs_SVC = results(clf, dt)
            val_score = mean(rs_SVC.val_correct)
            return val_score

        val_score_best = val_SVC(Cval=exp(C_ln))
        C_ln_best = C_ln

        while C_ln < C_ln_max:
            C_ln += step
            val_score_test = val_SVC(Cval=exp(C_ln))
            if val_score_test > val_score_best:
                val_score_best = val_score_test
                C_ln_best = C_ln

        hp.SVC.C.value = exp(C_ln_best)

    if hp.DecisionTreeClassifier.enable:

        max_depth = 2
        max_depth_max = 30
        step = 1

        def val_DecisionTreeClassifier(max_depth_val):
            clf = DecisionTreeClassifier(max_depth=max_depth_val)
            rs_DecisionTreeClassifier = results(clf, dt)
            val_score = mean(rs_DecisionTreeClassifier.val_correct)
            return val_score

        val_score_best = val_DecisionTreeClassifier(max_depth)
        max_depth_best = max_depth

        while max_depth < max_depth_max:
            max_depth += step
            val_score_test = val_DecisionTreeClassifier(max_depth)
            if val_score_test > val_score_best:
                val_score_best = val_score_test
                max_depth_best = max_depth

        hp.DecisionTreeClassifier.max_depth.value = max_depth_best

    if hp.AdaBoostClassifier.enable:

        n_estimators = 1
        n_estimators_max = 30
        step = 1

        def val_AdaBoostClassifier(n_estimators_val):
            clf = AdaBoostClassifier(n_estimators=n_estimators_val)
            rs_AdaBoostClassifier = results(clf, dt)
            val_score = mean(rs_AdaBoostClassifier.val_correct)
            return val_score

        val_score_best = val_AdaBoostClassifier(n_estimators)
        n_estimators_best = n_estimators

        while n_estimators < n_estimators_max:
            n_estimators += step
            val_score_test = val_AdaBoostClassifier(n_estimators)
            if val_score_test > val_score_best:
                val_score_best = val_score_test
                n_estimators_best = n_estimators

        hp.AdaBoostClassifier.n_estimators.value = n_estimators_best

    if hp.LinearDiscriminantAnalysis.enable:

        priors = (.1, .9)
        step = 0.1

        def val_LinearDiscriminantAnalysis(priors_val):
            clf = LinearDiscriminantAnalysis(priors=priors_val)
            rs_LinearDiscriminantAnalysis = results(clf, dt)
            val_score = mean(rs_LinearDiscriminantAnalysis.val_correct)
            return val_score

        val_score_best = val_LinearDiscriminantAnalysis(priors)
        priors_best = priors

        while priors[0] < 0.9:
            priors = (priors[0] + step, priors[1] - step)
            val_score_test = val_LinearDiscriminantAnalysis(priors)
            if val_score_test > val_score_best:
                val_score_best = val_score_test
                priors_best = priors

        hp.LinearDiscriminantAnalysis.priors.value = priors_best

    if hp.QuadraticDiscriminantAnalysis.enable:

        priors = (.1, .9)
        step = 0.1

        def val_QuadraticDiscriminantAnalysis(priors_val):
            clf = QuadraticDiscriminantAnalysis(priors=priors_val)
            rs_QuadraticDiscriminantAnalysis = results(clf, dt)
            val_score = mean(rs_QuadraticDiscriminantAnalysis.val_correct)
            return val_score

        val_score_best = val_QuadraticDiscriminantAnalysis(priors)
        priors_best = priors

        while priors[0] < 0.9:
            priors = (priors[0] + step, priors[1] - step)
            val_score_test = val_QuadraticDiscriminantAnalysis(priors)
            if val_score_test > val_score_best:
                val_score_best = val_score_test
                priors_best = priors

        hp.QuadraticDiscriminantAnalysis.priors.value = priors_best

    return hp


def optimize_weights(en, rs, max_iter=1000):
    """
    A simple local search that re-weights the ensemble to maximize validation set performance
    :param en: the current ensemble tree
    :param rs: the current results tree
    :param max_iter:  max iterations
    :return: a tracker of the iteration scheme, the new ensemble, the best weights
    """

    num_models = en.LogisticRegression.enable + en.RandomForestClassifier.enable + en.LinearSVC.enable + en.KNeighborsClassifier.enable + en.SVC.enable + en.DecisionTreeClassifier.enable + en.AdaBoostClassifier.enable + en.LinearDiscriminantAnalysis.enable + en.QuadraticDiscriminantAnalysis.enable
    weights_best = zeros(num_models)
    i = 0
    if en.LogisticRegression.enable:
        weights_best[i] = en.LogisticRegression.weight
        i += 1

    if en.RandomForestClassifier.enable:
        weights_best[i] = en.RandomForestClassifier.weight
        i += 1

    if en.LinearSVC.enable:
        weights_best[i] = en.LinearSVC.weight
        i += 1

    if en.KNeighborsClassifier.enable:
        weights_best[i] = en.KNeighborsClassifier.weight
        i += 1

    if en.SVC.enable:
        weights_best[i] = en.SVC.weight
        i += 1

    if en.DecisionTreeClassifier.enable:
        weights_best[i] = en.DecisionTreeClassifier.weight
        i += 1

    if en.AdaBoostClassifier.enable:
        weights_best[i] = en.AdaBoostClassifier.weight
        i += 1

    if en.LinearDiscriminantAnalysis.enable:
        weights_best[i] = en.LinearDiscriminantAnalysis.weight
        i += 1

    if en.QuadraticDiscriminantAnalysis.enable:
        weights_best[i] = en.QuadraticDiscriminantAnalysis.weight
        i += 1

    en_best = en
    rs_best = rs
    tracker2 = (mean(en_best.prob_train_correct), mean(en_best.prob_val_correct), mean(en_best.prob_test_correct))
    i = 0

    moves = 0
    print "Weight Search"
    while i < max_iter:
        # print i
        weights = neighborhood_weights(en_best)
        en_candidate = build_ensemble(rs_best, weights)

        if mean(en_candidate.prob_val_correct) >= mean(en_best.prob_val_correct):
            en_best = en_candidate
            weights_best = weights
            moves += 1
            print "%d moves" % moves

        tracker2 = vstack((tracker2, (
            mean(en_best.prob_train_correct), mean(en_best.prob_val_correct), mean(en_best.prob_test_correct))))

        i += 1

    return tracker2, en_best, weights_best