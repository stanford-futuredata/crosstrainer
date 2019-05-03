from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import model_selection, metrics
import numpy as np
from scipy import sparse, constants

class CrossTrainer(BaseEstimator, ClassifierMixin):
    """Implementation of the method described in CrossTrainer: Practical Domain Adaptation with Loss Reweighting"""

    def __init__(self, clf, k=5, delta=0.01, verbose=False):
        """
        Initialize the CrossTrainer class.

        :param clf: The base classifier with a fit() function. (Ex: sklearn's SGDClassifier(...))
        :param k: The number of folds in k-fold cross-validation for fine-tuning the weighting parameter alpha.
        :param delta: The precision of the approximation of the optimal value of alpha.
        """
        self.clf = clf
        self.k = k
        self.delta = delta
        self.verbose = verbose

    def fit(self, Xtarget, ytarget, Xsource, ysource):
        """
        Approximates the optimal weighting parameter alpha through a specialized hyperparameter search and outputs the
        best model trained on a combination of target and source data.

        :param Xtarget: Input data corresponding to the target domain.
        :param ytarget: Labels corresponding to the target domain.
        :param Xsource: Input data corresponding to the supplemental source domain.
        :param ysource: Labels corresponding to the source domain.
        :return: Trained classifier and best alpha value.
        """
        self.Xtarget = Xtarget
        self.ytarget = ytarget
        self.Xsource = Xsource
        self.ysource= ysource
        self.datatype = _get_type(self.Xtarget)

        # Bracketing
        results = []
        acc_zero = self._cv_train_with_alpha(0, results)
        acc_one = self._cv_train_with_alpha(1, results)

        search_state = {"left": 0,
                        "right": 1,
                        "acc_left": acc_zero,
                        "acc_right": acc_one}

        self.bracket(search_state=search_state,
                     results=results)

        if search_state["acc_right"] > search_state["acc_middle"]:
            alpha_opt = search_state["right"]
        elif search_state["acc_left"] > search_state["acc_middle"]:
            alpha_opt = search_state["left"]
        else:
            alpha_opt = self.gss(search_state=search_state,
                                 results=results)

        if self.verbose:
            print("Optimal alpha: {:.4f}".format(alpha_opt))

        self._train_with_alpha(alpha=alpha_opt,
                               Xtarget=self.Xtarget,
                               ytarget=self.ytarget,
                               Xsource=self.Xsource,
                               ysource=self.ysource)

        return self.clf, alpha_opt


    def bracket(self, search_state, results):
        """
        Recursively tests values of alpha until either reaching the required precision or bracketing a local maximum.

        :param search_state: State of current search for optimal alpha.
        :param results: List of weight, accuracy tuples.
        """
        is_right_better = search_state["acc_right"] > search_state["acc_left"]
        middle = search_state["left"] + (search_state["right"] - search_state["left"]) / 2.0
        acc_middle = self._cv_train_with_alpha(middle, results)

        # End if we found a bracketing or reached the precision threshold
        if (search_state["right"] - middle < self.delta)  \
            or (is_right_better and acc_middle >= search_state["acc_right"])  \
            or (not is_right_better and acc_middle >= search_state["acc_left"]):
                search_state["middle"] = middle
                search_state["acc_middle"] = acc_middle
                return

        if is_right_better:
            search_state["left"] = middle
            search_state["acc_left"] = acc_middle
            return self.bracket(search_state=search_state,
                                results=results)
        else:
            search_state["right"] = middle
            search_state["acc_right"] = acc_middle
            return self.bracket(search_state=search_state,
                                results=results)

    def gss(self, search_state, results):
        """
        Performs golden section search to optimize the hyperparameter alpha.

        :param search_state: State of current search over alpha.
        :param results: List of weight, accuracy tuples.
        :return: Best value of alpha found as measured by validation accuracy.
        """
        ratio = 1 - (1 / constants.golden_ratio)
        longer_diff = max(search_state["middle"] - search_state["left"], search_state["right"] - search_state["middle"])
        while longer_diff > self.delta:
            is_right_longer = (search_state["middle"] - search_state["left"]) < \
                              (search_state["right"] - search_state["middle"])

            if is_right_longer:
                alpha = search_state["middle"] + ratio * longer_diff
            else:
                alpha = search_state["middle"] - ratio * longer_diff

            acc = self._cv_train_with_alpha(alpha, results)

            # New middle
            if acc > search_state["acc_middle"]:
                if is_right_longer:
                    search_state["left"] = search_state["middle"]
                    search_state["acc_left"] = search_state["acc_middle"]
                    search_state["middle"] = alpha
                    search_state["acc_middle"] = acc
                else:
                    search_state["right"] = search_state["middle"]
                    search_state["acc_right"] = search_state["acc_middle"]
                    search_state["middle"] = alpha
                    search_state["acc_middle"] = acc
            # New edge
            else:
                if is_right_longer:
                    search_state["right"] = alpha
                    search_state["acc_right"] = acc
                else:
                    search_state["left"] = alpha
                    search_state["acc_left"] = acc

            longer_diff = max(search_state["middle"] - search_state["left"], search_state["right"] - search_state["middle"])

        return search_state["middle"]

    def _cv_train_with_alpha(self, alpha, results):
        """
        Uses k-fold cross-validation to estimate the accuracy of a model trained with the given weight alpha.

        :param alpha: The value used to reweight the loss function for target and source data.
        :param results: List of weight, accuracy tuples.
        :return: Average accuracy over the k-fold cross-validation.
        """
        skf = model_selection.StratifiedKFold(n_splits=self.k)
        acc_sum = 0.0
        for train_index, val_index in skf.split(self.Xtarget, self.ytarget):
            Xtarget_train, Xtarget_val = self.Xtarget[train_index], self.Xtarget[val_index]
            ytarget_train, ytarget_val = self.ytarget[train_index], self.ytarget[val_index]
            self._train_with_alpha(alpha=alpha,
                                   Xtarget=Xtarget_train,
                                   ytarget=ytarget_train,
                                   Xsource=self.Xsource,
                                   ysource=self.ysource)
            acc_sum += 100 * metrics.accuracy_score(y_true=ytarget_val,
                                                    y_pred=self.clf.predict(Xtarget_val))

        acc_mean = acc_sum / self.k
        results.append((alpha, acc_mean))
        if self.verbose:
            print("Weight:              {:.4f}".format(alpha))
            print("Validation Accuracy: {:.3f}\n".format(acc_mean))
        return acc_mean

    def _train_with_alpha(self, alpha, Xtarget, ytarget, Xsource, ysource):
        """
        Trains a model with a given weight alpha.

        :param alpha: The value used to reweight the loss function for target and source data.
        :param Xtarget: Target inputs.
        :param ytarget: Target labels.
        :param Xsource: Source inputs.
        :param ysource: Source labels.
        """
        if self.datatype == 'numpy':
            Xtrain = np.vstack((Xtarget, Xsource))
        elif self.datatype == 'sparse':
            Xtrain = sparse.vstack((Xtarget, Xsource))
        ytrain = np.concatenate((ytarget, ysource))

        ntarget, nsource = len(ytarget), len(ysource)
        wtarget = np.ones_like(ytarget) * alpha
        wsource = np.ones_like(ysource) * (1 - alpha) * ntarget / nsource
        wtrain = np.concatenate((wtarget, wsource))

        self.clf.fit(Xtrain, ytrain, sample_weight=wtrain)


def _get_type(data):
    """
    Returns the type of the inputs to the model.
    """
    if type(data) is np.ndarray:
        return 'numpy'
    elif type(data) is sparse.csr_matrix:
        return 'sparse'
    else:
        raise Exception('Unknown data type: ' + str(type(data)))
