import unittest
import crosstrainer
import numpy as np
from scipy import sparse
from sklearn import linear_model, model_selection, metrics, datasets

class TestCrossTrainer(unittest.TestCase):
    def testa_sraa_500(self):
        datapath = "/Users/justinchen/Documents/projects/cross-train/crosstrainer/crosstrainer/tests/data/sraa/"
        Xtarget = sparse.load_npz(datapath + "x_real_train.npz")
        ytarget = np.load(datapath + "y_real_train.npy")
        Xtarget, __, ytarget, __ = model_selection.train_test_split(Xtarget, ytarget,
                                                                    train_size=500./len(ytarget),
                                                                    test_size=None)
        Xsource = sparse.load_npz(datapath + "x_sim.npz")
        ysource = np.load(datapath + "y_sim.npy")
        Xtest = sparse.load_npz(datapath + "x_real_test.npz")
        ytest = np.load(datapath + "y_real_test.npy")

        lr = linear_model.SGDClassifier(loss="log", tol=1e-3, warm_start=True)
        ct = crosstrainer.CrossTrainer(lr, k=5, delta=0.01, verbose=True)
        lr, alpha = ct.fit(Xtarget, ytarget, Xsource, ysource)

        acc_test = 100 * metrics.accuracy_score(y_true=ytest, y_pred=lr.predict(Xtest))
        print("Test accuracy: {:.3f}".format(acc_test))
        print("\n\n")

    def testb_sraa_1000(self):
        datapath = "/Users/justinchen/Documents/projects/cross-train/crosstrainer/crosstrainer/tests/data/sraa/"
        Xtarget = sparse.load_npz(datapath + "x_real_train.npz")
        ytarget = np.load(datapath + "y_real_train.npy")
        Xtarget, __, ytarget, __ = model_selection.train_test_split(Xtarget, ytarget,
                                                                    train_size=1000./len(ytarget),
                                                                    test_size=None)
        Xsource = sparse.load_npz(datapath + "x_sim.npz")
        ysource = np.load(datapath + "y_sim.npy")
        Xtest = sparse.load_npz(datapath + "x_real_test.npz")
        ytest = np.load(datapath + "y_real_test.npy")

        lr = linear_model.SGDClassifier(loss="log", tol=1e-3, warm_start=True)
        ct = crosstrainer.CrossTrainer(lr, k=5, delta=0.01, verbose=True)
        lr, alpha = ct.fit(Xtarget, ytarget, Xsource, ysource)

        acc_test = 100 * metrics.accuracy_score(y_true=ytest, y_pred=lr.predict(Xtest))
        print("Test accuracy: {:.3f}".format(acc_test))
        print("\n\n")


    def testc_sraa_6400(self):
        datapath = "/Users/justinchen/Documents/projects/cross-train/crosstrainer/crosstrainer/tests/data/sraa/"
        Xtarget = sparse.load_npz(datapath + "x_real_train.npz")
        ytarget = np.load(datapath + "y_real_train.npy")
        Xsource = sparse.load_npz(datapath + "x_sim.npz")
        ysource = np.load(datapath + "y_sim.npy")
        Xtest = sparse.load_npz(datapath + "x_real_test.npz")
        ytest = np.load(datapath + "y_real_test.npy")

        lr = linear_model.SGDClassifier(loss="log", tol=1e-3, warm_start=True)
        ct = crosstrainer.CrossTrainer(lr, k=5, delta=0.01, verbose=True)
        lr, alpha = ct.fit(Xtarget, ytarget, Xsource, ysource)

        acc_test = 100 * metrics.accuracy_score(y_true=ytest, y_pred=lr.predict(Xtest))
        print("Test accuracy: {:.3f}".format(acc_test))
        print("\n\n")


    def testd_target_only(self):
        Xtarget, ytarget = datasets.make_classification(n_samples=1000,
                                                        n_features=20,
                                                        n_informative=10,
                                                        n_classes=2)
        Xsource, ysource = datasets.make_classification(n_samples=1000,
                                                        n_features=20,
                                                        n_informative=10,
                                                        n_classes=2)

        lr = linear_model.SGDClassifier(loss="log", tol=1e-3, warm_start=True)
        ct = crosstrainer.CrossTrainer(lr, k=5, delta=0.01, verbose=True)
        lr, alpha = ct.fit(Xtarget, ytarget, Xsource, ysource)
        print("\n\n")


    def teste_both(self):
        Xtarget, ytarget = datasets.make_classification(n_samples=2000,
                                                        n_features=20,
                                                        n_informative=10,
                                                        n_classes=2)
        Xtarget, Xsource, ytarget, ysource = model_selection.train_test_split(Xtarget, ytarget,
                                                                              train_size=0.5,
                                                                              test_size=0.5)

        lr = linear_model.SGDClassifier(loss="log", tol=1e-3, warm_start=True)
        ct = crosstrainer.CrossTrainer(lr, k=5, delta=0.01, verbose=True)
        lr, alpha = ct.fit(Xtarget, ytarget, Xsource, ysource)
        print("\n\n")

