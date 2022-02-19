
import sklearn
import graphviz
import numpy
import pandas
import matplotlib
import scipy
import joblib
import threadpoolctl


def check_model_version():
    print("sklearn:%s"%sklearn.__version__)
    print("graphviz:%s"%graphviz.__version__)
    print("numpy:%s"%numpy.__version__)
    print("pandas:%s"%pandas.__version__)
    print("matplotlib:%s"%matplotlib.__version__)
    print("scipy:%s"%scipy.__version__)
    print("joblib:%s"%joblib.__version__)
    print("threadpoolctl:%s"%threadpoolctl.__version__)


if __name__ == '__main__':
    check_model_version()
