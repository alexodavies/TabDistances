import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_wine


@pytest.fixture
def iris_df():
    data = load_iris()
    return pd.DataFrame(data.data, columns=data.feature_names)


@pytest.fixture
def wine_df():
    data = load_wine()
    return pd.DataFrame(data.data, columns=data.feature_names)


@pytest.fixture
def breast_cancer_df():
    data = load_breast_cancer()
    return pd.DataFrame(data.data, columns=data.feature_names)


@pytest.fixture
def iris_df_with_target():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["species"] = pd.Categorical([data.target_names[t] for t in data.target])
    return df
