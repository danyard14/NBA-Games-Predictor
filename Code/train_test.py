from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from data_loader import encode_data


def train_decision_tree(train_data_path: str, allstar_path: str, prev_year_standings_path: str):
    x_all, labels = encode_data(train_data_path, allstar_path, prev_year_standings_path)
    print(x_all.shape)
    dtc = DecisionTreeClassifier(random_state=14)
    dtc.fit(x_all, labels)
    return dtc


def test_decision_tree(tree, test_data_path: str, allstar_path: str, prev_year_standings_path: str):
    x_all, labels = encode_data(test_data_path, allstar_path, prev_year_standings_path)
    print(x_all.shape)
    y_pred = tree.predict(x_all)
    y_true = [1] * len(y_pred)
    print(accuracy_score(labels, y_pred=y_pred))
    print(accuracy_score(labels, y_pred=y_true))