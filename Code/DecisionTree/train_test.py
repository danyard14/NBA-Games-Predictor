from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from data_loader import encode_data


def train_decision_tree(train_data_path: str, allstar_path: str, prev_year_standings_path: str, criterion, splitter, random_state):
    x_all, labels = encode_data(train_data_path, allstar_path, prev_year_standings_path)
    dtc = DecisionTreeClassifier(criterion=criterion, splitter=splitter, random_state=random_state)
    dtc.fit(x_all, labels)
    return dtc


def test_decision_tree(tree, test_data_path: str, allstar_path: str, prev_year_standings_path: str):
    x_all, labels = encode_data(test_data_path, allstar_path, prev_year_standings_path)
    y_pred = tree.predict(x_all)
    # y_true = [1] * len(y_pred)
    print('\taccuracy:', accuracy_score(labels, y_pred=y_pred))