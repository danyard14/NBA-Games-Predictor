from train_test import train_decision_tree, test_decision_tree
from Utils.names import *

if __name__ == '__main__':
    decision_tree = train_decision_tree(TRAIN_DATA_PATH, TRAIN_ALLSTAR_PATH, TRAIN_STANDINGS_PATH)
    test_decision_tree(decision_tree, TEST_DATA_PATH, TEST_ALLSTAR_PATH, TEST_STANDINGS_PATH)
