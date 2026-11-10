# Exact source-definition subset of Healthy-AI/malearn (see extraction_manifest.json).
# Original license is retained in vendor/malearn/LICENSE.md.
import math, copy, warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, is_classifier, clone, _fit_context
from sklearn.utils import check_random_state, Bunch
from sklearn.utils.multiclass import unique_labels
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.tree import _tree
from sklearn.tree._tree import Tree, _build_pruned_tree_ccp, ccp_pruning_path
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, _check_sample_weight


class Node:
    def __init__(
        self,
        index,
        parent=None,
        left_child=None,
        right_child=None,
    ):
        self.index = index
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
    
    def add_children(self):
        self.left_child = Node(index=2 * self.index + 1, parent=self)
        self.right_child = Node(index=2 * self.index + 2, parent=self)

    def remove_children(self):
        self.left_child = None
        self.right_child = None
    
    @property
    def layer_index(self):
        return int(math.log2(self.index + 1))
    
    @property
    def is_leaf(self):
        return self.left_child is None and self.right_child is None

def count_nodes(node):
    if node is not None:
        return (
            1 + count_nodes(node.left_subtree) + count_nodes(node.right_subtree)
        )
    return 0

class TreeNode:
    def __init__(
        self,
        value,
        impurity,
        *,
        n_node_samples,
        weighted_n_node_samples,
        depth,
        feature=None,
        threshold=None,
        left_subtree=None,
        right_subtree=None,
        missingness_reliance=None,
    ):
        self.value = value
        self.impurity = impurity

        self.n_node_samples = n_node_samples
        self.weighted_n_node_samples = weighted_n_node_samples

        # We do not handle missing values in this implementation.
        self.missing_go_to_left = 0

        self.depth = depth

        undefined = _tree.TREE_UNDEFINED
        self.feature = feature if feature is not None else undefined
        self.threshold = threshold if threshold is not None else undefined

        self.left_subtree = left_subtree
        self.right_subtree = right_subtree

        self.missingness_reliance = missingness_reliance

    def predict(self, x, m=None, return_depth=False):
        if self.feature == _tree.TREE_UNDEFINED:
            if return_depth:
                return self.value, self.depth
            return self.value
        if m is not None and m[self.feature] == 1:
            if return_depth:
                return self.value, self.depth
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left_subtree.predict(x, m, return_depth)
        else:
            return self.right_subtree.predict(x, m, return_depth)

    @property
    def is_leaf(self):
        return self.feature == _tree.TREE_UNDEFINED

    def __eq__(self, other):
        return (
            isinstance(other, TreeNode)
            # np.array_equal works for both arrays and scalars.
            and np.array_equal(self.value, other.value)
        )

def _populate_node_list(node, node_list):
    node = copy.deepcopy(node)

    if node.feature == _tree.TREE_UNDEFINED:
        node.left_child = _tree.TREE_LEAF
        node.right_child = _tree.TREE_LEAF

        node_list.append(node)
    else:
        left_idx = len(node_list) + 1
        right_idx = left_idx + count_nodes(node.left_subtree)

        node.left_child = left_idx
        node.right_child = right_idx

        node_list.append(node)

        _populate_node_list(node.left_subtree, node_list)
        _populate_node_list(node.right_subtree, node_list)

def convert_to_sklearn_tree(root_node, n_features, n_classes):
    nodes = []
    _populate_node_list(root_node, nodes)

    max_depth = max(node.depth for node in nodes)
    node_count = len(nodes)

    values = [node.value for node in nodes]
    values = np.array(values).reshape(-1, 1, n_classes)

    nodes = [
        (
            node.left_child,
            node.right_child,
            node.feature,
            node.threshold,
            node.impurity,
            node.n_node_samples,
            node.weighted_n_node_samples,
            node.missing_go_to_left,
        )
        for node in nodes
    ]
    nodes = np.array(nodes, dtype=_tree.NODE_DTYPE)

    tree = _tree.Tree(n_features, np.atleast_1d(n_classes), 1)
    tree.__setstate__(
        {
            "max_depth": max_depth,
            "node_count": node_count,
            "nodes": nodes,
            "values": values,
        }
    )

    return tree

class ExtendedTree:
    def __init__(self, original_tree, root_node):
        self.n_features = original_tree.n_features
        self.n_classes = original_tree.n_classes
        self.n_outputs = original_tree.n_outputs
        state = original_tree.__getstate__()
        state["node_array"] = state.pop("nodes")
        state["value_array"] = state.pop("values")
        self.__dict__.update(state)
        self.nodes = list(self._yield_nodes(root_node))

    @property
    def children_left(self):
        return self.node_array["left_child"][:self.node_count]

    @property
    def children_right(self):
        return self.node_array["right_child"][:self.node_count]

    @property
    def feature(self):
        return self.node_array["feature"][:self.node_count]

    @property
    def threshold(self):
        return self.node_array["threshold"][:self.node_count]

    @property
    def impurity(self):
        return self.node_array["impurity"][:self.node_count]

    @property
    def n_node_samples(self):
        return self.node_array["n_node_samples"][:self.node_count]

    @property
    def weighted_n_node_samples(self):
        return self.node_array["weighted_n_node_samples"][:self.node_count]

    @property
    def missing_go_to_left(self):
        return self.node_array["missing_go_to_left"][:self.node_count]

    @property
    def value(self):
        return self.value_array[:self.node_count]

    @property
    def missingness_reliance(self):
        return [node.missingness_reliance for node in self.nodes]

    def _yield_nodes(self, root_node):
        node_stack = [root_node]
        while node_stack:
            node = node_stack.pop()
            yield node
            if node.right_subtree is not None:
                node_stack.append(node.right_subtree)
            if node.left_subtree is not None:
                node_stack.append(node.left_subtree)

    def apply(self, X):
        out = []
        n_samples = X.shape[0]
        indexed_nodes = [(i, *node) for i, node in enumerate(self.node_array)]
        for i in range(n_samples):
            node = indexed_nodes[0]  # Start at the root
            while node[1] != _tree.TREE_LEAF:
                if X[i, node[3]] <= node[4]:
                    node = indexed_nodes[node[1]]
                else:
                    node = indexed_nodes[node[2]]
            out.append(node[0])
        return np.array(out)

def update_tree_structure(old_node, new_node, nodes_after_pruning):
    if new_node[0] == _tree.TREE_LEAF and new_node[1] == _tree.TREE_LEAF:
        old_node.feature = old_node.threshold = _tree.TREE_UNDEFINED
        old_node.left_child = old_node.right_child = _tree.TREE_LEAF
        old_node.left_subtree = old_node.right_subtree = None
        return
    else:
        assert new_node[0] != _tree.TREE_LEAF or new_node[1] != _tree.TREE_LEAF
        update_tree_structure(
            old_node.left_subtree,
            nodes_after_pruning[new_node[0]],
            nodes_after_pruning,
        )
        update_tree_structure(
            old_node.right_subtree,
            nodes_after_pruning[new_node[1]],
            nodes_after_pruning,
        )

def entropy(distr):
    n = sum(distr)
    ps = [n_i / n for n_i in distr]
    return -sum(p * np.log2(p) if p > 0 else 0 for p in ps)

def info_gain_scorer(n_low, low_distr, n_high, high_distr):
    return -(
        (n_low * entropy(low_distr) + n_high * entropy(high_distr)) 
        / (n_low + n_high)
    )

def gini_impurity(distr):
    n = sum(distr)
    ps = [n_i / n for n_i in distr]
    return 1 - sum(p**2 for p in ps)

def gini_scorer(n_low, low_distr, n_high, high_distr):
    return -(
        (n_low * gini_impurity(low_distr) + n_high * gini_impurity(high_distr))
        / (n_low + n_high)
    )

def check_missingness_mask(M, X):
    if M is None:
        return None
    M = check_array(M)
    if not M.shape[0] == X.shape[0]:
        raise ValueError("The number of samples in X and M must be the same.")
    return M

class ClassifierMixin:
    _estimator_type = "classifier"

    accepted_metrics = {
        "accuracy": {
            "function": metrics.accuracy_score,
            "requires_proba": False,
            "kwargs": {},
            "lower_is_better": False,
        },
        "roc_auc_ovr": {
            "function": metrics.roc_auc_score,
            "requires_proba": True,
            "kwargs": {"multi_class": "ovr", "average": "macro"},
            "lower_is_better": False,
        },
        "roc_auc_ovr_weighted": {
            "function": metrics.roc_auc_score,
            "requires_proba": True,
            "kwargs": {"multi_class": "ovr", "average": "weighted"},
            "lower_is_better": False,
        },
        "log_loss": {
            "function": metrics.log_loss,
            "requires_proba": True,
            "kwargs": {"normalize": True},
            "lower_is_better": True,
        },
    }

    def score(
        self,
        X,
        y,
        M=None,
        sample_weight=None,
        metric="accuracy",
        **predict_params,
    ):
        is_accepted_metric = (
            metric in self.accepted_metrics
            or metric == "missingness_reliance_score"
        )
        if not is_accepted_metric:
            raise ValueError(
                f"Got invalid metric {metric}. Valid metrics are "
                f"'missingness_reliance_score' and {self.accepted_metrics.keys()}."
            )

        if metric == "missingness_reliance_score":
            if M is None:
                raise ValueError(
                    "The metric 'missingness_reliance' requires the "
                    "missingness mask `M`."
                )
            return (-1) * self.compute_missingness_reliance(X, M)

        if self.accepted_metrics[metric]["requires_proba"]:
            yp = self.predict_proba(X, **predict_params)
            if yp.shape[1] == 2:
                yp = yp[:, 1]
        else:
            yp = self.predict(X, **predict_params)

        score_params = self.accepted_metrics[metric]["kwargs"]
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        
        try:
            return self.accepted_metrics[metric]["function"](y, yp, **score_params)
        except ValueError:
            return np.nan

    def _more_tags(self):
        return {"requires_y": True}

class BaseMADT(BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: dict = {
        "max_depth": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "alpha": [Interval(Real, 0.0, None, closed="left")],
        "compute_rho_per_node": [bool],
        "ccp_alpha": [Interval(Real, 0.0, None, closed="left")],
    }

    def __init__(
        self,
        max_depth,
        random_state,
        alpha,
        compute_rho_per_node,
        ccp_alpha,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.random_state = random_state
        self.alpha = alpha
        self.compute_rho_per_node = compute_rho_per_node
        self.ccp_alpha = ccp_alpha

    def _fit(
        self,
        X,
        y,
        M=None,
        *,
        clear_X_M=True,
        check_input=True,
        sample_weight=None,
        missing_values_in_feature_mask=None,
        keep_original_tree=False,
    ):
        if check_input:
            X, y = check_X_y(X, y)

        # Meta-estimators may pass labels as a column vector, but we need them
        # as a 1-D array.
        if y.ndim > 1:
            y = y.ravel()

        if is_classifier(self):
            labels = unique_labels(y)
            n_labels = len(labels)
            if not np.all(labels == np.arange(n_labels)):
                raise ValueError(
                    "Labels are expected to be in the range 0 to n_classes-1."
                )

        if missing_values_in_feature_mask is not None:
            if M is not None:
                warnings.warn(
                    "Both `M` and `missing_values_in_feature_mask` are "
                    "provided. `M` will be used."
                )
            else:
                M = missing_values_in_feature_mask

        M = check_missingness_mask(M, X)

        sample_weight = _check_sample_weight(sample_weight, X)

        random_state = check_random_state(self.random_state)

        self.n_features_in_ = X.shape[1]
        if is_classifier(self):
            self.classes_ = unique_labels(y)
            self.n_classes_ = len(self.classes_)

        self._split_features = []
        self.root_ = self._make_tree(X, y, M, sample_weight, 0, random_state)

        if self.compute_rho_per_node and M is not None:
            self.root_.X, self.root_.M = X, M
            self._compute_missingness_reliance_for_all_nodes(self.root_, clear_X_M)

        n_classes = self.n_classes_ if is_classifier(self) else 1
        self.tree_ = convert_to_sklearn_tree(
            self.root_, self.n_features_in_, n_classes
        )

        self._prune_tree()

        # Update `self.root_` to match the pruned tree.
        nodes_after_pruning = self.tree_.__getstate__()["nodes"]
        update_tree_structure(
            self.root_, nodes_after_pruning[0], nodes_after_pruning,
        )

        # Since sklearn.tree._tree.Tree is implemented in Cython, it does not
        # support dynamically adding new attributes like missingness_reliance.
        # To work around this, we create a new tree instance and set the
        # attribute there, allowing us to use, e.g., the `plot_tree` method.
        #
        # Note: We need an instance of sklearn.tree._tree.Tree to utilize the
        # pruning method in scikit-learn.
        if not keep_original_tree:
            self.tree_ = ExtendedTree(self.tree_, self.root_)

        return self

    def _prune_tree(self):
        """Prune tree using Minimal Cost-Complexity Pruning."""
        check_is_fitted(self)

        if self.ccp_alpha == 0.0:
            return

        # Build pruned tree.
        if is_classifier(self):
            n_classes = np.atleast_1d(self.n_classes_)
            pruned_tree = Tree(self.n_features_in_, n_classes, 1)
        else:
            pruned_tree = Tree(
                self.n_features_in_, np.array([1], dtype=np.intp), 1,
            )
        _build_pruned_tree_ccp(pruned_tree, self.tree_, self.ccp_alpha)

        self.tree_ = pruned_tree

    def cost_complexity_pruning_path(self, X, y, M=None, sample_weight=None):
        est = clone(self).set_params(ccp_alpha=0.0)
        est._fit(X, y, M, sample_weight=sample_weight, keep_original_tree=True)
        return Bunch(**ccp_pruning_path(est.tree_))

    def _validate_X_predict(self, X, check_input=True):
        if check_input:
            X = check_array(X)
        n_features = X.shape[1]
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.n_features_in_} "
                "features were expected."
            )
        return X

    def _support_missing_values(self, X):
        return False

    def _compute_missing_values_in_feature_mask(self, X, estimator_name=None):
        return None

    def predict(self, X, check_input=True, M=None, return_proba=False):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        M = check_missingness_mask(M, X)
        if M is not None:
            yp = np.array([self.root_.predict(x, m) for x, m in zip(X, M)])
        else:
            yp = np.array([self.root_.predict(x) for x in X])
        if is_classifier(self) and not return_proba:
            # Classes are ordered from 0 to n_classes-1.
            return np.argmax(yp, axis=1)
        return yp

    def get_prediction_depth(self, X, M=None):
        check_is_fitted(self)
        X = self._validate_X_predict(X)
        M = check_missingness_mask(M, X)
        if M is not None:
            return np.array(
                [
                    self.root_.predict(x, m, return_depth=True)[-1]
                    for x, m in zip(X, M)
                ]
            )
        return np.array(
            [self.root_.predict(x, return_depth=True)[-1] for x in X]
        )

    def _make_tree(self, X, y, M, sample_weight, depth, random_state):
        """Recursively build the decision tree."""

        node_value = self._get_node_value(y, sample_weight)
        node_impurity = self._get_node_impurity(node_value, y, sample_weight)
        n_node_samples = len(X)
        weighted_n_node_samples = sum(sample_weight)

        # Check if the training set is completely homogeneous, or if the
        # maximum depth has been reached.
        if (
            depth >= self.max_depth
            or self._is_homogeneous(node_value, y, sample_weight)
        ):
            return TreeNode(
                node_value,
                node_impurity,
                n_node_samples=n_node_samples,
                weighted_n_node_samples=weighted_n_node_samples,
                depth=depth,
            )

        # Select the best feature to split on.
        _split_score, split_feature, split_threshold = max(
            self._get_split_candidates(
                X, y, M, sample_weight, depth, random_state
            ),
        )
        self._split_features.append(split_feature)

        if split_feature is None:
            return TreeNode(
                node_value,
                node_impurity,
                n_node_samples=n_node_samples,
                weighted_n_node_samples=weighted_n_node_samples,
                depth=depth,
            )

        # Split the training set into two parts and build the subtrees.

        # TODO: Update the missingness mask. Features with missing values
        # should not be penalized for being selected further down the tree if
        # they are already used in the tree.
        #
        # Note: We should take sample weights into account when updating the
        # missingness mask.

        data_left, data_right = self._split_by_feature(
            X, y, M, sample_weight, split_feature, split_threshold
        )

        left_subtree = self._make_tree(*data_left, depth+1, random_state)
        right_subtree = self._make_tree(*data_right, depth+1, random_state)

        if self.compute_rho_per_node and M is not None:
            # Store the training data and the missingness mask in the nodes to
            # compute the missingness reliance.
            left_subtree.X, left_subtree.M = data_left[0], data_left[2]
            right_subtree.X, right_subtree.M = data_right[0], data_right[2]

        if left_subtree == right_subtree:
            return left_subtree

        return TreeNode(
            node_value,
            node_impurity,
            n_node_samples=n_node_samples,
            weighted_n_node_samples=weighted_n_node_samples,
            feature=split_feature,
            threshold=split_threshold,
            left_subtree=left_subtree,
            right_subtree=right_subtree,
            depth=depth,
        )

    def _split_by_feature(self, X, y, M, sw, feature, threshold):
        left = X[:, feature] <= threshold
        right = ~left
        Xl, Xr = X[left], X[right]
        yl, yr = y[left], y[right]
        Ml, Mr = (M[left], M[right]) if M is not None else (None, None)
        swl, swr = sw[left], sw[right]
        return (Xl, yl, Ml, swl), (Xr, yr, Mr, swr)

    def _get_features_along_decision_path(self, x, max_depth=None):
        """Get the features along the decision path for the input `x`."""
        node = self.root_
        if max_depth is None:
            max_depth = self.max_depth
        while node.depth <= max_depth and node.left_subtree is not None:
            yield node.feature
            if x[node.feature] <= node.threshold:
                node = node.left_subtree
            else:
                node = node.right_subtree

    def _compute_missingness_reliance_for_all_nodes(self, node, clear_X_M=True):
        if node is None:
            return
        node.missingness_reliance = self.compute_missingness_reliance(
            node.X, node.M, max_depth=node.depth
        )
        if clear_X_M:
            del node.X
            del node.M
        self._compute_missingness_reliance_for_all_nodes(node.left_subtree, clear_X_M)
        self._compute_missingness_reliance_for_all_nodes(node.right_subtree, clear_X_M)

    def compute_missingness_reliance(
        self,
        X,
        M,
        sample_mask=None,
        reduce=True,
        max_depth=None,
    ):
        # TODO: Enable the use of sample weights as input.
        # TODO: Compute the average missingness reliance along the decision
        # path.
        check_is_fitted(self)
        miss_reliance = np.zeros_like(M)
        for i, (x, m) in enumerate(zip(X, M)):
            if sample_mask is not None and not sample_mask[i]:
                continue
            for feature in self._get_features_along_decision_path(x, max_depth):
                miss_reliance[i, feature] = m[feature]
        if reduce:
            return np.mean(np.max(miss_reliance, axis=1))
        else:
            return miss_reliance

    def _get_split_candidates(self, X, y, M, sample_weight, depth, random_state):
        features = np.arange(self.n_features_in_)
        random_state.shuffle(features)
        for feature in features:
            yield self._best_split(
                X, y, feature, sample_weight=sample_weight, M=M
            )

    @abstractmethod
    def _get_node_value(self, y, sample_weight):
        """Get the value of a node in the decision tree."""

    @abstractmethod
    def _get_node_impurity(self, node_value, y, sample_weight):
        """Get the impurity of a node in the decision tree."""

    @abstractmethod
    def _is_homogeneous(self, node_value, y, sample_weight):
        """Determine whether a node is homogeneous."""

    @abstractmethod
    def _best_split(self, X, y, feature, sample_weight, M=None):
        """Find the best split for a given feature."""

CRITERIA_CLF = {"info_gain": info_gain_scorer, "gini": gini_scorer}

class MADTClassifier(ClassifierMixin, BaseMADT):
    """Missingness-avoiding decision tree classifier."""

    _parameter_constraints: dict = {
        **BaseMADT._parameter_constraints,
        "criterion": [StrOptions({"info_gain", "gini"})],
    }

    def __init__(
        self,
        criterion="gini",
        max_depth=3,
        random_state=None,
        alpha=1.0,
        compute_rho_per_node=False,
        ccp_alpha=0.0,
    ):
        super().__init__(
            max_depth, random_state, alpha, compute_rho_per_node, ccp_alpha
        )
        self.criterion = criterion

    def _get_node_value(self, y, sample_weight):
        node_value = np.zeros(self.n_classes_, dtype=float)
        for class_label, weight in zip(y, sample_weight):
            node_value[class_label] += weight
        node_value /= sum(sample_weight)
        return node_value

    def _get_node_impurity(self, node_value, y, sample_weight):
        return (
            gini_impurity(node_value)
            if self.criterion == "gini" else entropy(node_value)
        )

    def _is_homogeneous(self, node_value, y, sample_weight):
        return max(node_value) == 1.0

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, M=None, sample_weight=None):
        return super()._fit(X, y, M, sample_weight=sample_weight)

    def predict_proba(self, X, check_input=True):
        return self.predict(X, check_input, return_proba=True)

    def _best_split(self, X, y, feature, sample_weight, M=None):
        sorted_indices = np.argsort(X[:, feature])
        X_sorted = X[sorted_indices, feature]
        y_sorted = y[sorted_indices]
        sample_weight_sorted = sample_weight[sorted_indices]

        low_distr = np.zeros(self.n_classes_, dtype=float)
        high_distr = np.zeros(self.n_classes_, dtype=float)

        for class_label, weight in zip(y_sorted, sample_weight_sorted):
            high_distr[class_label] += weight

        n_low = 0
        n_high = np.sum(sample_weight)

        max_score = -np.inf
        i_max_score = None

        n = len(y)
        for i in range(n - 1):
            yi = y_sorted[i]
            wi = sample_weight_sorted[i]

            low_distr[yi] += wi
            high_distr[yi] -= wi

            n_low += wi
            n_high -= wi

            if n_low == 0:
                continue

            if n_high == 0:
                break

            # If the input is equal to the input at the next position, we will
            # not consider a split here.
            if np.isclose(X_sorted[i], X_sorted[i+1]):
                continue

            criterion_function = CRITERIA_CLF[self.criterion]
            score = criterion_function(n_low, low_distr, n_high, high_distr)

            if score > max_score:
                max_score = score
                i_max_score = i

        if i_max_score is None:
            return -np.inf, None, None

        if M is not None:
            max_score -= self.alpha * np.mean(sample_weight * M[:, feature])

        split_threshold = (
            0.5 * (X_sorted[i_max_score] + X_sorted[i_max_score + 1])
        )
        return max_score, feature, split_threshold
