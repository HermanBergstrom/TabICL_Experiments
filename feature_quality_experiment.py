import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

try:
    from tabicl import TabICLClassifier
except ImportError:
    TabICLClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from tabpfn import TabPFNClassifier
except ImportError:
    TabPFNClassifier = None

try:
    from tabpfn_factory import build_tabpfn_classifier, PreprocessingPreset
    _tabpfn_factory_available = True
except ImportError:
    _tabpfn_factory_available = False


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Minimal sklearn-compatible binary MLP classifier backed by PyTorch.
    """

    def __init__(
        self,
        hidden_layer_sizes=(256, 128),
        learning_rate=1e-3,
        batch_size=256,
        max_epochs=200,
        random_state=42,
        device=None,
        verbose=False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _build_network(self, input_dim):
        layers = []
        last_dim = input_dim
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(torch.nn.Linear(last_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            last_dim = hidden_dim
        layers.append(torch.nn.Linear(last_dim, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, X, y):
        if torch is None:
            raise ImportError(
                "PyTorch is required for TorchMLPClassifier but is not installed."
            )

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.classes_ = np.array([0, 1])

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.device_ = self.device
        if self.device_ is None:
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_ = self._build_network(X.shape[1]).to(self.device_)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
        )

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model_.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)

                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.shape[0]

            if self.verbose and (epoch + 1) % 50 == 0:
                mean_loss = epoch_loss / len(dataset)
                print(f"TorchMLP epoch={epoch + 1}, loss={mean_loss:.6f}")

        return self

    def predict_proba(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchMLPClassifier must be fit before prediction.")

        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor).squeeze(1)
            p1 = torch.sigmoid(logits).cpu().numpy()

        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class TorchLogisticClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible binary logistic regression backed by PyTorch.

    penalty='none' / C=inf  →  unregularized, solved with LBFGS
    penalty='l2'             →  L2 regularization, solved with LBFGS
    penalty='l1'             →  L1 regularization, solved with ISTA
                                (proximal gradient with soft-thresholding)

    Uses CUDA when available, falls back to CPU.
    """

    def __init__(self, penalty="l2", C=1.0, max_iter=200, random_state=42, device=None):
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        if torch is None:
            raise ImportError("PyTorch is required for TorchLogisticClassifier.")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        n_samples, n_features = X.shape

        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.random_state)

        X_t = torch.tensor(X, device=self.device_)
        y_t = torch.tensor(y, device=self.device_)

        no_reg = self.penalty in ("none", None) or self.C == np.inf
        reg = 0.0 if no_reg else 1.0 / (self.C * n_samples)

        if self.penalty == "l1" and not no_reg:
            self.w_, self.b_ = self._fit_ista(X_t, y_t, l1_reg=reg)
        else:
            self.w_, self.b_ = self._fit_lbfgs(X_t, y_t, l2_reg=reg)

        return self

    def _fit_lbfgs(self, X_t, y_t, l2_reg):
        n_features = X_t.shape[1]
        w = torch.zeros(n_features, device=X_t.device, requires_grad=True)
        b = torch.zeros(1, device=X_t.device, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [w, b],
            max_iter=self.max_iter,
            tolerance_grad=1e-6,
            tolerance_change=1e-10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                X_t @ w + b, y_t, reduction="mean"
            )
            if l2_reg > 0:
                loss = loss + l2_reg * 0.5 * (w * w).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        return w.detach(), b.detach()

    def _fit_ista(self, X_t, y_t, l1_reg):
        n, p = X_t.shape

        # Estimate Lipschitz constant via power iteration.
        # For logistic loss, L = sigma_max(X)^2 / (4n).
        with torch.no_grad():
            v = torch.randn(p, device=X_t.device)
            v = v / v.norm()
            for _ in range(30):
                v = X_t.T @ (X_t @ v)
                sigma_sq = v.norm()
                v = v / sigma_sq
            L = float(sigma_sq) / (4 * n)

        step = 1.0 / max(L, 1e-8)
        threshold = step * l1_reg

        w = torch.zeros(p, device=X_t.device)
        b = torch.zeros(1, device=X_t.device)

        for _ in range(self.max_iter * 10):
            w_var = w.detach().requires_grad_(True)
            b_var = b.detach().requires_grad_(True)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                X_t @ w_var + b_var, y_t, reduction="mean"
            )
            loss.backward()

            with torch.no_grad():
                w_half = w - step * w_var.grad
                w = torch.sign(w_half) * torch.clamp(w_half.abs() - threshold, min=0.0)
                b = b - step * b_var.grad

        return w, b

    def predict_proba(self, X):
        if not hasattr(self, "w_"):
            raise RuntimeError("TorchLogisticClassifier must be fit before prediction.")
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X, device=self.device_)
        with torch.no_grad():
            p1 = torch.sigmoid(X_t @ self.w_ + self.b_).cpu().numpy().ravel()
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def generate_base_data(n_samples=1000, n_informative=10, random_state=42):
    """
    Generates a clean, baseline dataset where ALL features are informative.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_informative,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=random_state
    )
    return X, y


def inject_pure_noise(X, n_noise_features, random_state=42):
    """
    Experiment 1: Injects completely uninformative, random Gaussian noise.
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    # Generate random noise with standard normal distribution
    noise = rng.standard_normal((n_samples, n_noise_features))

    # Concatenate noise to the original features
    X_noisy = np.hstack((X, noise))
    return X_noisy


def inject_low_rank_features(X, n_added_features, method='concatenate', only_projected=False, random_state=42):
    """
    Experiment 2: Injects low-rank features via random projection.

    Parameters:
    - method: 'concatenate' keeps original features and adds projections.
              'project' replaces original features with a higher-dim projection.
    - only_projected: If True and method='concatenate', return only the projected
                      features without the original features.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_orig_features = X.shape

    if method == 'concatenate':
        # Create a projection matrix to map original features to new redundant features
        projection_matrix = rng.standard_normal(
            (n_orig_features, n_added_features))

        # Generate the redundant features
        redundant_features = np.dot(X, projection_matrix)

        if only_projected:
            X_low_rank = redundant_features
        else:
            # Stack original features with redundant features
            X_low_rank = np.hstack((X, redundant_features))

    elif method == 'project':
        # Project original features into a strictly higher dimensional space
        total_features = n_orig_features + n_added_features
        projection_matrix = rng.standard_normal(
            (n_orig_features, total_features))

        # The new matrix has higher dimensions but the same rank as the original
        X_low_rank = np.dot(X, projection_matrix)

    else:
        raise ValueError("Method must be 'concatenate' or 'project'")

    return X_low_rank

def inject_sparse_low_rank_features(X, n_added_features, nnz_per_feature=3, only_projected=False, random_state=42):
    """
    Injects low-rank features via a *sparse* random projection.

    Unlike inject_low_rank_features (which uses a dense projection matrix so
    every added feature is a linear combination of ALL original features), here
    each added feature is a weighted sum of only `nnz_per_feature` randomly
    chosen original features.  The result is still low-rank (at most
    rank(X) additional rank contribution) but the support of each new feature
    is local, mimicking the kind of sparse correlations that appear in real
    tabular data.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_orig_features)
    n_added_features : int
        Number of new sparse-linear-combination features to append.
    nnz_per_feature : int
        Number of non-zero weights per added feature (must be <=
        n_orig_features).
    only_projected : bool
        If True, return only the sparse projected features without the originals.
    random_state : int
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_orig_features = X.shape
    nnz = min(nnz_per_feature, n_orig_features)

    # Build a sparse projection matrix (n_orig_features x n_added_features)
    proj = np.zeros((n_orig_features, n_added_features), dtype=np.float64)
    for j in range(n_added_features):
        chosen = rng.choice(n_orig_features, size=nnz, replace=False)
        weights = rng.standard_normal(nnz)
        proj[chosen, j] = weights

    sparse_features = X @ proj  # (n_samples, n_added_features)
    if only_projected:
        return sparse_features
    return np.hstack((X, sparse_features))


def inject_low_signal_features(X, y, n_weak_features, signal_strength=0.1, only_projected=False, random_state=42):
    """
    Experiment 3: Injects 'weak' features that have a slight correlation with y.

    Parameters:
    - signal_strength: Float (0 to 1). Higher means more correlation with y.
                       At 0.1, the feature is mostly noise but has a slight 'lean'.
    - only_projected: If True, return only the weak features without the originals.
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    # Standardize y to use as a base for the signal (handles binary 0/1)
    y_signal = (y - np.mean(y)) / (np.std(y) + 1e-8)

    weak_features = []
    for _ in range(n_weak_features):
        # Create a feature: signal_strength * target + (1 - signal_strength) * noise
        noise = rng.standard_normal(n_samples)
        feat = (signal_strength * y_signal) + ((1 - signal_strength) * noise)
        weak_features.append(feat.reshape(-1, 1))

    if only_projected:
        return np.hstack(weak_features)
    return np.hstack([X] + weak_features)


ALL_STRATEGIES = ("noise", "low_rank", "sparse_low_rank", "low_signal")


def build_experiment_datasets(
    X_base,
    y_base,
    noise_levels=(10, 50),
    low_rank_levels=(10, 50),
    sparse_low_rank_levels=(10, 50),
    sparse_nnz=3,
    low_signal_levels=(10, 50),
    signal_strength=0.1,
    only_projected=False,
    strategies=None,
    random_state=42,
):
    """
    Creates multiple feature-quality scenarios from the same base data.

    Parameters
    ----------
    only_projected : bool
        If True, the constructed (projected/weak) features are returned *without*
        the original features — i.e. only the newly constructed columns.
        Has no effect on the "noise" strategy (pure noise never includes signal).
    strategies : sequence of str or None
        Subset of strategies to build. Must be from ALL_STRATEGIES
        ('noise', 'low_rank', 'sparse_low_rank', 'low_signal').
        If None (default), all strategies are built.
    """
    if strategies is None:
        strategies = ALL_STRATEGIES
    unknown = set(strategies) - set(ALL_STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Choose from {ALL_STRATEGIES}.")

    datasets = {"base": X_base}

    if "noise" in strategies:
        for n_noise in noise_levels:
            datasets[f"noise_{n_noise}"] = inject_pure_noise(
                X_base,
                n_noise_features=n_noise,
                random_state=random_state,
            )

    if "low_rank" in strategies:
        key_prefix = "low_rank_proj" if only_projected else "low_rank_concat"
        for n_added in low_rank_levels:
            datasets[f"{key_prefix}_{n_added}"] = inject_low_rank_features(
                X_base,
                n_added_features=n_added,
                method="concatenate",
                only_projected=only_projected,
                random_state=random_state,
            )

    if "sparse_low_rank" in strategies:
        key_prefix = "sparse_low_rank_proj" if only_projected else "sparse_low_rank"
        for n_added in sparse_low_rank_levels:
            datasets[f"{key_prefix}_{n_added}"] = inject_sparse_low_rank_features(
                X_base,
                n_added_features=n_added,
                nnz_per_feature=sparse_nnz,
                only_projected=only_projected,
                random_state=random_state,
            )

    if "low_signal" in strategies:
        key_prefix = "low_signal_proj" if only_projected else "low_signal"
        for n_weak in low_signal_levels:
            datasets[f"{key_prefix}_{n_weak}"] = inject_low_signal_features(
                X_base,
                y_base,
                n_weak_features=n_weak,
                signal_strength=signal_strength,
                only_projected=only_projected,
                random_state=random_state,
            )

    return datasets


def reduce_with_pca(X, n_components, random_state=42):
    """
    Reduces features with PCA to n_components.
    """
    if X.shape[1] <= n_components:
        return X

    reducer = PCA(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(X)


def reduce_with_random_projection(X, n_components, random_state=42):
    """
    Reduces features with Gaussian random projection to n_components.
    """
    if X.shape[1] <= n_components:
        return X

    reducer = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def build_reduced_datasets(datasets, n_components, random_state=42):
    """
    Builds PCA and random-projection reduced variants for each dataset.
    """
    reduced_datasets = {}

    for dataset_name, X_data in datasets.items():
        reduced_datasets[f"pca_{dataset_name}"] = reduce_with_pca(
            X_data,
            n_components=n_components,
            random_state=random_state,
        )
        reduced_datasets[f"rp_{dataset_name}"] = reduce_with_random_projection(
            X_data,
            n_components=n_components,
            random_state=random_state,
        )

    return reduced_datasets


def get_model_factories(random_state=42):
    """
    Returns callables that construct fresh model instances per run.
    """
    _use_torch_lr = torch is not None

    if _use_torch_lr:
        _ur_clf = lambda: TorchLogisticClassifier(
            penalty="none", max_iter=200, random_state=random_state
        )
        _l1_clf = lambda: TorchLogisticClassifier(
            penalty="l1", C=1.0, max_iter=200, random_state=random_state
        )
    else:
        _ur_clf = lambda: LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=random_state, C=np.inf
        )
        _l1_clf = lambda: LogisticRegression(
            penalty="elasticnet", l1_ratio=1.0, solver="saga",
            max_iter=2000, random_state=random_state,
        )

    factories = {
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            random_state=random_state,
        ),
        "logistic_regression_ur": lambda: make_pipeline(StandardScaler(), _ur_clf()),
        "logistic_regression_l1": lambda: make_pipeline(StandardScaler(), _l1_clf()),
        "small_mlp": lambda: make_pipeline(
            StandardScaler(),
            TorchMLPClassifier(
                hidden_layer_sizes=(256, 128),
                learning_rate=1e-3,
                batch_size=256,
                max_epochs=200,
                random_state=random_state,
                device="cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
            ),
        ),
    }

    if XGBClassifier is not None:
        _xgb_use_gpu = torch is not None and torch.cuda.is_available()
        if _xgb_use_gpu:
            factories["xgboost"] = lambda: XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device="cuda",
                eval_metric="logloss",
                random_state=random_state,
                verbosity=0,
            )
        else:
            factories["xgboost"] = lambda: XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="logloss",
                random_state=random_state,
                verbosity=0,
            )

    if CatBoostClassifier is not None:
        _cb_use_gpu = torch is not None and torch.cuda.is_available()
        factories["catboost"] = lambda: CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            task_type="GPU" if _cb_use_gpu else "CPU",
            random_seed=random_state,
            verbose=False,
        )

    if TabPFNClassifier is not None:
        _tabpfn_device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        if _tabpfn_factory_available:
            for _preset in PreprocessingPreset:
                factories[f"tabpfn_{_preset.value}"] = (
                    lambda p=_preset, d=_tabpfn_device: build_tabpfn_classifier(
                        preset=p, device=d, random_state=random_state
                    )
                )
        else:
            factories["tabpfn"] = lambda: TabPFNClassifier(
                device=_tabpfn_device,
                random_state=random_state,
            )

    if TabICLClassifier is not None:
        factories["tabicl"] = lambda: TabICLClassifier(
            random_state=random_state,
            device="cuda",
            verbose=False,
            #n_estimators=16,
        )

    return factories


def _entropy_effective_n(weights, n_features):
    """Shared helper: entropy-based effective-n from a weight vector."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights[weights > 0]
    if len(weights) == 0:
        return 1.0, 1.0 / n_features
    p = weights / weights.sum()
    effective_n = float(np.exp(-np.sum(p * np.log(p))))
    return effective_n, effective_n / n_features


def compute_signal_imbalance(X, y, random_state=42):
    """
    Estimates signal imbalance across features using decision tree importances.

    Fits a decision tree on (X, y) and treats the resulting feature importances
    as a probability distribution. Computes the entropy-based effective number
    of predictive features:

        effective_n = exp(-sum(p_i * log(p_i)))

    where p_i is the normalized importance of feature i (zero-importance
    features are excluded).

    Returns
    -------
    effective_n : float
        Ranges from 1 (all signal concentrated in one feature) to n_features
        (signal spread perfectly uniformly).
    normalized_effective_n : float
        effective_n / n_features. Comparable across datasets of different widths.
    """
    tree = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state,
    )
    tree.fit(X, y)
    return _entropy_effective_n(tree.feature_importances_, X.shape[1])


def compute_signal_imbalance_lr(X, y, random_state=42):
    """
    Estimates signal imbalance using L2-regularized logistic regression coefficients.

    Scales X then treats |coef_| as per-feature importance, applying the same
    entropy-based effective-n formula as compute_signal_imbalance.  L2
    regularization keeps all features non-zero, giving a smoother estimate than
    L1.  Uses a PyTorch/LBFGS GPU implementation when CUDA is available, falling
    back to sklearn on CPU otherwise.

    Returns
    -------
    effective_n : float
    normalized_effective_n : float
    """
    if torch is not None and torch.cuda.is_available():
        return _compute_signal_imbalance_lr_torch(X, y, random_state)
    return _compute_signal_imbalance_lr_sklearn(X, y, random_state)


def _compute_signal_imbalance_lr_sklearn(X, y, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state,
    )
    lr.fit(X_scaled, y)
    importances = np.abs(lr.coef_).sum(axis=0)
    return _entropy_effective_n(importances, X.shape[1])


def _compute_signal_imbalance_lr_torch(X, y, random_state=42):
    device = "cuda"
    n_samples, n_features = X.shape

    # Standardize on CPU before moving to GPU.
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_scaled = ((X - mean) / std).astype(np.float32)

    X_t = torch.tensor(X_scaled, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    torch.manual_seed(random_state)
    w = torch.zeros(n_features, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)

    # L2 penalty scaled to match sklearn's C=1: (1/N) * ||w||^2 / 2
    l2_scale = 1.0 / n_samples

    optimizer = torch.optim.LBFGS(
        [w, b],
        max_iter=200,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        logits = X_t @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_t, reduction="mean"
        )
        l2 = l2_scale * 0.5 * (w * w).sum()
        (loss + l2).backward()
        return loss + l2

    optimizer.step(closure)

    importances = w.detach().abs().cpu().numpy()
    return _entropy_effective_n(importances, n_features)


def evaluate_models_on_dataset(X, y, model_factories, random_state=42, test_size=0.3):
    """
    Splits data once and evaluates each model with common metrics.
    """
    import time

    # Shuffle feature columns once so all models see the same randomized order.
    rng = np.random.default_rng(random_state)
    feature_perm = rng.permutation(X.shape[1])
    X = X[:, feature_perm]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    results = []
    for model_name, build_model in model_factories.items():
        model = build_model()
        run_result = {
            "model": model_name,
            "accuracy": np.nan,
            "f1": np.nan,
            "auc": np.nan,
            "fit_seconds": np.nan,
            "predict_seconds": np.nan,
            "error": "",
        }

        try:
            start_fit = time.perf_counter()
            model.fit(X_train, y_train)
            run_result["fit_seconds"] = time.perf_counter() - start_fit

            start_pred = time.perf_counter()
            y_pred = model.predict(X_test)
            run_result["predict_seconds"] = time.perf_counter() - start_pred

            run_result["accuracy"] = accuracy_score(y_test, y_pred)
            run_result["f1"] = f1_score(y_test, y_pred)


            #print("Evaluated model:", model_name)
            #print("Time to fit:", run_result["fit_seconds"])
            #print("Time to predict:", run_result["predict_seconds"])

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    run_result["auc"] = roc_auc_score(y_test, y_proba[:, 1])
                elif y_proba.ndim == 1:
                    run_result["auc"] = roc_auc_score(y_test, y_proba)
        except Exception as exc:
            run_result["error"] = str(exc)

        results.append(run_result)

    return results


def print_results_table(all_results):
    """
    Pretty-prints benchmark results.
    """
    header = (
        f"{'dataset':<22} {'model':<24} {'acc':>8} {'f1':>8} "
        f"{'auc':>8} {'fit_s':>10} {'pred_s':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in all_results:
        print(
            f"{row['dataset']:<22} {row['model']:<24} "
            f"{row['accuracy']:>8.4f} {row['f1']:>8.4f} {row['auc']:>8.4f} "
            f"{row['fit_seconds']:>10.4f} {row['predict_seconds']:>10.4f}"
        )
        if row["error"]:
            print(f"  -> error: {row['error']}")


def save_results_csv(
    all_results,
    output_dir="results",
    split_by_method=False,
    file_suffix="",
):
    """
    Saves benchmark results to CSV.
    If split_by_method is True, saves separate files for pca_ and rp_ datasets.
    Returns a dict of {method: output_file_path} or a single file path.
    """
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seed",
        "signal_imbalance_dt",
        "normalized_signal_imbalance_dt",
        "signal_imbalance_lr",
        "normalized_signal_imbalance_lr",
        "dataset",
        "model",
        "accuracy",
        "f1",
        "auc",
        "fit_seconds",
        "predict_seconds",
        "error",
    ]

    if split_by_method:
        # Split results by reduction method
        no_reduction_results = [
            r for r in all_results
            if not r["dataset"].startswith("pca_")
            and not r["dataset"].startswith("rp_")
        ]
        pca_results = [r for r in all_results if r["dataset"].startswith("pca_")]
        rp_results = [r for r in all_results if r["dataset"].startswith("rp_")]

        saved_paths = {}

        if no_reduction_results:
            base_file = output_dir / f"feature_quality_experiment_results{file_suffix}.csv"
            with base_file.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(no_reduction_results)
            saved_paths["no_reduction"] = base_file

        if pca_results:
            pca_file = output_dir / f"feature_quality_pca_results{file_suffix}.csv"
            with pca_file.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(pca_results)
            saved_paths["pca"] = pca_file

        if rp_results:
            rp_file = output_dir / f"feature_quality_rp_results{file_suffix}.csv"
            with rp_file.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rp_results)
            saved_paths["rp"] = rp_file

        return saved_paths
    else:
        # Save all results to a single file
        output_file = output_dir / f"feature_quality_experiment_results{file_suffix}.csv"
        with output_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        return output_file


def seed_result_paths(output_dir, seed):
    """
    Returns expected per-seed output paths for each method.
    """
    suffix = f"_seed{seed}"
    return {
        "no_reduction": output_dir / f"feature_quality_experiment_results{suffix}.csv",
        "pca": output_dir / f"feature_quality_pca_results{suffix}.csv",
        "rp": output_dir / f"feature_quality_rp_results{suffix}.csv",
    }


def seed_results_exist(output_dir, seed, reduction_only=False):
    """
    Checks whether all expected per-seed output files already exist.
    """
    paths = seed_result_paths(output_dir, seed)
    required = ["pca", "rp"] if reduction_only else ["no_reduction", "pca", "rp"]
    return all(paths[key].exists() for key in required)


# ==========================================
# Example Usage & Sanity Check
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run feature quality experiments with optional dimensionality reduction."
    )
    parser.add_argument(
        "--reduction-only",
        action="store_true",
        help="Only evaluate reduced datasets (PCA and Random Projection), skip original datasets.",
    )
    parser.add_argument(
        "--n-informative",
        type=int,
        default=200,
        help="Number of informative base features.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1200,
        help="Number of samples for base dataset.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of seeds to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Specific seed values to run (e.g. --seeds 42 43 44). "
            "If provided, overrides --random-state and --num-seeds."
        ),
    )
    parser.add_argument(
        "--skip-existing-seeds",
        action="store_true",
        help="Skip seeds that already have per-seed result files saved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where results will be saved.",
    )
    parser.add_argument(
        "--only-projected",
        action="store_true",
        help=(
            "Return only the constructed (projected/weak) features, without "
            "concatenating the original features. No effect on the 'noise' strategy."
        ),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(ALL_STRATEGIES),
        default=None,
        metavar="STRATEGY",
        help=(
            f"Subset of strategies to run. Choices: {list(ALL_STRATEGIES)}. "
            "If omitted, all strategies are run."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=(
            "Subset of models to evaluate (e.g. --models tabpfn_default tabpfn_no_svd xgboost). "
            "If omitted, all available models are run."
        ),
    )
    args = parser.parse_args()

    n_informative_base = args.n_informative
    n_samples = args.n_samples
    base_random_state = args.random_state
    if args.seeds is not None and len(args.seeds) > 0:
        seed_values = args.seeds
        print(f"Running experiment over explicit seeds: {seed_values}")
    else:
        seed_values = [base_random_state + i for i in range(args.num_seeds)]
        print(f"Running experiment over {args.num_seeds} seeds: {seed_values}")

    all_results = []
    for seed_idx, seed in enumerate(tqdm(seed_values, desc="Evaluating seeds")):
        if args.skip_existing_seeds and seed_results_exist(
            args.output_dir,
            seed,
            reduction_only=args.reduction_only,
        ):
            print(f"Skipping seed {seed}: per-seed files already exist.")
            continue

        X_base, y_base = generate_base_data(
            n_samples=n_samples,
            n_informative=n_informative_base,
            random_state=seed,
        )

        signal_imbalance_dt, normalized_signal_imbalance_dt = compute_signal_imbalance(
            X_base, y_base, random_state=seed
        )
        signal_imbalance_lr, normalized_signal_imbalance_lr = compute_signal_imbalance_lr(
            X_base, y_base, random_state=seed
        )

        datasets = build_experiment_datasets(
            X_base,
            y_base,
            noise_levels=(200, 400, 2000),
            low_rank_levels=(100, 250, 500, 1000),
            sparse_low_rank_levels=(200, 400, 1000, 2000),
            sparse_nnz=3,
            low_signal_levels=(200, 400, 1000, 2000),
            signal_strength=0.01,
            only_projected=args.only_projected,
            strategies=args.strategies,
            random_state=seed,
        )

        reduced_datasets = build_reduced_datasets(
            datasets,
            n_components=n_informative_base,
            random_state=seed,
        )

        if seed_idx == 0:
            if not args.reduction_only:
                print("Dataset Summary")
                print("-" * 60)
                for dataset_name, X_data in datasets.items():
                    print(
                        f"{dataset_name:<22} shape={X_data.shape} "
                        f"rank={np.linalg.matrix_rank(X_data)}"
                    )
                print()

            print("Reduced Dataset Summary")
            print("-" * 60)
            for dataset_name, X_data in reduced_datasets.items():
                print(
                    f"{dataset_name:<22} shape={X_data.shape} "
                    f"rank={np.linalg.matrix_rank(X_data)}"
                )
            print()

        if args.reduction_only:
            eval_datasets = reduced_datasets
            eval_desc = f"Seed {seed}: reduced datasets"
        else:
            eval_datasets = {}
            eval_datasets.update(datasets)
            eval_datasets.update(reduced_datasets)
            eval_desc = f"Seed {seed}: all datasets"

        model_factories = get_model_factories(random_state=seed)
        if args.models is not None:
            unknown_models = set(args.models) - set(model_factories)
            if unknown_models:
                print(f"Warning: unknown model(s) requested: {sorted(unknown_models)}. "
                      f"Available: {sorted(model_factories)}")
            model_factories = {k: v for k, v in model_factories.items() if k in args.models}
        if seed_idx == 0:
            if TabICLClassifier is None:
                print("Warning: tabicl is not installed. Skipping TabICL.")
            if XGBClassifier is None:
                print("Warning: xgboost is not installed. Skipping XGBoost.")
            if CatBoostClassifier is None:
                print("Warning: catboost is not installed. Skipping CatBoost.")
            if TabPFNClassifier is None:
                print("Warning: tabpfn is not installed. Skipping TabPFN.")
            print(f"Running models: {list(model_factories)}")

        seed_results = []
        for dataset_name, X_data in tqdm(
            eval_datasets.items(),
            desc=eval_desc,
            leave=False,
        ):
            dataset_results = evaluate_models_on_dataset(
                X_data,
                y_base,
                model_factories=model_factories,
                random_state=seed,
            )
            for row in dataset_results:
                row["seed"] = seed
                row["signal_imbalance_dt"] = signal_imbalance_dt
                row["normalized_signal_imbalance_dt"] = normalized_signal_imbalance_dt
                row["signal_imbalance_lr"] = signal_imbalance_lr
                row["normalized_signal_imbalance_lr"] = normalized_signal_imbalance_lr
                row["dataset"] = dataset_name
            seed_results.extend(dataset_results)

        # Save each seed immediately so interrupted runs can resume later.
        save_results_csv(
            seed_results,
            output_dir=args.output_dir,
            split_by_method=True,
            file_suffix=f"_seed{seed}",
        )
        all_results.extend(seed_results)

    print("Model Benchmark Results")
    print("-" * 60)
    print_results_table(all_results)

    saved_paths = save_results_csv(
        all_results,
        output_dir=args.output_dir,
        split_by_method=True,
    )
    print()
    if isinstance(saved_paths, dict):
        for method, path in saved_paths.items():
            print(f"Saved {method.upper()} results to: {path}")
    else:
        print(f"Saved results to: {saved_paths}")
