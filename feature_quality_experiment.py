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


def inject_low_rank_features(X, n_added_features, method='concatenate', random_state=42):
    """
    Experiment 2: Injects low-rank features via random projection.

    Parameters:
    - method: 'concatenate' keeps original features and adds projections.
              'project' replaces original features with a higher-dim projection.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_orig_features = X.shape

    if method == 'concatenate':
        # Create a projection matrix to map original features to new redundant features
        projection_matrix = rng.standard_normal(
            (n_orig_features, n_added_features))

        # Generate the redundant features
        redundant_features = np.dot(X, projection_matrix)

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

def inject_sparse_low_rank_features(X, n_added_features, nnz_per_feature=3, random_state=42):
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
    return np.hstack((X, sparse_features))


def inject_low_signal_features(X, y, n_weak_features, signal_strength=0.1, random_state=42):
    """
    Experiment 3: Injects 'weak' features that have a slight correlation with y.
    
    Parameters:
    - signal_strength: Float (0 to 1). Higher means more correlation with y.
                       At 0.1, the feature is mostly noise but has a slight 'lean'.
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
    
    X_weak = np.hstack([X] + weak_features)
    return X_weak


def build_experiment_datasets(
    X_base,
    y_base,
    noise_levels=(10, 50),
    low_rank_levels=(10, 50),
    sparse_low_rank_levels=(10, 50),
    sparse_nnz=3,
    low_signal_levels=(10, 50),
    signal_strength=0.1,
    random_state=42,
):
    """
    Creates multiple feature-quality scenarios from the same base data.
    """
    datasets = {"base": X_base}

    for n_noise in noise_levels:
        datasets[f"noise_{n_noise}"] = inject_pure_noise(
            X_base,
            n_noise_features=n_noise,
            random_state=random_state,
        )

    for n_added in low_rank_levels:
        datasets[f"low_rank_concat_{n_added}"] = inject_low_rank_features(
            X_base,
            n_added_features=n_added,
            method="concatenate",
            random_state=random_state,
        )

    for n_added in sparse_low_rank_levels:
        datasets[f"sparse_low_rank_{n_added}"] = inject_sparse_low_rank_features(
            X_base,
            n_added_features=n_added,
            nnz_per_feature=sparse_nnz,
            random_state=random_state,
        )

    for n_weak in low_signal_levels:
        datasets[f"low_signal_{n_weak}"] = inject_low_signal_features(
            X_base,
            y_base,
            n_weak_features=n_weak,
            signal_strength=signal_strength,
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
    factories = {
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            random_state=random_state,
        ),
        "logistic_regression_ur": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=random_state,
                C=np.inf
            ),
        ),
        "logistic_regression_l1": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="elasticnet",
                l1_ratio=1.0,
                solver="saga",
                max_iter=4000,
                random_state=random_state,
            ),
        ),
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

    if TabICLClassifier is not None:
        factories["tabicl"] = lambda: TabICLClassifier(
            random_state=random_state,
            device="cuda",
            verbose=False,
            n_estimators=1,
        )

    return factories


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

        datasets = build_experiment_datasets(
            X_base,
            y_base,
            noise_levels=(50, 100, 200, 400, 2000),
            low_rank_levels=(50, 100, 200, 400, 1000, 2000),
            sparse_low_rank_levels=(50, 100, 200, 400, 1000, 2000),
            sparse_nnz=3,
            low_signal_levels=(50, 100, 200, 400, 1000, 2000),
            signal_strength=0.01,
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
        if seed_idx == 0 and TabICLClassifier is None:
            print("Warning: tabicl is not installed. Skipping TabICL.")

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
