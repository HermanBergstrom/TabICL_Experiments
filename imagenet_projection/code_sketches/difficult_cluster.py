import faiss
import numpy as np
import pickle

class CentroidClusterSampler:
    def __init__(self, cluster_assignments, class_to_images, max_n_classes):
        """
        Args:
            cluster_assignments: dict mapping a cluster ID (int) to a list of class IDs.
                                 Generated offline via K-Means on DINOv3 centroids.
            class_to_images: dict mapping class ID to a list of image paths/feature tensors.
            max_n_classes: The maximum number of classes you will ever request.
        """
        self.class_to_images = class_to_images
        
        # Pre-filter: Only keep clusters dense enough to provide N classes
        self.valid_clusters = {
            c_id: classes for c_id, classes in cluster_assignments.items()
            if len(classes) >= max_n_classes
        }
        self.cluster_keys = list(self.valid_clusters.keys())
        
        print(f"Initialized Cluster Sampler. Found {len(self.cluster_keys)} dense clusters.")

    def sample(self, num_classes, num_images_total):
        """
        Fast O(1) sampling for the training loop.
        """
        # 1. Pick a random dense cluster (highly entangled feature space)
        cluster_id = random.choice(self.cluster_keys)
        
        # 2. Sample N classes from this cluster
        sampled_classes = random.sample(self.valid_clusters[cluster_id], num_classes)
        
        # 3. Sample images 
        images_per_class = num_images_total // num_classes
        
        batch_images = []
        batch_labels = []
        
        for class_id in sampled_classes:
            images = random.sample(self.class_to_images[class_id], images_per_class)
            batch_images.extend(images)
            batch_labels.extend([class_id] * images_per_class)
            
        return batch_images, batch_labels

def generate_multi_clusters(centroids, num_runs=10, k=500):
    """
    centroids: (21841, d) numpy array of mean DINOv3 features per class
    num_runs: How many different clustering 'universes' to create
    k: Number of clusters per run
    """
    d = centroids.shape[1]
    all_runs = []

    for i in range(num_runs):
        kmeans = faiss.Kmeans(d, k, niter=20, verbose=False, seed=42 + i)
        kmeans.train(centroids.astype('float32'))
        
        # Get cluster assignment for every class
        _, labels = kmeans.index.search(centroids.astype('float32'), 1)
        
        # Organize: {cluster_id: [class_idx1, class_idx2, ...]}
        cluster_map = {}
        for class_idx, cluster_id in enumerate(labels.ravel()):
            cluster_map.setdefault(int(cluster_id), []).append(class_idx)
        
        all_runs.append(cluster_map)
        
    return all_runs # Save this with pickle


class MultiRunClusterSampler:
    def __init__(self, cluster_runs, class_to_images):
        """
        cluster_runs: List of dicts from the previous step.
        class_to_images: List of lists (indexed by class_idx) containing image IDs.
        """
        self.cluster_runs = cluster_runs
        self.class_to_images = class_to_images

    def sample(self, num_classes, num_images_per_task):
        # 1. Randomly select which clustering universe to use for this batch
        run_idx = random.randint(0, len(self.cluster_runs) - 1)
        current_run = self.cluster_runs[run_idx]
        
        # 2. Pick a cluster that has enough classes
        valid_clusters = [c for c in current_run.values() if len(c) >= num_classes]
        chosen_cluster = random.choice(valid_clusters)
        
        # 3. Sample classes and their images
        sampled_class_indices = random.sample(chosen_cluster, num_classes)
        images_per_class = num_images_per_task // num_classes
        
        task_features = []
        for c_idx in sampled_class_indices:
            img_ids = random.sample(self.class_to_images[c_idx], images_per_class)
            task_features.append(img_ids) # Logic to fetch DINO features here
            
        return task_features