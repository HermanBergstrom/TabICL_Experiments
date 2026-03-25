import torch
import torch.nn.functional as F

#Note for the agent: This should be instantiated once, likely right before initializing the DataLoader or inside the Dataset's __init__ if the dataset manages its own sampling.

class HardNegativeClassSampler:
    
	def __init__(
		self, 
		class_centroids: torch.Tensor, 
		device: torch.device, 
		temperature: float = 0.1
	):
		"""
		Args:
			class_centroids: Tensor of shape (num_classes, feature_dim)
			device: Target device (e.g., 'cuda:0')
			temperature: Controls the strictness of hard sampling. 
							Lower = stricter nearest neighbors.
		"""
		self.device = device
		self.temperature = temperature
		self.num_classes = class_centroids.shape[0]

		# 1. L2 Normalize to prepare for Cosine Similarity
		centroids_norm = F.normalize(class_centroids.to(device), p=2, dim=1)
		
		# 2. Compute the K x K Cosine Similarity Matrix
		# Value range: [-1, 1], where 1 is identical
		self.sim_matrix = torch.matmul(centroids_norm, centroids_norm.T)

    #Note for the agent: Place this method inside the HardNegativeClassSampler class.
	@torch.no_grad()
	def sample_classes(self, k_classes: int, hard_prob: float = 0.5) -> torch.Tensor:
		"""
		Samples a set of classes for a single episode.
		Returns a 1D tensor of class indices.
		"""
		# 1. The 50/50 Coin Flip (Maintain Global Structure)
		if torch.rand(1, device=self.device).item() > hard_prob:
			return torch.randperm(self.num_classes, device=self.device)[:k_classes]

		# 2. Hard Negative Initialization
		selected = torch.zeros(k_classes, dtype=torch.long, device=self.device)

		# Pick the first class uniformly at random
		selected[0] = torch.randint(0, self.num_classes, (1,), device=self.device)

		# Boolean mask tracking which classes are still available to be picked
		available_mask = torch.ones(self.num_classes, dtype=torch.bool, device=self.device)
		available_mask[selected[0]] = False

		# Track the maximum similarity to ANY selected class.
		# Initialize with the similarity row of our first random class.
		current_max_sim = self.sim_matrix[selected[0]].clone()

		# 3. Iterative Sequential Sampling
		for i in range(1, k_classes):
			# Mask out already selected classes by setting their similarity to -inf
			# This ensures e^-inf = 0 probability after softmax
			logits = current_max_sim.clone()
			logits[~available_mask] = -float('inf')

			# Apply temperature and convert to probabilities
			probs = F.softmax(logits / self.temperature, dim=0)

			# Sample the next class based on these probabilities
			next_class = torch.multinomial(probs, num_samples=1)[0]
			selected[i] = next_class

			# Update state for the next loop iteration
			available_mask[next_class] = False
			
			# The Magic Step: Update the running maximum similarity
			# If the new class is closer to candidate X than previous classes were,
			# candidate X's probability of being selected next goes up.
			current_max_sim = torch.maximum(current_max_sim, self.sim_matrix[next_class])

		return selected