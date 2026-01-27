#Open /home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/tabarena_results/APSFailure.pkl and analyze the results
import argparse
import pickle
import numpy as np
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import textwrap

from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

for results_dir in os.listdir('/home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/tabarena_results/'):
    
    dataset_name = results_dir
    #Check if results file exists
    result_path = os.path.join('/home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/tabarena_results/', results_dir, 'results.pkl')
    if os.path.exists(result_path):
        with open(result_path, 'rb') as f:
            results_dict = pickle.load(f)
            
    min_sim = np.min(cosine_similarity(results_dict['test_to_train_attention_matrices'][0]))

    print(f"Dataset: {dataset_name}, Min Cosine Similarity in Attention Matrix: {min_sim}")