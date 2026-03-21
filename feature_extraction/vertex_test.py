from typing import Optional
import google.auth.credentials

from google.oauth2 import service_account

# Path to your downloaded service account JSON key file
# Make sure this path is accessible from your Slurm job
SERVICE_ACCOUNT_KEY_FILE = "multimodal-embeddings-489820-03efef353fa6.json"

import vertexai
# vertex ai sdk
from vertexai.vision_models import Image as VMImage
from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.vision_models import Video as VMVideo
from vertexai.vision_models import VideoSegmentConfig

def init_sample(
    project: Optional[str] = None,
    location: Optional[str] = None,
    experiment: Optional[str] = None,
    staging_bucket: Optional[str] = None,
    credentials: Optional[google.auth.credentials.Credentials] = None,
    encryption_spec_key_name: Optional[str] = None,
):
    
    # Load credentials from the service account key file
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_KEY_FILE
    )

    vertexai.init(
        project=project,
        location=location,
        experiment=experiment,
        staging_bucket=staging_bucket,
        credentials=credentials,
        encryption_spec_key_name=encryption_spec_key_name,
        service_account=service_account,
    )

    mm_embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

    print("Vertex AI initialized successfully with service account credentials!")



init_sample(project="multimodal-embeddings-489820", location="us-central1")