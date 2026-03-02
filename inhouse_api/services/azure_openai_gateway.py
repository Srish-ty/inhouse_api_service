import os
import requests
import numpy as np
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


def get_embedding_ada(input_text: str, deployment_name: str = "text-embedding-ada-002", api_version: str = "2023-05-15") -> Optional[np.ndarray]:
    """
    Generate embeddings using Azure OpenAI text-embedding-ada-002.
    
    Args:
        input_text (str): The text to embed
        deployment_name (str): Azure deployment name for ada-002 (default: "text-embedding-ada-002")
        api_version (str): Azure API version (default: "2023-05-15")
    
    Returns:
        np.ndarray: The embedding vector as numpy array, or None if request fails
    """
    azure_endpoint = os.getenv("azure_openai_endpoint")
    azure_key = os.getenv("azure_openai_key")
    
    if not azure_endpoint or not azure_key:
        print("❌ Error: Azure OpenAI credentials not found in .env file")
        return None
    
    url = f"{azure_endpoint}openai/deployments/{deployment_name}/embeddings?api-version={api_version}"
    
    headers = {
        'api-key': azure_key,
        'Content-Type': 'application/json'
    }
    
    payload = {
        "input": input_text
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        embedding = response.json()['data'][0]['embedding']
        return np.array(embedding)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling Azure OpenAI API: {e}, {deployment_name} failed")
        return None


def get_embed_ada(text: str) -> Optional[np.ndarray]:
    """
    Convenience wrapper for get_embedding_ada.
    
    Args:
        text (str): The text to embed
    
    Returns:
        np.ndarray: The embedding vector as numpy array, or None if request fails
    """
    return get_embedding_ada(text)

 