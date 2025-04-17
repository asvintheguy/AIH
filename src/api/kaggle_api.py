"""
Kaggle API integration for dataset search and download.
"""

import os
import shutil
from urllib.parse import urlparse
from kaggle.api.kaggle_api_extended import KaggleApi
from src.config import DOWNLOAD_PATH


def initialize_kaggle_api():
    """
    Initialize and authenticate the Kaggle API.
    
    Returns:
        KaggleApi: Authenticated Kaggle API object
    """
    api = KaggleApi()
    api.authenticate()
    return api


def search_kaggle_datasets(query):
    """
    Search Kaggle datasets using the API.
    
    Args:
        query (str): Search query for Kaggle datasets
        
    Returns:
        list: List of dataset information dictionaries
    """
    api = initialize_kaggle_api()
    try:
        print(f"🔎 Searching for datasets with query: {query}")
        search_results = api.dataset_list(search=query, sort_by="hottest", file_type="csv")
        
        if not search_results:
            print("⚠️ No datasets found for this query")
            return []
            
        # Return top dataset with its ref and title (or fewer if none available)
        top_datasets = []
        num_datasets = min(1, len(search_results))
        for ds in search_results[:num_datasets]:
            dataset_info = {
                "ref": ds.ref,
                "title": ds.title,
                "url": f"https://www.kaggle.com/datasets/{ds.ref}"
            }
            top_datasets.append(dataset_info)
        return top_datasets
    except Exception as e:
        print(f"❌ Error during Kaggle API search: {e}")
        return []


def extract_slug_from_url(kaggle_url):
    """
    Extracts the dataset slug from a full Kaggle dataset URL.
    Example:
        'https://www.kaggle.com/datasets/owner/dataset-name'
        -> 'owner/dataset-name'
        
    Args:
        kaggle_url (str): Full Kaggle dataset URL
        
    Returns:
        str: Dataset slug in format 'owner/dataset-name'
        
    Raises:
        ValueError: If URL is not a valid Kaggle dataset URL
    """
    parsed = urlparse(kaggle_url)
    parts = parsed.path.strip("/").split("/")
    if "datasets" in parts:
        idx = parts.index("datasets")
        slug = "/".join(parts[idx + 1:])  # Everything after 'datasets/'
        return slug
    raise ValueError("Invalid Kaggle dataset URL")


def download_kaggle_dataset(kaggle_url, download_path=DOWNLOAD_PATH):
    """
    Download a Kaggle dataset from URL.
    
    Args:
        kaggle_url (str): URL of the Kaggle dataset
        download_path (str): Path to download the dataset
        
    Returns:
        tuple: Path to CSV file and README content
        
    Raises:
        FileNotFoundError: If no CSV files found in the downloaded dataset
    """
    slug = extract_slug_from_url(kaggle_url)
    api = initialize_kaggle_api()

    if os.path.exists(download_path):
        shutil.rmtree(download_path)
    os.makedirs(download_path, exist_ok=True)

    print(f"📥 Downloading: {slug} ...")
    api.dataset_download_files(slug, path=download_path, unzip=True, force=True)
    print("✅ Download complete.")

    # Look for CSV files
    csv_files = [f for f in os.listdir(download_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found.")

    csv_path = os.path.join(download_path, csv_files[0])

    # Look for README or description files
    readme_files = [f for f in os.listdir(download_path) if 'readme' in f.lower()]
    readme_content = ""
    if not readme_files:
        print("⚠️ No README file found.")
    else:
        for readme_file in readme_files:
            readme_path = os.path.join(download_path, readme_file)
            print(f"\n📄 Reading README file: {readme_path}")
            with open(readme_path, 'r', encoding='utf-8') as file:
                readme_content = file.read()
                print(readme_content)  # You can process this content as needed
    return csv_path, readme_content 