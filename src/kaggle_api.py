"""
Kaggle API integration for dataset search and download.
"""

import os
import shutil
from urllib.parse import urlparse
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

# Load environment variables
load_dotenv()
DOWNLOAD_PATH = os.getenv("DOWNLOAD_PATH", "kaggle_dataset")
MAX_DATASET_SIZE_MB = float(os.getenv("MAX_DATASET_SIZE_MB", "300"))  # 300MB default limit


def initialize_kaggle_api():
    """
    Initialize and authenticate the Kaggle API.
    
    Returns:
        KaggleApi: Authenticated Kaggle API object
    """
    api = KaggleApi()
    api.authenticate()
    return api


def get_dataset_size_mb(api, dataset_slug):
    """
    Get the size of a Kaggle dataset in megabytes.
    
    Args:
        api (KaggleApi): Authenticated Kaggle API object
        dataset_slug (str): Dataset slug in format 'owner/dataset-name'
        
    Returns:
        float: Size of the dataset in MB, or -1 if size cannot be determined
    """
    try:
        # Get dataset metadata
        dataset = api.dataset_view(dataset_slug)
        
        # Calculate total size in MB (totalBytes is in bytes)
        if hasattr(dataset, 'totalBytes') and dataset.totalBytes is not None:
            size_mb = dataset.totalBytes / (1024 * 1024)
            return size_mb
        
        # If totalBytes not available, try getting file sizes
        file_list = api.dataset_list_files(dataset_slug).files
        total_size_bytes = sum(file.size for file in file_list if hasattr(file, 'size'))
        size_mb = total_size_bytes / (1024 * 1024)
        return size_mb
    except Exception as e:
        print(f"⚠️ Error determining dataset size: {e}")
        return -1  # Return -1 to indicate unknown size


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
        tuple: Path to CSV file and README content, or (None, None) if dataset is too large
        
    Raises:
        FileNotFoundError: If no CSV files found in the downloaded dataset
        ValueError: If dataset exceeds size limit
    """
    slug = extract_slug_from_url(kaggle_url)
    api = initialize_kaggle_api()
    
    # Check dataset size before downloading
    size_mb = get_dataset_size_mb(api, slug)
    if size_mb > MAX_DATASET_SIZE_MB:
        print(f"⚠️ Dataset size ({size_mb:.2f} MB) exceeds limit of {MAX_DATASET_SIZE_MB} MB. Skipping.")
        raise ValueError(f"Dataset too large: {size_mb:.2f} MB (limit: {MAX_DATASET_SIZE_MB} MB)")
    elif size_mb > 0:
        print(f"📊 Dataset size: {size_mb:.2f} MB")

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
                print(readme_content)
    return csv_path, readme_content 