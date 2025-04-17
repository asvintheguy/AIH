"""
Helper functions for the chatbot application.
"""

import time


def print_slow(text, delay=0.05):
    """
    Print text with a slight delay between characters for a typing effect.
    
    Args:
        text (str): Text to print
        delay (float): Delay between characters in seconds
    """
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def deduplicate_datasets(datasets_list):
    """
    Remove duplicate datasets from a list based on reference.
    
    Args:
        datasets_list (list): List of dataset dictionaries
        
    Returns:
        list: Deduplicated dataset list
    """
    seen_refs = set()
    unique_datasets = []
    for dataset in datasets_list:
        if dataset["ref"] not in seen_refs:
            unique_datasets.append(dataset)
            seen_refs.add(dataset["ref"])
    return unique_datasets 