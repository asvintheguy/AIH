# Health Risk Assessment System

A streamlined system that analyzes symptoms, finds relevant datasets, trains machine learning models, and generates health risk reports.

## Overview

This system implements a straightforward, effective approach to health risk assessment:

1. **Analyze symptoms** and generate search queries
2. **Find datasets** on Kaggle relevant to the symptoms
3. **Train models** on these datasets
4. **Collect user data** needed for the specific models
5. **Assess health risks** using the trained models
6. **Generate** a comprehensive health risk report

## How It Works

1. The user describes their symptoms
2. The system uses an LLM to generate effective search queries
3. The system searches Kaggle for datasets related to these queries
4. For each relevant dataset found, the system:
   - Downloads and processes the dataset
   - Trains machine learning models on the dataset
   - Asks the user for specific data needed for the models
   - Assesses the user's health risk for the related condition
5. Finally, the system combines all assessments into a comprehensive report

## Architecture

The system is built using:

- **BAML**: For defining LLM tools that generate search queries and assess risks
- **Python**: For the core application logic and machine learning
- **Kaggle API**: For searching and downloading relevant health datasets
- **scikit-learn & XGBoost**: For training predictive models

## Running the System

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- Kaggle API credentials
- OpenAI API key (or other supported LLM provider)

### Environment Setup

1. Set up environment variables:

```bash
# Create a .env file with:
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=the_base_url
BAML_LOG=OFF
```

2. Set up Kaggle API credentials:

```bash
# 1. Create a Kaggle account at https://www.kaggle.com if you don't have one
# 2. Go to your Kaggle account settings (https://www.kaggle.com/account)
# 3. Scroll down to API section and click "Create New API Token"
# 4. This will download a kaggle.json file with your credentials
# 5. Create a .kaggle directory in your home folder if it doesn't exist
# 6. Place the kaggle.json file in this directory:

# For Windows:
mkdir %USERPROFILE%\.kaggle
copy downloaded_kaggle.json %USERPROFILE%\.kaggle\kaggle.json

# For Linux/Mac:
mkdir -p ~/.kaggle
cp downloaded_kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json  # Set appropriate permissions
```

3. Run these commands to setup the project:

```bash
python -m venv .venv # This command will initialize the env
./.venv/scripts/activate # This command will activate it
pip install poetry # This command installs poetry
poetry update # This command will update poetry
poetry install # This command will install
```

4. Run the project
```bash
poetry run chatbot
```

## Project Structure

```
├── baml_src/                # BAML definitions
│   ├── schema.baml          # Data schemas
│   ├── agent_tools.baml     # LLM functions
│   ├── agent_workflow.baml  # Main agent workflow
│   └── clients.baml         # LLM client configurations
├── src/                     # Python source code
│   ├── agent.py             # Core system implementation
│   ├── api/                 # External API integrations
│   ├── data/                # Data handling components
│   ├── models/              # Machine learning models
│   └── utils/               # Utility functions
├── .env                     # Environment variables
└── main.py                  # Main entry point
```

## Note

This is a demonstration system and is not intended for actual medical diagnosis. All health risk assessments are for educational purposes only and should not replace professional medical advice. 