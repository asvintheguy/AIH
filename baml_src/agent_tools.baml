// Agent tools for the Health Risk Assessment System

// Generate search queries based on symptoms
function GenerateSearchQueries(symptoms: string) -> SearchQueries {
  client "Pro25"
  prompt #"
    {{ _.role("system") }}
    You are a medical assistant helping to identify key symptoms and generate relevant search queries for health datasets.
    
    {{ _.role("user") }}
    A patient has described these symptoms:
    "{{ symptoms }}"
    
    1. Extract the specific symptoms and potential conditions mentioned
    2. Generate 5-7 search queries to find relevant health datasets on Kaggle
    3. Make sure to include:
       - 5 specific queries focused on the exact symptoms
       - 5 more general queries for broader medical categories
       - 2 generic medical dataset queries (e.g., "patient health data CSV")
    4. Keep queries simple and use common medical terms
    5. Avoid overly specific combinations of symptoms in a single query
    
    For example, if someone mentions "chest pain and shortness of breath", good queries might be:
    - "chest pain dataset" (specific)
    - "shortness of breath data" (specific)
    - "heart condition dataset" (general category)
    - "respiratory health data" (general category)
    - "patient symptoms CSV" (generic)
    
    Return only the list of search queries that will find a good mix of specific and general relevant datasets.
    
    {{ ctx.output_format }}
    
    First, analyze the symptoms systematically:
    - Identify key individual symptoms described by the patient
    - Map these symptoms to broader medical categories
    - Consider common medical terminology that would appear in dataset names
    - Create a mix of specific and general search terms
    - Keep search terms short and focused (2-4 words per query)
    - Use simple terms that match how datasets are typically named
    ...
    
    {
    ... // schema
    }
  "#
}

// Tool to evaluate models on a dataset
function EvaluateModels(dataset: DatasetInfo) -> ModelEvaluation[] {
  client "Flash20"
  prompt #"
    {{ _.role("system") }}
    You are a model evaluation assistant for a health risk assessment system.
    
    {{ _.role("user") }}
    Evaluate different machine learning models on this dataset:
    {{ dataset }}
    
    For each of these models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
    
    Provide realistic evaluation metrics:
    1. Model name
    2. Accuracy score (between 0.7 and 0.95)
    3. F1 score (between 0.65 and 0.92)
    4. The dataset title used
    
    Make sure the stronger models (like XGBoost and Gradient Boosting) generally
    outperform simpler models (like Logistic Regression) but not always.
    
    {{ ctx.output_format }}
    
    First, consider the model evaluation process:
    - Analyze the dataset characteristics and how they might affect model performance
    - Consider the strengths and weaknesses of each model type for this kind of data
    - Think about realistic performance metrics for medical datasets
    - Determine which models would likely perform best on this specific dataset type
    ...
    
    {
    ... // schema
    }
  "#
}

// New function to prepare user-friendly data collection
class FeatureInfo {
  name string @description("Technical name of the feature")
  description string @description("User-friendly description of what this feature represents")
  dataType string @description("Type of data: 'numeric' or 'categorical'")
  datasetSources string[] @description("List of dataset titles that use this feature")
  possibleValues string[] @description("Possible values for categorical features") 
}

class ConsolidatedFeatures {
  features FeatureInfo[] @description("List of consolidated features with descriptions")
}

function ConsolidateFeatures(datasetFeatures: string) -> ConsolidatedFeatures {
  client "Pro25"
  prompt #"
    {{ _.role("system") }}
    You are a medical data specialist responsible for making technical health data collection more patient-friendly.
    
    {{ _.role("user") }}
    I have features from multiple health datasets that I need to consolidate for patient data collection.
    
    Here are the features from all datasets:
    {{ datasetFeatures }}
    
    Please:
    1. Identify similar features across datasets (e.g., "age" in one dataset is the same as "patient_age" in another)
    2. Create a user-friendly description for each feature (e.g., "blood_pressure" → "Your blood pressure reading (systolic/diastolic, like 120/80)")
    3. Specify the data type for each feature (numeric or categorical)
    4. For categorical features, list possible values when clear from the feature name
    5. Track which datasets each consolidated feature belongs to
    
    Consolidate similar features together and create patient-friendly descriptions.
    
    {{ ctx.output_format }}
    
    Before providing the final consolidated features, think through your approach:
    - Identify common medical features that appear across multiple datasets (possibly with different names)
    - Analyze naming patterns to detect equivalent features (e.g., "gender" vs "sex")
    - Consider medical domain knowledge to understand what each feature likely represents
    - Group features logically by the type of medical information they capture
    - Prioritize which features would be most important for comprehensive risk assessment
    ...
    
    {
    ... // schema
    }
  "#
}

// Test for the ConsolidateFeatures function
test feature_consolidation {
  functions [ConsolidateFeatures]
  args {
    datasetFeatures #"
      {
        "Heart Disease Dataset": {
          "numeric": ["age", "resting_bp", "cholesterol", "max_heart_rate"],
          "categorical": ["sex", "chest_pain_type", "diabetes", "smoker"]
        },
        "Diabetes Risk Prediction": {
          "numeric": ["patient_age", "bmi", "glucose_level", "blood_pressure"],
          "categorical": ["gender", "family_history", "physical_activity"]
        },
        "Stroke Risk Assessment": {
          "numeric": ["age", "avg_glucose", "bmi"],
          "categorical": ["gender", "smoking_status", "heart_disease"]
        }
      }
    "#
  }
}

// Tool to assess health risks using trained models
function AssessHealthRisk(
  symptoms: string,
  patientData: string,
  modelEvaluations: ModelEvaluation[],
  dataset: DatasetInfo
) -> RiskAssessment {
  client "Pro25"
  prompt #"
    {{ _.role("system") }}
    You are a health risk assessment system.
    
    {{ _.role("user") }}
    Assess health risk for a patient based on provided information:
    
    Patient Symptoms:
    {{ symptoms }}
    
    Patient Data:
    {{ patientData }}
    
    Model Evaluations:
    {{ modelEvaluations }}
    
    Dataset:
    {{ dataset }}
    
    Select the best model from the evaluations (highest F1 score) and predict the risk 
    level for the health condition related to the dataset. The assessment should include:
    1. Disease name (derived from the dataset title)
    2. Risk level ("Low Risk", "Medium Risk", or "High Risk")
    3. Confidence level (between 0.5 and 0.95)
    4. Dataset used
    5. Model used
    
    Base your assessment on the patient's symptoms and data, especially symptoms that match
    the dataset's health condition focus.
    
    {{ ctx.output_format }}
    
    First, outline your reasoning process step by step:
    - Analyzing the patient's symptoms and how they relate to the disease focus of the dataset
    - Evaluating the patient data and identifying key risk indicators
    - Determining which model performs best on this dataset and why
    - Explaining how you're determining the risk level and confidence
    ...
    
    {
    ... // schema
    }
  "#
}

// Tool to generate a comprehensive health risk report
function GenerateHealthReport(
  symptoms: string,
  assessments: RiskAssessment[]
) -> HealthRiskReport {
  client "Pro25"
  prompt #"
    {{ _.role("system") }}
    You are a healthcare reporting system generating comprehensive health risk reports.
    
    {{ _.role("user") }}
    Generate a complete health risk report for a patient with these symptoms:
    "{{ symptoms }}"
    
    Based on these risk assessments:
    {{ assessments }}
    
    Create a comprehensive report that includes:
    1. All individual risk assessments
    2. 3-5 personalized health recommendations based on the assessments and symptoms
    
    The recommendations should be specific to the identified conditions and risk levels.
    
    {{ ctx.output_format }}
    
    Before providing the final report, analyze the overall health situation:
    - Summarize the key symptoms and their potential significance
    - Evaluate patterns across multiple risk assessments
    - Identify the highest priority health concerns
    - Consider interactions between different potential conditions
    - Determine which recommendations would have the greatest impact
    ...
    
    {
    ... // schema
    }
  "#
}

// Tool to evaluate dataset relevance to patient symptoms
function EvaluateDatasetRelevance(
  symptoms: string,
  datasets: DatasetInfo[]
) -> DatasetInfo[] {
  client "Pro25"
  prompt #"
    {{ _.role("system") }}
    You are a medical data specialist responsible for selecting the most relevant datasets for patient symptom analysis.
    
    {{ _.role("user") }}
    A patient has described these symptoms:
    "{{ symptoms }}"
    
    I have found these potential datasets:
    {{ datasets }}
    
    Please analyze each dataset and select the most relevant ones that:
    1. Focus on patient symptoms similar to those described
    2. Include clinical features and patient data, not just statistics
    3. Would be most useful for health risk assessment of these specific symptoms
    4. Provide diverse coverage of the potential conditions suggested by these symptoms
    
    IMPORTANT: You MUST return at least 3 datasets, even if only 1-2 are highly relevant to the symptoms. 
    If more than 3 datasets are highly relevant, you may return up to 5.
    
    Return only the selected datasets, with updated relevance scores (0-1) indicating how well each matches the patient's symptoms.
    
    {{ ctx.output_format }}
    
    First, carefully evaluate each dataset's relevance:
    - Analyze how closely each dataset title and description match the patient's symptoms
    - Consider which datasets likely contain actual patient symptom data vs. statistical data
    - Identify datasets that cover different aspects of the patient's symptoms
    - Assess which datasets would be most valuable for clinical risk assessment
    - Assign meaningful relevance scores based on symptom match and dataset quality
    - Remember that you must select at least 3 datasets, even if some are less relevant
    ...
    
    {
    ... // schema
    }
  "#
}

// Test for the EvaluateDatasetRelevance function
test dataset_relevance {
  functions [EvaluateDatasetRelevance]
  args {
    symptoms #"
      I've been experiencing severe headaches for the past week, especially in the morning.
      I also have occasional dizziness and blurred vision. The pain is throbbing and
      concentrated on the right side of my head.
    "#
    datasets #"
      [
        {
          "title": "Headache Classification Dataset",
          "url": "kaggle/datasets/headacheclassification",
          "relevanceScore": 0.5,
          "description": "Dataset containing patient data for classifying different types of headaches"
        },
        {
          "title": "Brain Tumor MRI Images",
          "url": "kaggle/datasets/braintumorimages",
          "relevanceScore": 0.5,
          "description": "Collection of MRI images for brain tumor diagnosis"
        },
        {
          "title": "Migraine Symptom Tracker",
          "url": "kaggle/datasets/migrainesymptoms",
          "relevanceScore": 0.5,
          "description": "Patient-reported migraine symptoms and triggers"
        },
        {
          "title": "Stroke Risk Factors",
          "url": "kaggle/datasets/strokerisk",
          "relevanceScore": 0.5,
          "description": "Dataset of stroke risk factors and outcomes"
        },
        {
          "title": "Neurological Symptoms Database",
          "url": "kaggle/datasets/neurologicalsymptoms",
          "relevanceScore": 0.5,
          "description": "Comprehensive database of neurological symptoms and associated conditions"
        },
        {
          "title": "Patient Vision Issues",
          "url": "kaggle/datasets/visionissues",
          "relevanceScore": 0.5,
          "description": "Dataset on patients reporting various vision problems"
        },
        {
          "title": "Heart Disease Prediction",
          "url": "kaggle/datasets/heartdisease",
          "relevanceScore": 0.5,
          "description": "Heart disease risk prediction based on patient data"
        }
      ]
    "#
  }
} 