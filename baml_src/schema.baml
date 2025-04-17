// Schema definitions for the Health Risk Assessment System

// Structure for dataset information
class DatasetInfo {
  title string @description("Title of the dataset")
  url string @description("URL to the Kaggle dataset")
  relevanceScore float @description("Score indicating relevance to patient symptoms (0-1)")
  description string? @description("Brief description of what the dataset contains")
}

// Structure for model evaluation results
class ModelEvaluation {
  modelName string @description("Name of the ML model")
  accuracy float @description("Accuracy score of the model")
  f1Score float @description("F1 score of the model")
  datasetTitle string @description("Title of the dataset used")
}

// Structure for risk assessment results
class RiskAssessment {
  disease string @description("Name of the disease or condition")
  riskLevel string @description("Risk level (e.g., 'Low Risk', 'High Risk')")
  confidence float @description("Confidence level of the prediction (0-1)")
  datasetUsed string @description("Dataset used for this assessment")
  modelUsed string @description("Model used for this assessment")
}

// Aggregated result from multiple risk assessments
class HealthRiskReport {
  assessments RiskAssessment[] @description("List of all risk assessments performed")
  recommendations string[] @description("General health recommendations based on assessments")
}

// Search query for datasets
class SearchQueries {
  queries string[] @description("List of search queries to find datasets")
} 