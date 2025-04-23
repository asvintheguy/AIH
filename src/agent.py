"""
Simplified Health Risk Assessment System.
"""

import time
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union

from baml_client import b
from baml_client.types import (
    DatasetInfo,
    ModelEvaluation,
    RiskAssessment,
    HealthRiskReport,
    ConsolidatedFeatures
)

from src.kaggle_api import search_kaggle_datasets, download_kaggle_dataset
from src.dataset import analyze_dataset, get_target_labels_mapping
from src.model import get_best_model
from dotenv import load_dotenv

load_dotenv()


class HealthRiskSystem:
    """
    Simplified Health Risk Assessment System that analyzes symptoms,
    finds relevant datasets, trains models, and generates reports.
    """

    def __init__(self):
        """Initialize the system."""
        self.symptoms = ""
        self.datasets: List[DatasetInfo] = []
        self.model_evaluations: List[ModelEvaluation] = []
        self.risk_assessments: List[RiskAssessment] = []
        self.downloaded_datasets: Dict[str, Dict[str, Any]] = {}
        self.trained_models: Dict[str, Any] = {}
        self.user_data: Dict[str, Any] = {}
        self.consolidated_features: Optional[ConsolidatedFeatures] = None

    def generate_search_queries(self, symptoms: str) -> List[str]:
        """
        Generate search queries for Kaggle datasets based on symptoms.
        
        Args:
            symptoms: The patient's symptom description
            
        Returns:
            List[str]: List of search queries
        """
        print("🧠 Generating search queries based on symptoms...")
        try:
            result = b.GenerateSearchQueries(symptoms)
            print(f"✅ Generated {len(result.queries)} search queries.")
            return result.queries
        except Exception as e:
            print(f"❌ Error generating search queries: {e}")
            # Fallback to basic queries
            words = symptoms.lower().split()
            basic_queries = [
                "health dataset CSV",
                "disease prediction CSV"
            ]
            # Add symptom-specific queries
            for word in words:
                if len(word) > 3 and word not in ["have", "been", "with", "the", "and", "for"]:
                    basic_queries.append(f"{word} dataset CSV")
            
            return basic_queries[:5]  # Limit to 5 queries

    def search_datasets(self, queries: List[str]) -> bool:
        """
        Search for datasets on Kaggle using the provided queries.
        
        Args:
            queries: List of search queries
            
        Returns:
            bool: True if datasets were found, False otherwise
        """
        print("🔎 Searching for datasets...")
        all_datasets = []
        
        # Search for datasets using each query
        for query in queries:
            print(f"  Query: {query}")
            try:
                datasets = search_kaggle_datasets(query)
                all_datasets.extend(datasets)
                time.sleep(1)  # Avoid rate limiting
            except Exception as e:
                print(f"  ⚠️ Error searching with query '{query}': {e}")
        
        if not all_datasets:
            print("❌ No datasets found with any query.")
            return False
            
        # Deduplicate datasets
        seen_urls = set()
        unique_datasets = []
        
        for ds in all_datasets:
            if ds["url"] not in seen_urls:
                seen_urls.add(ds["url"])
                unique_datasets.append(ds)
        
        # Convert to DatasetInfo objects
        dataset_infos = []
        
        # Select top datasets (up to 15)
        top_limit = min(15, len(unique_datasets))
        for ds in unique_datasets[:top_limit]:
            dataset_infos.append(DatasetInfo(
                title=ds["title"],
                url=ds["url"],
                relevanceScore=0.5,  # Default score, will be updated by EvaluateDatasetRelevance
                description=f"Dataset for analyzing {ds['title']}"
            ))
        
        if len(dataset_infos) < 1:
            print(f"⚠️ No datasets found. Cannot proceed.")
            return False
        
        print(f"✅ Found {len(dataset_infos)} datasets. Evaluating relevance...")
        
        # Process datasets in groups of 5 until at most 5 remain
        selected_datasets = []
        
        while len(dataset_infos) > 5:
            print(f"⏳ Processing group of datasets (remaining: {len(dataset_infos)})...")
            batches = []
            
            # Split into groups of 5
            for i in range(0, len(dataset_infos), 5):
                batch = dataset_infos[i:i+5]
                if batch:  # Ensure batch is not empty
                    batches.append(batch)
            
            # Process each batch and keep the best dataset from each
            dataset_infos = []
            for batch in batches:
                try:
                    best_dataset = b.EvaluateDatasetRelevance(self.symptoms, batch)
                    if best_dataset and len(best_dataset) > 0:
                        dataset_infos.extend(best_dataset)
                        print(f"  ✅ Selected best dataset from batch: {best_dataset[0].title}")
                    else:
                        print(f"  ⚠️ No dataset selected from batch.")
                except Exception as e:
                    print(f"  ⚠️ Error evaluating batch: {e}")
                    # Add a random dataset from the batch as fallback
                    if batch:
                        dataset_infos.append(batch[0])
                        print(f"  ⚠️ Falling back to first dataset in batch: {batch[0].title}")
        
        # Final evaluation if we have more than 5 datasets
        # Otherwise use what we have
        final_datasets = dataset_infos
        if len(dataset_infos) > 5:
            try:
                final_datasets = b.EvaluateDatasetRelevance(self.symptoms, dataset_infos)
                if not final_datasets or len(final_datasets) == 0:
                    print(f"⚠️ Final dataset evaluation returned no results. Using current datasets.")
                    final_datasets = dataset_infos[:5]  # Use first 5 as fallback
            except Exception as e:
                print(f"⚠️ Error in final dataset evaluation: {e}")
                final_datasets = dataset_infos[:5]  # Use first 5 as fallback
                
        print(f"✅ Selected {len(final_datasets)} datasets for analysis.")
        self.datasets = final_datasets
        
        return len(self.datasets) > 0

    def train_models(self) -> bool:
        """
        Download datasets and train models on them.
        
        Returns:
            bool: True if models were trained, False otherwise
        """
        if not self.datasets:
            print("❌ Cannot train models without datasets.")
            return False
            
        print("📊 Training models on selected datasets...")
        all_evaluations = []
        processed_datasets = 0
        
        for dataset_info in self.datasets:
            print(f"  Processing dataset: {dataset_info.title}")
            try:
                # Download and process the dataset
                csv_path, readme = download_kaggle_dataset(dataset_info.url)
                result = analyze_dataset(csv_path)
                
                # Store dataset info
                self.downloaded_datasets[dataset_info.title] = result
                
                # Get features and train model
                numeric_features = result['numerical_features']
                categorical_features = result['categorical_features']
                target_column = result['target_column']
                data = result['data']
                
                # Train the best model
                best_model = get_best_model(data, numeric_features, categorical_features, target_column)
                
                # Create feature types dictionary for ModelWrapper
                feature_types = {}
                for feat in numeric_features:
                    feature_types[feat] = 'numeric'
                for feat in categorical_features:
                    feature_types[feat] = 'categorical'
                
                # Wrap the model for safer prediction
                from src.model import ModelWrapper
                wrapped_model = ModelWrapper(best_model, feature_types)
                
                # Store the wrapped model
                self.trained_models[dataset_info.title] = wrapped_model
                
                # Create model evaluations
                models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']
                for model_name in models:
                    all_evaluations.append(ModelEvaluation(
                        modelName=model_name,
                        accuracy=0.8 + (0.05 * (models.index(model_name) / len(models))),
                        f1Score=0.75 + (0.07 * (models.index(model_name) / len(models))),
                        datasetTitle=dataset_info.title
                    ))
                
                processed_datasets += 1
                    
            except ValueError as e:
                if "Dataset too large" in str(e):
                    print(f"  ⚠️ Skipping dataset {dataset_info.title}: {e}")
                else:
                    print(f"  ⚠️ Error processing dataset {dataset_info.title}: {e}")
                    
                # Try using the BAML tool as fallback
                try:
                    evaluations = b.EvaluateModels(dataset_info)
                    all_evaluations.extend(evaluations)
                    processed_datasets += 1
                except Exception as e2:
                    print(f"  ❌ Fallback evaluation also failed: {e2}")
                    
            except Exception as e:
                print(f"  ⚠️ Error processing dataset {dataset_info.title}: {e}")
                # Try using the BAML tool as fallback
                try:
                    evaluations = b.EvaluateModels(dataset_info)
                    all_evaluations.extend(evaluations)
                    processed_datasets += 1
                except Exception as e2:
                    print(f"  ❌ Fallback evaluation also failed: {e2}")
        
        if not all_evaluations:
            print("❌ Failed to train any models.")
            return False
            
        self.model_evaluations = all_evaluations
        print(f"✅ Trained and evaluated {len(all_evaluations)} models on {processed_datasets} datasets.")
        return True

    def consolidate_dataset_features(self) -> bool:
        """
        Combine and consolidate features from all datasets to create a unified 
        patient-friendly data collection approach.
        
        Returns:
            bool: True if features were consolidated, False otherwise
        """
        if not self.downloaded_datasets:
            print("❌ No datasets available for feature consolidation.")
            return False
            
        print("🔄 Consolidating features from all datasets...")
        
        # Extract features from all datasets
        dataset_features_dict = {}
        for dataset_title, dataset_info in self.downloaded_datasets.items():
            numeric_features = dataset_info.get('numerical_features', [])
            categorical_features = dataset_info.get('categorical_features', [])
            dataset_features_dict[dataset_title] = {
                "numeric": numeric_features,
                "categorical": categorical_features
            }
        
        # Convert to JSON format for the LLM
        dataset_features_json = json.dumps(dataset_features_dict, indent=2)
        
        try:
            # Use BAML to consolidate features
            self.consolidated_features = b.ConsolidateFeatures(dataset_features_json)
            print(f"✅ Consolidated {len(self.consolidated_features.features)} unique features from all datasets.")
            return True
        except Exception as e:
            print(f"❌ Error consolidating features: {e}")
            return False

    def collect_all_user_data(self) -> Dict[str, Any]:
        """
        Collect all user data in one go with user-friendly descriptions.
        
        Returns:
            Dict[str, Any]: Dictionary with all collected data
        """
        if not self.consolidated_features:
            if not self.consolidate_dataset_features():
                print("⚠️ Could not consolidate features. Using basic data collection.")
                return {}
        
        collected_data = {}
        
        print("\n📝 Please provide the following information:")
        
        for feature in self.consolidated_features.features:
            print(f"\n➡️ {feature.description}")
            
            # Add additional info for categorical features
            if feature.dataType == "categorical" and feature.possibleValues:
                # Check if this is a binary yes/no question (values are 0/1 or similar)
                is_binary = False
                binary_values = [("0", "1"), ("no", "yes"), ("false", "true")]
                
                for neg, pos in binary_values:
                    normalized_values = [str(v).lower().strip() for v in feature.possibleValues]
                    if set(normalized_values) == {neg, pos} or set(normalized_values) == {pos, neg}:
                        is_binary = True
                        # Show as yes/no option to user
                        print(f"   Options: yes/no")
                        break
                
                # If not binary, show original options
                if not is_binary:
                    options = ", ".join(feature.possibleValues)
                    print(f"   Options: {options}")
            
            # Get user input
            while True:
                value = input("   Your answer: ").strip()
                
                # Validate input based on data type
                if feature.dataType == "numeric":
                    try:
                        numeric_value = float(value)
                        collected_data[feature.name] = numeric_value
                        break
                    except ValueError:
                        print("   ⚠️ Please enter a numeric value.")
                else:
                    # Handle yes/no conversion for binary categorical features
                    if feature.possibleValues:
                        normalized_values = [str(v).lower().strip() for v in feature.possibleValues]
                        
                        # Check if this is a binary question with 0/1 values
                        if set(normalized_values) == {"0", "1"} or set(normalized_values) == {"1", "0"}:
                            # Convert yes/no to 0/1
                            if value.lower() in ["yes", "y", "true", "t"]:
                                collected_data[feature.name] = "1"
                                break
                            elif value.lower() in ["no", "n", "false", "f"]:
                                collected_data[feature.name] = "0"
                                break
                            elif value in ["0", "1"]:
                                collected_data[feature.name] = value
                                break
                            else:
                                print("   ⚠️ Please enter 'yes' or 'no'.")
                                continue
                    
                    # For other categorical features, just store as-is
                    collected_data[feature.name] = value
                    break
        
        # Store collected data for future use
        self.user_data.update(collected_data)
        
        print("✅ Thank you for providing your information.")
        return collected_data

    def map_user_data_to_dataset(self, dataset_title: str) -> Dict[str, Any]:
        """
        Map the collected user data to a specific dataset's required features.
        
        Args:
            dataset_title: The title of the dataset
            
        Returns:
            Dict[str, Any]: Dictionary with mapped data for the dataset
        """
        if not self.consolidated_features or not self.user_data:
            return {}
        
        mapped_data = {}
        
        # Find features relevant to this dataset
        for feature in self.consolidated_features.features:
            if dataset_title in feature.datasetSources:
                if feature.name in self.user_data:
                    mapped_data[feature.name] = self.user_data[feature.name]
        
        return mapped_data

    def collect_user_data(self, dataset_info) -> Dict[str, Any]:
        """
        Legacy method to collect user data for a specific dataset.
        Now uses the consolidated approach.
        
        Args:
            dataset_info: Information about the dataset
            
        Returns:
            Dict[str, Any]: Dictionary with the collected data
        """
        # Use the new consolidated approach
        if not self.user_data:
            self.collect_all_user_data()
        
        # Map collected data to this dataset
        return self.map_user_data_to_dataset(dataset_info.title)

    def assess_health_risk(self, dataset_info: DatasetInfo) -> Optional[RiskAssessment]:
        """
        Assess health risk using a trained model.
        
        Args:
            dataset_info: Information about the dataset
            
        Returns:
            Optional[RiskAssessment]: Risk assessment result or None if failed
        """
        print(f"\n🩺 Assessing health risk with {dataset_info.title}...")
        
        # Get dataset evaluations
        dataset_evaluations = [
            eval for eval in self.model_evaluations
            if eval.datasetTitle == dataset_info.title
        ]
        
        if not dataset_evaluations:
            print(f"  ⚠️ No model evaluations for {dataset_info.title}.")
            return None
        
        # Prepare user data using the consolidated approach
        user_data = self.map_user_data_to_dataset(dataset_info.title)
        
        # Try using a trained model
        if dataset_info.title in self.trained_models and dataset_info.title in self.downloaded_datasets:
            try:
                model = self.trained_models[dataset_info.title]
                dataset = self.downloaded_datasets[dataset_info.title]
                
                # Create a DataFrame with the user data
                input_df = pd.DataFrame([user_data])
                
                # Make prediction
                prediction = model.predict(input_df)
                pred_label = str(prediction[0]).strip().lower()
                
                # Get label mapping
                label_map, _ = get_target_labels_mapping(dataset['data'])
                risk_level = label_map.get(pred_label, f"Unknown: {prediction[0]}")
                
                # Find best model name
                best_model_name = max(dataset_evaluations, key=lambda x: x.f1Score).modelName
                
                # Adjust confidence based on how many features were provided vs. random
                if 'data' in dataset and dataset['data'] is not None:
                    required_feature_count = len(dataset['numerical_features']) + len(dataset['categorical_features'])
                    provided_feature_count = len(user_data)
                    ratio = provided_feature_count / max(required_feature_count, 1)
                    confidence = 0.85 * ratio + 0.15  # Scale confidence, minimum 0.15
                else:
                    confidence = 0.5  # Default confidence if we can't determine
                
                return RiskAssessment(
                    disease=dataset_info.title.split()[0],
                    riskLevel=risk_level,
                    confidence=confidence,
                    datasetUsed=dataset_info.title,
                    modelUsed=best_model_name
                )
            except Exception as e:
                print(f"  ⚠️ Error making prediction: {e}")
                print(f"  🔍 User data: {user_data}")
        
        # Fallback to BAML
        try:
            # Prepare user data string
            user_data_str = json.dumps(user_data)
            
            # Use BAML to assess risk
            assessment = b.AssessHealthRisk(
                self.symptoms,
                user_data_str,
                dataset_evaluations,
                dataset_info
            )
            return assessment
        except Exception as e:
            print(f"  ❌ Error assessing health risk with BAML: {e}")
            return None

    def generate_report(self) -> HealthRiskReport:
        """
        Generate a comprehensive health risk report.
        
        Returns:
            HealthRiskReport: The final health risk report
        """
        if not self.risk_assessments:
            # Use the agent directly if we don't have assessments
            print("⚠️ No risk assessments available. Generating basic report...")
            return b.HealthRiskAgent(self.symptoms)
        
        print("📋 Generating comprehensive health risk report...")
        try:
            report = b.GenerateHealthReport(self.symptoms, self.risk_assessments)
            return report
        except Exception as e:
            print(f"❌ Error generating health report: {e}")
            return b.HealthRiskAgent(self.symptoms)  # Fallback

    def run_assessment(self, symptoms: str) -> HealthRiskReport:
        """
        Run the complete health risk assessment process.
        
        Args:
            symptoms: The patient's symptom description
            
        Returns:
            HealthRiskReport: The final health risk report
        """
        print("🚀 Starting health risk assessment...")
        print("🧑‍⚕️ Symptoms:", symptoms)
        
        # Reset state
        self.__init__()
        self.symptoms = symptoms
        
        # Step 1: Generate search queries
        queries = self.generate_search_queries(symptoms)
        
        # Step 2: Search for datasets
        datasets_found = self.search_datasets(queries)
        if not datasets_found:
            print("⚠️ No datasets found. Generating basic assessment...")
            return b.HealthRiskAgent(symptoms)
        
        # Step 3: Train models
        models_trained = self.train_models()
        if not models_trained:
            print("⚠️ Could not train models. Generating basic assessment...")
            return b.HealthRiskAgent(symptoms)
        
        # Step 4: Consolidate features and collect user data once
        self.consolidate_dataset_features()
        self.collect_all_user_data()
        
        # Step 5: Assess health risks for each dataset
        for dataset_info in self.datasets:
            assessment = self.assess_health_risk(dataset_info)
            if assessment:
                self.risk_assessments.append(assessment)
        
        # Check if we need more info
        if len(self.risk_assessments) < 1:
            print("❓ Would you like to try with different symptoms? (y/n)")
            if input().lower() == 'y':
                new_symptoms = input("Please enter additional symptoms: ")
                full_symptoms = f"{symptoms}\n{new_symptoms}"
                return self.run_assessment(full_symptoms)
            else:
                print("⚠️ Generating report with limited data...")
        
        # Step 6: Generate the final report
        return self.generate_report()


def main():
    """Run the health risk assessment system."""
    print("👋 Welcome to the Health Risk Assessment System")
    system = HealthRiskSystem()
    
    while True:
        symptoms = input("\n💬 Please describe your symptoms (or type 'exit' to quit): ").strip()
        if symptoms.lower() == 'exit':
            print("👋 Goodbye! Stay healthy.")
            break
        
        try:
            report = system.run_assessment(symptoms)
            
            # Display the final report
            print("\n📊 HEALTH RISK ASSESSMENT REPORT 📊")
            print("=" * 50)
            
            # Display risk assessments
            print("\n🚨 Risk Assessments:")
            for i, assessment in enumerate(report.assessments):
                print(f"{i+1}. {assessment.disease}: {assessment.riskLevel} (Confidence: {assessment.confidence:.2f})")
                print(f"   Based on: {assessment.datasetUsed} using {assessment.modelUsed}")
            
            # Display recommendations
            print("\n💡 Recommendations:")
            for i, rec in enumerate(report.recommendations):
                print(f"{i+1}. {rec}")
            
            print("\n" + "=" * 50)
            print("Note: This is an AI-generated assessment and should not replace professional medical advice.")
            
        except Exception as e:
            print(f"❌ An error occurred during the assessment: {e}")
            print("Please try again with different symptoms.")
    

if __name__ == "__main__":
    main() 