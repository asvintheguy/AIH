"""
Main entry point for the Health Risk Assessment System.
"""

from src.agent import HealthRiskSystem

def main():
    """Main entry point for the application."""
    print("👋 Welcome to the Health Risk Assessment System")
    print("This system analyzes symptoms, finds datasets, trains models, and generates health risk reports.")
    
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