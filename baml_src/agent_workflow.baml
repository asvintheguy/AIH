// Main agent workflow for the Health Risk Assessment System

// Main agent that produces a health risk report based on symptoms
function HealthRiskAgent(symptoms: string) -> HealthRiskReport {
  client "Pro25"
  prompt #"
    {{ _.role("system") }}
    You are a healthcare risk assessment agent. Your goal is to analyze a patient's symptoms and provide a health risk assessment.
    
    {{ _.role("user") }}
    A patient has described these symptoms:
    "{{ symptoms }}"
    
    Please provide a detailed health risk assessment based on these symptoms. Include:
    
    1. Potential conditions that match these symptoms
    2. Risk level for each condition (Low, Medium, or High)
    3. Confidence level for each assessment
    4. Specific recommendations based on the identified risks
    
    Be accurate, clear, and helpful without causing unnecessary concern.
    
    {{ ctx.output_format }}
    
    Outline your medical reasoning process before providing the final assessment. 
    Example:
    - Analyzing key symptoms: ...
    - Considering relevant risk factors: ...
    - Evaluating potential conditions that match these symptoms: ...
    ...
    
    {
    ... // schema
    }
  "#
}

// Test for the agent workflow
test basic_workflow {
  functions [HealthRiskAgent]
  args {
    symptoms #"
      I've been feeling extremely tired lately, with frequent urination and constant thirst.
      I've also noticed some blurry vision. My family has a history of diabetes.
    "#
  }
} 