# Transparency.ai - AI Hiring Bias Detection & Mitigation System

A comprehensive framework for detecting, mitigating, and explaining biases in AI-assisted hiring systems using Large Language Models (LLMs). This project combines cutting-edge bias detection techniques with interpretable AI methods to create more fair and transparent hiring decisions.

## 🎯 Project Overview

This system addresses critical concerns about embedded biases in AI-driven hiring processes by providing:

- **Bias Detection**: Quantitative analysis of hiring biases across demographic groups
- **Bias Mitigation**: Implementation of counterfactual data augmentation and chain-of-thought prompting
- **Explainable AI**: LIME-based explanations for hiring decisions and reasoning faithfulness
- **Interactive Interface**: User-friendly web application for real-time bias analysis

## 🔬 Research Background

AI models trained on historical hiring data often perpetuate or amplify biases related to gender, race, age, and educational background. Our research demonstrates that combining counterfactual data augmentation with structured reasoning prompts can meaningfully reduce bias in LLM-driven hiring decisions while maintaining decision quality.

### Key Findings
- **10%+ reduction** in demographic parity gap for gender bias
- Improved transparency through chain-of-thought reasoning
- Enhanced interpretability using LIME explanations
- Maintained reasonable decision accuracy despite bias mitigation

## 🏗️ System Architecture

The system consists of three main components:

### 1. Bias Analysis Engine
- Implements fairness metrics (demographic parity, equalized odds, predictive rate parity)
- Processes candidate profiles from Stack Overflow Survey (73,462 entries) and Resume datasets (962 entries)
- Generates statistical significance assessments using bootstrapped confidence intervals

### 2. Mitigation Techniques
- **Counterfactual Data Augmentation**: Creates paired profiles with swapped sensitive attributes
- **Chain-of-Thought Prompting**: Structured reasoning process with step-by-step evaluation
- **LIME Integration**: Local interpretable model-agnostic explanations for decision transparency

### 3. Web Interface
- Real-time candidate profile analysis
- Interactive bias visualization dashboards
- Explainable AI decision breakdowns
- Comparative analysis tools

## 🚀 Features

### Bias Detection & Analysis
- Multi-demographic bias detection (gender, race, age, education)
- Statistical significance testing with confidence intervals
- Comparative analysis across different LLMs (Mistral 7B, Ollama 3.2)
- Comprehensive fairness metrics computation

### Explainable AI
- **LIME-based Decision Analysis**: Identifies which features most influence hiring decisions
- **Chain-of-Thought Faithfulness**: Measures how well reasoning steps align with candidate profiles
- **Feature Attribution**: Highlights job-relevant vs. protected attribute influences
- **Instance-level Explanations**: Provides transparent reasoning for individual decisions

### Interactive Frontend
- **Dashboard Overview**: Real-time bias metrics and trends
- **Candidate Analysis**: Upload and analyze individual candidate profiles
- **Comparative Studies**: Side-by-side bias analysis across different models
- **Explanation Viewer**: Interactive LIME explanations and reasoning chains
- **Export Capabilities**: Generate reports and visualizations


### Prerequisites
- Python 3.9+
- Node.js & npm/yarn
- Ollama (for local LLM inference)


## 📊 Usage Examples

### Analyzing Candidate Bias
```python
# Example API call for bias analysis
POST /api/analyze-candidate
{
  "profile": "Software Engineer with 5 years experience in Python and ML",
  "demographics": {
    "gender": "female",
    "age": 28,
    "education": "Masters in Computer Science"
  }
}
```

### Generating Explanations
```python
# Get LIME explanations for a decision
POST /api/explain-decision
{
  "candidate_id": "12345",
  "explanation_type": "lime",
  "include_cot": true
}
```

## 🎨 Frontend Interface
<img width="2530" height="1630" alt="image" src="https://github.com/user-attachments/assets/1bf3a14c-384d-4056-9fc2-4d794049ab21" />

The web interface provides several key screens:

### 1. Dashboard
- Overview of bias metrics across all analyzed candidates
- Real-time statistics and trend visualizations
- Model performance comparisons

### 2. Candidate Analysis
- Upload candidate profiles (resume text, structured data)
- Real-time bias scoring and decision prediction
- Interactive explanation panels

### 3. Bias Visualization
- Demographic parity charts
- Equalized odds comparisons
- Statistical significance indicators

### 4. Explanation Explorer
- LIME feature importance plots
- Chain-of-thought reasoning breakdown
- Faithfulness scores for explanations

## 📈 Evaluation Metrics

### Fairness Metrics
- **Demographic Parity**: Difference in selection rates between groups
- **Equalized Odds**: Group-specific false positive/negative rates
- **Predictive Rate Parity**: Comparative positive predictive values

### Explanation Quality
- **Faithfulness**: How well explanations reflect actual model behavior
- **Relevance**: Alignment between reasoning steps and candidate data
- **Consistency**: Stability of explanations across similar profiles

