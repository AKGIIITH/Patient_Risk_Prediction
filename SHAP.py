"""
SHAP Analysis Script for Clinical Readmission Prediction Models
This script provides interpretability analysis using SHAP for multiple HuggingFace models
"""

import torch
import shap
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Model configurations
MODELS = {
    'longformer_frozen': 'AKG2/clinical-longformer-readmission-frozen',
    'longformer_stl': 'AKG2/clinical-longformer-readmission-stl',
    'hybrid': 'AKG2/hybrid-readmission-classifier',
    'mtl': 'AKG2/mtl-readmission-clinical',
    'bio_stl': 'AKG2/Bio_clinical-readmission-classifier-stl'
}

class ClinicalReadmissionExplainer:
    def __init__(self, model_name, model_key):
        """Initialize model and tokenizer"""
        self.model_key = model_key
        self.model_name = model_name
        self.device = torch.device('cpu')
        
        print(f"Loading {model_key}...")
        
        # Load model first to check its config
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Get the actual base model name from config
        base_model = self.model.config._name_or_path
        if 'longformer' in base_model.lower() or 'longformer' in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        
        # Resize model embeddings to match tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.model.to(self.device)
        self.model.eval()
                
    def predict(self, texts):
        """Make predictions on clinical texts"""
        # Handle different input types from SHAP
        if isinstance(texts, np.ndarray):
            # SHAP passes masked strings, just treat as regular text
            if texts.ndim == 1 and texts.dtype.kind in ['U', 'S', 'O']:
                texts = [str(t) for t in texts]
            elif texts.ndim > 1:
                texts = [' '.join(str(t) for t in row) for row in texts]
            else:
                texts = [str(texts)]
        elif isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            texts = [str(t) for t in texts] if hasattr(texts, '__iter__') else [str(texts)]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
            return_token_type_ids=False,
            add_special_tokens=True
        )
        
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        return probs.cpu().numpy()  
    def create_shap_explainer(self, background_texts):
        """Create SHAP explainer using background data"""
        # Create a wrapper function for SHAP
        def f(texts):
            if isinstance(texts, str):
                texts = [texts]
            return self.predict(texts)
        
        # Convert texts to token arrays for SHAP
        background_tokens = self.tokenizer(
            background_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='np'
        )['input_ids']
        
        # Use Partition explainer with tokenized input
        masker = shap.maskers.Text(self.tokenizer)
        self.explainer = shap.Explainer(f, masker)
        return self.explainer
    
    def explain_prediction(self, text, class_index=1):
        """Generate SHAP explanation for a single prediction"""
        shap_values = self.explainer([text])
        return shap_values
    
    def plot_text_explanation(self, text, class_index=1, save_path=None):
        """Visualize SHAP values for text"""
        shap_values = self.explain_prediction(text, class_index)
        
        # Text plot
        shap.plots.text(shap_values[0, :, class_index], display=True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved explanation to {save_path}")
    
    def batch_explain(self, texts, class_index=1):
        """Explain multiple texts and return summary statistics"""
        shap_values = self.explainer(texts)
        
        # Get mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values.values[:, :, class_index]).mean(axis=0)
        
        return shap_values, mean_abs_shap
    
    def plot_summary(self, texts, class_index=1, save_path=None):
        """Create summary plot for multiple predictions"""
        shap_values = self.explainer(texts)
        
        # Handle different SHAP value shapes
        if len(shap_values.values.shape) == 3:
            values = shap_values.values[:, :, class_index]
        elif len(shap_values.values.shape) == 2:
            values = shap_values.values
        else:
            values = shap_values.values
        
        # Summary plot
        shap.summary_plot(
            values,
            texts,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def waterfall_plot(self, text, class_index=1, save_path=None):
        """Create waterfall plot for a single prediction"""
        shap_values = self.explain_prediction(text, class_index)
        
        # Handle different SHAP value shapes
        if len(shap_values.values.shape) == 3:
            values = shap_values[0, :, class_index]
        elif len(shap_values.values.shape) == 2:
            values = shap_values[0]
        else:
            values = shap_values
        
        shap.plots.waterfall(values, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


def run_shap_analysis(model_key='longformer_stl', 
                      sample_texts=None,
                      background_texts=None):
    """
    Main function to run SHAP analysis on a specific model
    
    Args:
        model_key: Key from MODELS dict
        sample_texts: List of texts to explain
        background_texts: Background dataset for SHAP
    """
    
    # Default sample texts if none provided
    if sample_texts is None:
        sample_texts = [
            "Patient admitted with acute myocardial infarction. History of diabetes and hypertension. Discharged on aspirin, beta blocker, and ACE inhibitor.",
            "85 year old female with congestive heart failure exacerbation. Multiple previous admissions. Poor medication adherence noted.",
            "Post-operative complications following hip replacement surgery. Developed wound infection requiring IV antibiotics."
        ]
    
    # Use subset of sample texts as background if not provided
    if background_texts is None:
        background_texts = sample_texts[:2]
    
    # Initialize explainer
    model_name = MODELS[model_key]
    explainer = ClinicalReadmissionExplainer(model_name, model_key)
    
    # Create SHAP explainer
    print("\nCreating SHAP explainer...")
    explainer.create_shap_explainer(background_texts)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = explainer.predict(sample_texts)
    
    print("\nPredictions (No Readmission | Readmission):")
    for i, (text, pred) in enumerate(zip(sample_texts, predictions)):
        print(f"\nText {i+1}: {text[:100]}...")
        print(f"Prediction: {pred}")
        print(f"Readmission probability: {pred[1]:.3f}")
    
    # Generate explanations
    print("\n" + "="*80)
    print("GENERATING SHAP EXPLANATIONS")
    print("="*80)
    
    # Explain first prediction in detail
    print("\nDetailed explanation for first text:")
    explainer.waterfall_plot(
        sample_texts[0], 
        class_index=1,
        save_path=f'shap_waterfall_{model_key}.png'
    )
    
    # Summary plot for all texts
    if len(sample_texts) > 1:
        print("\nGenerating summary plot...")
        explainer.plot_summary(
            sample_texts,
            class_index=1,
            save_path=f'shap_summary_{model_key}.png'
        )
    
    return explainer, predictions


def compare_models(sample_text, model_keys=None):
    """
    Compare SHAP explanations across different models
    
    Args:
        sample_text: Single text to explain
        model_keys: List of model keys to compare
    """
    if model_keys is None:
        model_keys = list(MODELS.keys())
    
    results = {}
    
    for key in model_keys:
        print(f"\n{'='*80}")
        print(f"Analyzing with {key}")
        print(f"{'='*80}")
        
        explainer = ClinicalReadmissionExplainer(MODELS[key], key)
        explainer.create_shap_explainer([sample_text])
        
        # Get prediction
        pred = explainer.predict([sample_text])[0]
        
        # Get SHAP values
        shap_values = explainer.explain_prediction(sample_text, class_index=1)
        
        results[key] = {
            'prediction': pred,
            'shap_values': shap_values,
            'explainer': explainer
        }
        
        print(f"Readmission probability: {pred[1]:.3f}")
    
    return results


# Example usage
if __name__ == "__main__":
    print("SHAP Analysis for Clinical Readmission Models")
    print("=" * 80)
    
    # Example 1: Analyze single model
    print("\n## EXAMPLE 1: Single Model Analysis ##\n")
    
    explainer, predictions = run_shap_analysis(
        model_key='longformer_stl',
        sample_texts=[
            "Patient with COPD exacerbation, multiple comorbidities including diabetes and heart failure. Frequent ED visits in past 6 months.",
            "Routine appendectomy, uncomplicated recovery, no significant medical history."
        ]
    )
    
    # Example 2: Compare across models
    print("\n\n## EXAMPLE 2: Model Comparison ##\n")
    
    test_text = "Patient admitted with pneumonia. History of COPD and recent hospitalization 2 weeks ago. Lives alone with limited support."
    
    comparison_results = compare_models(
        test_text,
        model_keys=['longformer_stl', 'hybrid', 'mtl']
    )
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    for model_key, results in comparison_results.items():
        pred = results['prediction']
        print(f"{model_key:30s}: Readmission Risk = {pred[1]:.3f}")
    
    print("\n✓ Analysis complete! Check the saved plots.")