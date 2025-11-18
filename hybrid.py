"""
Phase 3: Hybrid Model - ClinicalBERT Embeddings + Structured Features
Extracts embeddings from trained model and combines with structured data
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc,
    classification_report
)
import lightgbm as lgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class CONFIG:
    # Paths
    DATA_PATH = "./Dataset"
    
    # Your trained model from HuggingFace
    TRAINED_MODEL_HF = "your-username/mtl-readmission-clinical"  # Update this!
    
    # Or use base model if you want to test
    BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
    
    # Which model to use
    USE_TRAINED_MODEL = True  # Set to False to use base model
    
    # Processing
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Structured features to include
    STRUCTURED_FEATURES = [
        'age',
        'gender',
        'insurance',
        'admission_type',
        'admission_location'
    ]
    
    # LightGBM parameters
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'max_depth': 6,
        'min_child_samples': 20
    }
    
    RANDOM_SEED = 42


# ============================================================================
# LOAD YOUR TRAINED MODEL
# ============================================================================
class MTLReadmissionModel(nn.Module):
    """Same architecture as training code."""
    
    def __init__(self, model_name, num_admission_types=5, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        
        self.readmit_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        self.mortality_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.adm_type_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_admission_types)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output  # Return embeddings only


def load_model(config):
    """Load trained model or base model."""
    print(f"Loading model on {config.DEVICE}...")
    
    if config.USE_TRAINED_MODEL:
        model_name = config.TRAINED_MODEL_HF
        print(f"Loading trained model from: {model_name}")
    else:
        model_name = config.BASE_MODEL
        print(f"Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        # Try loading as MTL model first
        model = MTLReadmissionModel(model_name)
    except:
        # Fallback to base model
        print("Loading as base model...")
        model = AutoModel.from_pretrained(model_name)
    
    model = model.to(config.DEVICE)
    model.eval()
    
    return tokenizer, model


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
def load_full_data(config):
    """Load all data including structured features."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load main admissions
    admissions = pd.read_csv(f"{config.DATA_PATH}/admissions_with_readmission_labels.csv")
    print(f"✓ Loaded admissions: {admissions.shape}")
    
    # Load notes
    discharge = pd.read_csv(f"{config.DATA_PATH}/discharge_notes.csv")
    radiology = pd.read_csv(f"{config.DATA_PATH}/radiology_notes.csv")
    
    # Combine notes
    discharge_grouped = discharge.groupby('hadm_id')['text'].apply(
        lambda x: ' '.join(x.astype(str))
    ).reset_index()
    discharge_grouped.columns = ['hadm_id', 'discharge_text']
    
    radiology_grouped = radiology.groupby('hadm_id')['text'].apply(
        lambda x: ' '.join(x.astype(str))
    ).reset_index()
    radiology_grouped.columns = ['hadm_id', 'radiology_text']
    
    notes = discharge_grouped.merge(radiology_grouped, on='hadm_id', how='outer')
    notes['combined_text'] = (
        notes['discharge_text'].fillna('') + ' ' + 
        notes['radiology_text'].fillna('')
    ).str.strip()
    
    # Merge everything
    df = admissions.merge(notes[['hadm_id', 'combined_text']], on='hadm_id', how='left')
    df['text'] = df['combined_text'].fillna('')
    
    # Filter out empty texts
    df = df[df['text'].str.len() > 50].reset_index(drop=True)
    
    # Prepare target
    df['readmitted_30day'] = df['readmitted_30day'].astype(int)
    
    print(f"✓ Final dataset: {df.shape}")
    print(f"  - Readmission rate: {df['readmitted_30day'].mean():.3f}")
    
    return df


def prepare_structured_features(df, config):
    """Prepare and encode structured features."""
    print("\n" + "="*80)
    print("PREPARING STRUCTURED FEATURES")
    print("="*80)
    
    structured_df = pd.DataFrame()
    label_encoders = {}
    
    # Numeric features
    if 'age' in df.columns:
        structured_df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())
    
    # Categorical features - encode them
    categorical_cols = ['gender', 'insurance', 'admission_type', 'admission_location']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            structured_df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"✓ Encoded {col}: {len(le.classes_)} categories")
    
    print(f"\n✓ Structured features shape: {structured_df.shape}")
    print(f"  Features: {list(structured_df.columns)}")
    
    return structured_df.values, label_encoders


# ============================================================================
# EXTRACT EMBEDDINGS
# ============================================================================
def extract_embeddings(texts, tokenizer, model, config):
    """Extract embeddings from clinical notes."""
    print("\n" + "="*80)
    print("EXTRACTING EMBEDDINGS")
    print("="*80)
    
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), config.BATCH_SIZE)):
            batch_texts = texts[i:i + config.BATCH_SIZE]
            
            # Tokenize
            encoding = tokenizer(
                batch_texts,
                truncation=True,
                max_length=config.MAX_LENGTH,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(config.DEVICE)
            attention_mask = encoding['attention_mask'].to(config.DEVICE)
            
            # Get embeddings
            if isinstance(model, MTLReadmissionModel):
                batch_embeddings = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            
            embeddings.append(batch_embeddings.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    print(f"✓ Extracted embeddings: {embeddings.shape}")
    
    return embeddings


# ============================================================================
# BUILD HYBRID MODEL
# ============================================================================
def build_hybrid_model(text_embeddings, structured_features, labels, config):
    """Combine embeddings with structured features and train LightGBM."""
    print("\n" + "="*80)
    print("BUILDING HYBRID MODEL")
    print("="*80)
    
    # Combine features
    X_combined = np.hstack([text_embeddings, structured_features])
    y = labels
    
    print(f"✓ Combined features shape: {X_combined.shape}")
    print(f"  - Text embeddings: {text_embeddings.shape[1]} features")
    print(f"  - Structured features: {structured_features.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, 
        test_size=0.2, 
        random_state=config.RANDOM_SEED,
        stratify=y
    )
    
    print(f"\n✓ Train size: {len(X_train)}")
    print(f"✓ Test size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print(f"\n✓ Class distribution:")
    print(f"  - Negative: {neg_count}")
    print(f"  - Positive: {pos_count}")
    print(f"  - Scale pos weight: {scale_pos_weight:.2f}")
    
    # Train LightGBM
    print("\n" + "="*80)
    print("TRAINING LIGHTGBM")
    print("="*80)
    
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
    
    params = config.LGBM_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Predictions
    y_pred_proba = model.predict(X_test_scaled, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auprc = auc(recall, precision)
    
    print(f"\n📊 HYBRID MODEL PERFORMANCE:")
    print(f"  - ROC-AUC: {roc_auc:.4f}")
    print(f"  - AUPRC: {auprc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted'])}")
    
    # Feature importance
    print("\n" + "="*80)
    print("TOP 20 FEATURE IMPORTANCES")
    print("="*80)
    
    importance = model.feature_importance(importance_type='gain')
    feature_names = [f"embed_{i}" for i in range(text_embeddings.shape[1])] + \
                   [f"struct_{i}" for i in range(structured_features.shape[1])]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(20).to_string(index=False))
    
    # Check if structured features are important
    struct_importance = importance_df[importance_df['feature'].str.startswith('struct_')]['importance'].sum()
    total_importance = importance_df['importance'].sum()
    struct_pct = (struct_importance / total_importance) * 100
    
    print(f"\n✓ Structured features contribute {struct_pct:.1f}% of total importance")
    
    return model, scaler, roc_auc, auprc


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*80)
    print("PHASE 3: HYBRID MODEL")
    print("ClinicalBERT Embeddings + Structured Features + LightGBM")
    print("="*80)
    
    config = CONFIG()
    
    # Set seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Load model and tokenizer
    tokenizer, model = load_model(config)
    
    # Load data
    df = load_full_data(config)
    
    # Sample for testing (remove this line for full dataset)
    # df = df.sample(n=10000, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    # Prepare structured features
    structured_features, label_encoders = prepare_structured_features(df, config)
    
    # Extract embeddings
    embeddings = extract_embeddings(
        df['text'].tolist(),
        tokenizer,
        model,
        config
    )
    
    # Build hybrid model
    hybrid_model, scaler, roc_auc, auprc = build_hybrid_model(
        embeddings,
        structured_features,
        df['readmitted_30day'].values,
        config
    )
    
    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    hybrid_model.save_model('hybrid_lgbm_model.txt')
    
    import joblib
    joblib.dump(scaler, 'hybrid_scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    print("✓ Saved: hybrid_lgbm_model.txt")
    print("✓ Saved: hybrid_scaler.pkl")
    print("✓ Saved: label_encoders.pkl")
    
    print("\n" + "="*80)
    print("🎉 HYBRID MODEL COMPLETE!")
    print("="*80)
    print(f"\nFinal Performance:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")


if __name__ == "__main__":
    main()