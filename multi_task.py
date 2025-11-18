"""
Optimized Multi-Task Learning for Hospital Readmission Prediction
- Balanced sampling for class imbalance
- RTX 4080 Super optimized settings
- Hugging Face Hub integration with checkpointing
- Improved training dynamics
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
class CONFIG:
    # Data paths
    DATA_PATH = "./Dataset"  # Update for local path
    
    # Model selection - using smaller, faster model for RTX 4080 Super
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # Faster than Longformer
    # Alternative: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # Resume from checkpoint (set to your HF model if continuing training)
    RESUME_FROM_CHECKPOINT = None  # e.g., "your-username/mtl-readmission-v1"
    
    # Hugging Face settings
    HF_USERNAME = os.getenv("HF_USERNAME", "your-username")  # Set in .env
    HF_TOKEN = os.getenv("HF_TOKEN")  # Required for pushing to HF
    if HF_TOKEN is None:
        print("⚠ HF_TOKEN not found")
    HF_REPO_NAME = "mtl-readmission-clinical"
    
    # Sampling strategy
    USE_BALANCED_SAMPLING = True
    SAMPLE_SIZE = 150000  # Use full dataset, or set to int for testing
    
    # RTX 4080 Super Optimization (16GB VRAM) - Optimized for 5-6 hour training
    TRAIN_BATCH_SIZE = 12  # Increased for better GPU utilization
    VALID_BATCH_SIZE = 24  # Increased for validation speed
    GRADIENT_ACCUMULATION = 2  # Reduced for more frequent updates
    USE_FP16 = True
    MAX_LENGTH = 512
    
    # Training parameters - Optimized for longer training
    EPOCHS = 5
    LEARNING_RATE = 2e-5  # Slightly reduced for stability over long training
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    WARMUP_STEPS = 300  # Reduced warmup steps
    
    # Loss weights (adjusted for better balance)
    WEIGHT_READMIT = 1.0
    WEIGHT_MORTALITY = 0.3  # Reduced to focus on main task
    WEIGHT_ADM_TYPE = 0.3  # Reduced to focus on main task
    
    # Checkpointing - Evaluate/Save after each complete epoch
    SAVE_STEPS = -1  # Save only at end of epoch
    EVAL_STEPS = -1  # Evaluate only at end of epoch (use epoch strategy)
    LOGGING_STEPS = 50
    SAVE_TOTAL_LIMIT = 3  # Increased to keep more checkpoints
    
    # Other
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    OUTPUT_DIR = "./mtl_readmission_output"


# ============================================================================
# HUGGING FACE AUTHENTICATION
# ============================================================================
def setup_huggingface():
    """Authenticate with Hugging Face."""
    if CONFIG.HF_TOKEN:
        login(token=CONFIG.HF_TOKEN)
        print(f"✓ Logged in to Hugging Face as {CONFIG.HF_USERNAME}")
        return True
    else:
        print("⚠ HF_TOKEN not found. Model won't be pushed to Hub.")
        return False


# ============================================================================
# DATA LOADING WITH BALANCED SAMPLING
# ============================================================================
def load_and_prepare_data(config):
    """Load and merge all data sources with balanced sampling."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Load main admissions file
    admissions = pd.read_csv(f"{config.DATA_PATH}/admissions_with_readmission_labels.csv")
    print(f"✓ Loaded admissions: {admissions.shape}")
    
    # Load discharge notes
    discharge = pd.read_csv(f"{config.DATA_PATH}/discharge_notes.csv")
    print(f"✓ Loaded discharge notes: {discharge.shape}")
    
    # Load radiology notes
    radiology = pd.read_csv(f"{config.DATA_PATH}/radiology_notes.csv")
    print(f"✓ Loaded radiology notes: {radiology.shape}")
    
    # Combine notes
    print("\nCombining notes...")
    discharge_grouped = discharge.groupby('hadm_id')['text'].apply(
        lambda x: ' '.join(x.astype(str))
    ).reset_index()
    discharge_grouped.columns = ['hadm_id', 'discharge_text']
    
    radiology_grouped = radiology.groupby('hadm_id')['text'].apply(
        lambda x: ' '.join(x.astype(str))
    ).reset_index()
    radiology_grouped.columns = ['hadm_id', 'radiology_text']
    
    notes_combined = discharge_grouped.merge(
        radiology_grouped, on='hadm_id', how='outer'
    )
    
    notes_combined['combined_text'] = (
        notes_combined['discharge_text'].fillna('') + ' ' + 
        notes_combined['radiology_text'].fillna('')
    )
    notes_combined['combined_text'] = notes_combined['combined_text'].str.strip()
    
    # Merge with admissions - only keep necessary columns
    df = admissions[['hadm_id', 'readmitted_30day', 'hospital_expire_flag', 'admission_type']].merge(
        notes_combined[['hadm_id', 'combined_text']], 
        on='hadm_id', 
        how='left'
    )
    
    df['final_text'] = df['combined_text'].fillna('')
    
    # Filter out empty texts
    df = df[df['final_text'].str.len() > 50].reset_index(drop=True)
    print(f"✓ After filtering empty texts: {df.shape}")
    
    # Prepare labels
    print("\nPreparing labels...")
    df['readmitted_30day'] = df['readmitted_30day'].astype(int)
    df['hospital_expire_flag'] = df['hospital_expire_flag'].astype(int)
    
    le = LabelEncoder()
    df['admission_type_encoded'] = le.fit_transform(df['admission_type'])
    num_admission_types = len(le.classes_)
    
    # Balanced sampling if enabled
    if config.USE_BALANCED_SAMPLING:
        print("\n🎯 Applying balanced sampling...")
        pos_indices = df[df['readmitted_30day'] == 1].index.tolist()
        neg_indices = df[df['readmitted_30day'] == 0].index.tolist()
        
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        
        print(f"  Original: {n_pos} positive, {n_neg} negative")
        
        # Balance by undersampling majority class
        target_samples = min(n_pos * 4, n_neg)
        
        np.random.seed(config.RANDOM_SEED)
        neg_sampled_indices = np.random.choice(neg_indices, size=target_samples, replace=False)
        balanced_indices = np.concatenate([pos_indices, neg_sampled_indices])
        np.random.shuffle(balanced_indices)
        
        df = df.iloc[balanced_indices].reset_index(drop=True)
        
        print(f"  Balanced: {n_pos} positive, {len(neg_sampled_indices)} negative")
    
    # Sample for testing if specified
    if config.SAMPLE_SIZE is not None:
        df = df.sample(
            n=min(config.SAMPLE_SIZE, len(df)), 
            random_state=config.RANDOM_SEED
        ).reset_index(drop=True)
        print(f"✓ Sampled {len(df)} rows for testing")
    
    print(f"\n✓ Final dataset: {df.shape}")
    print(f"  - Readmission: {df['readmitted_30day'].value_counts().to_dict()}")
    print(f"  - Mortality: {df['hospital_expire_flag'].value_counts().to_dict()}")
    print(f"  - Admission types: {num_admission_types} classes")
    
    # Calculate class weights
    pos_count = df['readmitted_30day'].sum()
    neg_count = len(df) - pos_count
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"  - Positive weight: {pos_weight:.2f}")
    
    # Keep only essential columns to save memory
    df = df[['hadm_id', 'final_text', 'readmitted_30day', 'hospital_expire_flag', 'admission_type_encoded']].copy()
    
    return df, le, num_admission_types, pos_weight


# ============================================================================
# LAZY LOADING DATASET - LOADS ONLY REQUIRED BATCHES
# ============================================================================
class MTLDataset(Dataset):
    """Custom dataset with lazy loading - tokenizes on-demand."""
    
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['final_text'])
        
        # Tokenize on-demand (not stored in memory)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'readmit_label': torch.tensor(row['readmitted_30day'], dtype=torch.float),
            'mortality_label': torch.tensor(row['hospital_expire_flag'], dtype=torch.float),
            'adm_type_label': torch.tensor(row['admission_type_encoded'], dtype=torch.long)
        }


# ============================================================================
# IMPROVED MULTI-TASK MODEL
# ============================================================================
class MTLReadmissionModel(nn.Module):
    """Multi-Task Learning model with improved architecture."""
    
    def __init__(self, model_name, num_admission_types, dropout=0.3):
        super().__init__()
        
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # Enable gradient checkpointing
        self.backbone.gradient_checkpointing = True
        
        # Improved task heads with hidden layers
        self.dropout = nn.Dropout(dropout)
        
        # Readmission head (main task)
        self.readmit_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Mortality head
        self.mortality_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Admission type head
        self.adm_type_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_admission_types)
        )
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing."""
        self.backbone.gradient_checkpointing = True
        if hasattr(self.backbone, 'enable_input_require_grads'):
            self.backbone.enable_input_require_grads()
    
    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        readmit_logits = self.readmit_head(cls_output)
        mortality_logits = self.mortality_head(cls_output)
        adm_type_logits = self.adm_type_head(cls_output)
        
        return {
            'readmit_logits': readmit_logits,
            'mortality_logits': mortality_logits,
            'adm_type_logits': adm_type_logits
        }


# ============================================================================
# CUSTOM TRAINER WITH IMPROVED LOSS
# ============================================================================
class MTLTrainer(Trainer):
    """Custom trainer with multi-task loss and gradient clipping."""
    
    def __init__(self, pos_weight, weight_readmit, weight_mortality, 
                 weight_adm_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.tensor([pos_weight])
        self.weight_readmit = weight_readmit
        self.weight_mortality = weight_mortality
        self.weight_adm_type = weight_adm_type
        
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.bce_loss_mortality = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model_device = next(model.parameters()).device
        if self.pos_weight.device != model_device:
            self.pos_weight = self.pos_weight.to(model_device)
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        readmit_labels = inputs.pop('readmit_label').to(model_device)
        mortality_labels = inputs.pop('mortality_label').to(model_device)
        adm_type_labels = inputs.pop('adm_type_label').to(model_device)
        inputs.pop('labels', None)
        
        outputs = model(**inputs)
        
        loss_readmit = self.bce_loss(
            outputs['readmit_logits'].squeeze(), 
            readmit_labels
        )
        loss_mortality = self.bce_loss_mortality(
            outputs['mortality_logits'].squeeze(), 
            mortality_labels
        )
        loss_adm_type = self.ce_loss(
            outputs['adm_type_logits'], 
            adm_type_labels
        )
        
        total_loss = (
            self.weight_readmit * loss_readmit +
            self.weight_mortality * loss_mortality +
            self.weight_adm_type * loss_adm_type
        )
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model_device = next(model.parameters()).device
        
        readmit_labels = inputs.get('readmit_label').to(model_device)
        mortality_labels = inputs.get('mortality_label').to(model_device)
        adm_type_labels = inputs.get('adm_type_label').to(model_device)
        
        labels = torch.stack([
            readmit_labels,
            mortality_labels,
            adm_type_labels.float()
        ], dim=1)
        
        inputs_for_model = {
            'input_ids': inputs['input_ids'].to(model_device),
            'attention_mask': inputs['attention_mask'].to(model_device)
        }
        
        with torch.no_grad():
            outputs = model(**inputs_for_model)
            loss = self.compute_loss(model, {
                'input_ids': inputs_for_model['input_ids'],
                'attention_mask': inputs_for_model['attention_mask'],
                'readmit_label': readmit_labels,
                'mortality_label': mortality_labels,
                'adm_type_label': adm_type_labels
            }, return_outputs=False)
        
        predictions = (
            outputs['readmit_logits'].detach().cpu(),
            outputs['mortality_logits'].detach().cpu(),
            outputs['adm_type_logits'].detach().cpu()
        )
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, predictions, labels.detach().cpu())


# ============================================================================
# METRICS
# ============================================================================
def compute_metrics(eval_pred):
    """Compute comprehensive metrics for readmission task."""
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        readmit_logits = predictions[0]
    else:
        readmit_logits = predictions
    
    readmit_labels = labels[:, 0]
    
    readmit_logits_np = readmit_logits.numpy() if isinstance(readmit_logits, torch.Tensor) else readmit_logits
    readmit_probs = 1 / (1 + np.exp(-readmit_logits_np.squeeze()))
    readmit_preds = (readmit_probs > 0.5).astype(int)
    
    try:
        roc_auc = roc_auc_score(readmit_labels, readmit_probs)
    except ValueError:
        roc_auc = 0.0
    
    metrics = {
        'roc_auc': roc_auc,
        'accuracy': accuracy_score(readmit_labels, readmit_preds),
        'precision': precision_score(readmit_labels, readmit_preds, zero_division=0),
        'recall': recall_score(readmit_labels, readmit_preds, zero_division=0),
        'f1': f1_score(readmit_labels, readmit_preds, zero_division=0)
    }
    
    return metrics


def custom_data_collator(features):
    """Custom collator for multiple labels."""
    batch = {
        'input_ids': torch.stack([f['input_ids'] for f in features]),
        'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        'readmit_label': torch.stack([f['readmit_label'] for f in features]),
        'mortality_label': torch.stack([f['mortality_label'] for f in features]),
        'adm_type_label': torch.stack([f['adm_type_label'] for f in features]),
    }
    return batch


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("OPTIMIZED MULTI-TASK LEARNING - RTX 4080 SUPER")
    print("Memory-Efficient Streaming Mode - ~5-6 hours")
    print("=" * 80 + "\n")
    
    # Setup Hugging Face
    hf_available = setup_huggingface()
    
    # Set seeds
    torch.manual_seed(CONFIG.RANDOM_SEED)
    np.random.seed(CONFIG.RANDOM_SEED)
    
    # Load data
    df, label_encoder, num_admission_types, pos_weight = load_and_prepare_data(CONFIG)
    
    # Split data
    print("\n" + "=" * 80)
    print("SPLITTING DATA")
    print("=" * 80)
    train_df, val_df = train_test_split(
        df, 
        test_size=CONFIG.TEST_SIZE, 
        random_state=CONFIG.RANDOM_SEED,
        stratify=df['readmitted_30day']
    )
    print(f"✓ Train size: {len(train_df)}")
    print(f"✓ Validation size: {len(val_df)}")
    
    # Load tokenizer and model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    
    model_name = CONFIG.RESUME_FROM_CHECKPOINT or CONFIG.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MTLReadmissionModel(model_name, num_admission_types)
    
    print(f"✓ Loaded: {model_name}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets with lazy loading
    train_dataset = MTLDataset(
        df=train_df,
        tokenizer=tokenizer,
        max_length=CONFIG.MAX_LENGTH
    )
    
    val_dataset = MTLDataset(
        df=val_df,
        tokenizer=tokenizer,
        max_length=CONFIG.MAX_LENGTH
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG.OUTPUT_DIR,
        num_train_epochs=CONFIG.EPOCHS,
        per_device_train_batch_size=CONFIG.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=CONFIG.VALID_BATCH_SIZE,
        gradient_accumulation_steps=CONFIG.GRADIENT_ACCUMULATION,
        learning_rate=CONFIG.LEARNING_RATE,
        weight_decay=CONFIG.WEIGHT_DECAY,
        warmup_ratio=CONFIG.WARMUP_RATIO,
        warmup_steps=CONFIG.WARMUP_STEPS,
        fp16=CONFIG.USE_FP16,
        logging_steps=CONFIG.LOGGING_STEPS,
        eval_strategy="epoch",  # Changed to evaluate at end of each epoch
        save_strategy="epoch",  # Changed to save at end of each epoch
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        save_total_limit=CONFIG.SAVE_TOTAL_LIMIT,
        report_to="none",
        seed=CONFIG.RANDOM_SEED,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        push_to_hub=hf_available,
        hub_model_id=f"{CONFIG.HF_USERNAME}/{CONFIG.HF_REPO_NAME}" if hf_available else None,
        hub_strategy="every_save" if hf_available else None,
        optim="adamw_torch",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )
    
    # Initialize trainer
    trainer = MTLTrainer(
        pos_weight=pos_weight,
        weight_readmit=CONFIG.WEIGHT_READMIT,
        weight_mortality=CONFIG.WEIGHT_MORTALITY,
        weight_adm_type=CONFIG.WEIGHT_ADM_TYPE,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    trainer.train(resume_from_checkpoint=CONFIG.RESUME_FROM_CHECKPOINT)
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    eval_results = trainer.evaluate()
    
    print("\n📊 RESULTS:")
    for key, value in eval_results.items():
        if key.startswith('eval_'):
            print(f"  {key[5:].upper()}: {value:.4f}")
    
    
    # Push to Hub
    if hf_available:
        print("\n" + "=" * 80)
        print("PUSHING TO HUGGING FACE HUB")
        print("=" * 80)
        trainer.push_to_hub(commit_message=f"Training complete - ROC-AUC: {eval_results['eval_roc_auc']:.4f}")
        print(f"✓ Model pushed to: {CONFIG.HF_USERNAME}/{CONFIG.HF_REPO_NAME}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()