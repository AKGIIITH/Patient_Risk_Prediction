"""
Multi-Task Learning Classifier for Hospital Readmission Prediction
Optimized for RTX 4080 Super (16GB VRAM)
- Chunked data loading
- Gradient checkpointing
- Memory-efficient training
- Automatic mixed precision (AMP)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
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
import warnings
import gc
import os
warnings.filterwarnings('ignore')

# Set memory optimization flags
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


# ============================================================================
# CONFIGURATION - OPTIMIZED FOR RTX 4080 SUPER
# ============================================================================
class CONFIG:
    DATA_PATH = "/kaggle/input/mimic-iv"  # Change to your data path
    MODEL_NAME = "yikuan8/Clinical-Longformer"
    
    # --- Data Loading (Chunked) ---
    SAMPLE_SIZE = None  # Use full dataset
    CHUNK_SIZE = 50000  # Load data in chunks to avoid memory overflow
    
    # --- Training Configuration ---
    EPOCHS = 3
    
    # --- GPU Optimization (RTX 4080 Super - 16GB) ---
    TRAIN_BATCH_SIZE = 1  # Small batch size per device
    VALID_BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 32  # Effective batch = 1 * 32 = 32
    USE_FP16 = True  # Mixed precision training
    MAX_LENGTH = 2048  # Reduced from 4096 to save memory
    
    # Gradient Checkpointing (saves ~40% memory)
    GRADIENT_CHECKPOINTING = True
    
    # --- Training Hyperparameters ---
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0  # Gradient clipping
    
    # --- Loss Weights ---
    WEIGHT_READMIT = 1.0
    WEIGHT_MORTALITY = 0.5
    WEIGHT_ADM_TYPE = 0.5
    
    # --- Other ---
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    OUTPUT_DIR = "./mtl_readmission_model_4080"
    
    # --- Memory Management ---
    DATALOADER_NUM_WORKERS = 4  # Parallel data loading
    PIN_MEMORY = True
    EMPTY_CACHE_STEPS = 50  # Clear GPU cache every N steps


# ============================================================================
# MEMORY-EFFICIENT DATA LOADING
# ============================================================================
def load_data_in_chunks(filepath, chunksize=50000):
    """Load large CSV files in chunks to avoid memory issues."""
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def load_and_prepare_data(config):
    """Load and merge all data sources with memory optimization."""
    print("=" * 80)
    print("LOADING DATA (CHUNKED FOR MEMORY EFFICIENCY)")
    print("=" * 80)
    
    # Load main admissions file
    print("Loading admissions...")
    admissions = pd.read_csv(f"{config.DATA_PATH}/admissions_with_readmission_labels.csv")
    print(f"✓ Loaded admissions: {admissions.shape}")
    
    # Load discharge notes in chunks
    print("Loading discharge notes (in chunks)...")
    discharge = load_data_in_chunks(
        f"{config.DATA_PATH}/discharge_notes.csv",
        chunksize=config.CHUNK_SIZE
    )
    print(f"✓ Loaded discharge notes: {discharge.shape}")
    
    # Load radiology notes in chunks (large file)
    print("Loading radiology notes (in chunks)...")
    radiology = load_data_in_chunks(
        f"{config.DATA_PATH}/radiology_notes.csv",
        chunksize=config.CHUNK_SIZE
    )
    print(f"✓ Loaded radiology notes: {radiology.shape}")
    
    # Combine notes: concatenate discharge and radiology by hadm_id
    print("\nCombining notes (memory-efficient groupby)...")
    
    # Process discharge notes
    discharge_grouped = discharge.groupby('hadm_id', as_index=False).agg({
        'text': lambda x: ' '.join(x.astype(str))
    })
    discharge_grouped.columns = ['hadm_id', 'discharge_text']
    del discharge
    gc.collect()
    
    # Process radiology notes
    radiology_grouped = radiology.groupby('hadm_id', as_index=False).agg({
        'text': lambda x: ' '.join(x.astype(str))
    })
    radiology_grouped.columns = ['hadm_id', 'radiology_text']
    del radiology
    gc.collect()
    
    # Merge notes
    print("Merging notes...")
    notes_combined = discharge_grouped.merge(
        radiology_grouped, on='hadm_id', how='outer'
    )
    del discharge_grouped, radiology_grouped
    gc.collect()
    
    # Combine all text
    notes_combined['combined_text'] = (
        notes_combined['discharge_text'].fillna('') + ' ' + 
        notes_combined['radiology_text'].fillna('')
    )
    notes_combined['combined_text'] = notes_combined['combined_text'].str.strip()
    
    # Drop intermediate columns
    notes_combined = notes_combined[['hadm_id', 'combined_text']]
    
    # Merge with admissions
    print("Merging with admissions...")
    df = admissions.merge(notes_combined, on='hadm_id', how='left')
    del admissions, notes_combined
    gc.collect()
    
    # Use combined_text
    df['final_text'] = df['combined_text'].fillna('')
    df = df.drop(['combined_text'], axis=1)
    
    print(f"✓ Final merged data: {df.shape}")
    print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Sample data if needed
    if config.SAMPLE_SIZE is not None:
        df = df.sample(n=min(config.SAMPLE_SIZE, len(df)), 
                       random_state=config.RANDOM_SEED)
        print(f"✓ Sampled {len(df)} rows for testing")
    
    # Prepare labels
    print("\nPreparing labels...")
    df['readmitted_30day'] = df['readmitted_30day'].astype(int)
    df['hospital_expire_flag'] = df['hospital_expire_flag'].astype(int)
    
    # Multiclass label
    le = LabelEncoder()
    df['admission_type_encoded'] = le.fit_transform(df['admission_type'])
    num_admission_types = len(le.classes_)
    
    print(f"  - Readmission distribution: {df['readmitted_30day'].value_counts().to_dict()}")
    print(f"  - Mortality distribution: {df['hospital_expire_flag'].value_counts().to_dict()}")
    print(f"  - Admission types: {num_admission_types} classes")
    
    # Calculate class weights
    pos_count = df['readmitted_30day'].sum()
    neg_count = len(df) - pos_count
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"  - Positive weight for readmission: {pos_weight:.2f}")
    
    return df, le, num_admission_types, pos_weight


# ============================================================================
# MEMORY-EFFICIENT DATASET
# ============================================================================
class MTLDataset(Dataset):
    """Memory-efficient dataset with on-the-fly tokenization."""
    
    def __init__(self, texts, readmit_labels, mortality_labels, 
                 adm_type_labels, tokenizer, max_length):
        # Store as numpy arrays (more memory efficient than lists)
        self.texts = np.array(texts)
        self.readmit_labels = np.array(readmit_labels, dtype=np.float32)
        self.mortality_labels = np.array(mortality_labels, dtype=np.float32)
        self.adm_type_labels = np.array(adm_type_labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize on-the-fly (saves memory vs pre-tokenizing)
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
            'readmit_label': torch.tensor(self.readmit_labels[idx], dtype=torch.float),
            'mortality_label': torch.tensor(self.mortality_labels[idx], dtype=torch.float),
            'adm_type_label': torch.tensor(self.adm_type_labels[idx], dtype=torch.long)
        }


# ============================================================================
# MEMORY-OPTIMIZED MODEL
# ============================================================================
class MTLReadmissionModel(nn.Module):
    """MTL model with gradient checkpointing for memory efficiency."""
    
    def __init__(self, model_name, num_admission_types, use_gradient_checkpointing=True):
        super().__init__()
        
        # Load backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Enable gradient checkpointing (saves ~40% memory)
        if use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        
        hidden_size = self.backbone.config.hidden_size
        
        # Task heads
        self.readmit_head = nn.Linear(hidden_size, 1)
        self.mortality_head = nn.Linear(hidden_size, 1)
        self.adm_type_head = nn.Linear(hidden_size, num_admission_types)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask, **kwargs):
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        # Three predictions
        readmit_logits = self.readmit_head(cls_output)
        mortality_logits = self.mortality_head(cls_output)
        adm_type_logits = self.adm_type_head(cls_output)
        
        return {
            'readmit_logits': readmit_logits,
            'mortality_logits': mortality_logits,
            'adm_type_logits': adm_type_logits
        }


# ============================================================================
# CUSTOM TRAINER WITH MEMORY MANAGEMENT
# ============================================================================
class MTLTrainer(Trainer):
    """Custom trainer with multi-task loss and memory management."""
    
    def __init__(self, pos_weight, weight_readmit, weight_mortality, 
                 weight_adm_type, empty_cache_steps=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.tensor([pos_weight])
        self.weight_readmit = weight_readmit
        self.weight_mortality = weight_mortality
        self.weight_adm_type = weight_adm_type
        self.empty_cache_steps = empty_cache_steps
        self.step_count = 0
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.bce_loss_mortality = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Move pos_weight to correct device
        model_device = next(model.parameters()).device
        if self.pos_weight.device != model_device:
            self.pos_weight = self.pos_weight.to(model_device)
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        # Extract labels
        readmit_labels = inputs.pop('readmit_label')
        mortality_labels = inputs.pop('mortality_label')
        adm_type_labels = inputs.pop('adm_type_label')
        inputs.pop('labels', None)
        
        # Forward pass
        outputs = model(**inputs)
        
        # Calculate losses
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
        
        # Combined loss
        total_loss = (
            self.weight_readmit * loss_readmit +
            self.weight_mortality * loss_mortality +
            self.weight_adm_type * loss_adm_type
        )
        
        # Memory management: clear cache periodically
        self.step_count += 1
        if self.step_count % self.empty_cache_steps == 0:
            torch.cuda.empty_cache()
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step for evaluation."""
        readmit_labels = inputs.get('readmit_label')
        mortality_labels = inputs.get('mortality_label')
        adm_type_labels = inputs.get('adm_type_label')
        
        labels = torch.stack([
            readmit_labels,
            mortality_labels,
            adm_type_labels.float()
        ], dim=1)
        
        inputs_for_model = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        
        with torch.no_grad():
            outputs = model(**inputs_for_model)
            loss = self.compute_loss(model, inputs.copy(), return_outputs=False)
        
        predictions = (
            outputs['readmit_logits'].detach().cpu(),
            outputs['mortality_logits'].detach().cpu(),
            outputs['adm_type_logits'].detach().cpu()
        )
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, predictions, labels.detach().cpu())


# ============================================================================
# EVALUATION METRICS
# ============================================================================
def compute_metrics(eval_pred):
    """Compute metrics for main task (readmission)."""
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        readmit_logits = predictions[0]
    else:
        readmit_logits = predictions
    
    readmit_labels = labels[:, 0]
    
    # Sigmoid
    readmit_logits_np = readmit_logits.numpy() if isinstance(readmit_logits, torch.Tensor) else readmit_logits
    readmit_probs = 1 / (1 + np.exp(-readmit_logits_np.squeeze()))
    readmit_preds = (readmit_probs > 0.5).astype(int)
    
    try:
        roc_auc = roc_auc_score(readmit_labels, readmit_probs)
    except ValueError:
        roc_auc = 0.0
    
    return {
        'roc_auc': roc_auc,
        'accuracy': accuracy_score(readmit_labels, readmit_preds),
        'precision': precision_score(readmit_labels, readmit_preds, zero_division=0),
        'recall': recall_score(readmit_labels, readmit_preds, zero_division=0),
        'f1': f1_score(readmit_labels, readmit_preds, zero_division=0)
    }


def custom_data_collator(features):
    """Custom collator for multiple labels."""
    return {
        'input_ids': torch.stack([f['input_ids'] for f in features]),
        'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        'readmit_label': torch.stack([f['readmit_label'] for f in features]),
        'mortality_label': torch.stack([f['mortality_label'] for f in features]),
        'adm_type_label': torch.stack([f['adm_type_label'] for f in features]),
    }


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("MULTI-TASK LEARNING FOR HOSPITAL READMISSION")
    print("OPTIMIZED FOR RTX 4080 SUPER (16GB VRAM)")
    print("=" * 80 + "\n")
    
    # Set seeds
    torch.manual_seed(CONFIG.RANDOM_SEED)
    np.random.seed(CONFIG.RANDOM_SEED)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠ WARNING: No GPU detected! Training will be very slow.")
    
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
    
    # Clear memory
    del df
    gc.collect()
    
    # Load tokenizer
    print("\n" + "=" * 80)
    print("LOADING TOKENIZER")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.MODEL_NAME)
    print(f"✓ Loaded tokenizer: {CONFIG.MODEL_NAME}")
    
    # Create datasets
    print("\n" + "=" * 80)
    print("CREATING DATASETS")
    print("=" * 80)
    train_dataset = MTLDataset(
        texts=train_df['final_text'].values,
        readmit_labels=train_df['readmitted_30day'].values,
        mortality_labels=train_df['hospital_expire_flag'].values,
        adm_type_labels=train_df['admission_type_encoded'].values,
        tokenizer=tokenizer,
        max_length=CONFIG.MAX_LENGTH
    )
    
    val_dataset = MTLDataset(
        texts=val_df['final_text'].values,
        readmit_labels=val_df['readmitted_30day'].values,
        mortality_labels=val_df['hospital_expire_flag'].values,
        adm_type_labels=val_df['admission_type_encoded'].values,
        tokenizer=tokenizer,
        max_length=CONFIG.MAX_LENGTH
    )
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Validation dataset: {len(val_dataset)} samples")
    
    # Clear dataframes
    del train_df, val_df
    gc.collect()
    
    # Initialize model
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    model = MTLReadmissionModel(
        CONFIG.MODEL_NAME,
        num_admission_types,
        use_gradient_checkpointing=CONFIG.GRADIENT_CHECKPOINTING
    )
    print(f"✓ Model created with {num_admission_types} admission type classes")
    if CONFIG.GRADIENT_CHECKPOINTING:
        print("✓ Gradient checkpointing enabled (saves ~40% VRAM)")
    
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
        max_grad_norm=CONFIG.MAX_GRAD_NORM,
        fp16=CONFIG.USE_FP16,
        dataloader_num_workers=CONFIG.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=CONFIG.PIN_MEMORY,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=CONFIG.RANDOM_SEED,
        remove_unused_columns=False,
        optim="adamw_torch",  # Memory-efficient optimizer
    )
    
    # Initialize trainer
    print("\n" + "=" * 80)
    print("INITIALIZING TRAINER")
    print("=" * 80)
    trainer = MTLTrainer(
        pos_weight=pos_weight,
        weight_readmit=CONFIG.WEIGHT_READMIT,
        weight_mortality=CONFIG.WEIGHT_MORTALITY,
        weight_adm_type=CONFIG.WEIGHT_ADM_TYPE,
        empty_cache_steps=CONFIG.EMPTY_CACHE_STEPS,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics,
    )
    print("✓ Trainer initialized")
    print(f"✓ Effective batch size: {CONFIG.TRAIN_BATCH_SIZE * CONFIG.GRADIENT_ACCUMULATION}")
    
    # Train
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    trainer.train()
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (MAIN TASK: 30-DAY READMISSION)")
    print("=" * 80)
    eval_results = trainer.evaluate()
    
    print("\n📊 FINAL RESULTS:")
    print(f"  ROC-AUC:   {eval_results['eval_roc_auc']:.4f}")
    print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall:    {eval_results['eval_recall']:.4f}")
    print(f"  F1 Score:  {eval_results['eval_f1']:.4f}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model saved to: {CONFIG.OUTPUT_DIR}")
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return trainer, eval_results


if __name__ == "__main__":
    trainer, results = main()