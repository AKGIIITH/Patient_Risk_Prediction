import pandas as pd
from google.cloud import bigquery
import os
from datetime import datetime

# Initialize BigQuery client with explicit project
# Replace 'your-project-id' with your actual GCP project ID
PROJECT_ID = 'deep-sphere-458917-f9'  # CHANGE THIS!

try:
    client = bigquery.Client(project=PROJECT_ID)
    print(f"Connected to project: {client.project}")
except Exception as e:
    print(f"Error initializing BigQuery client: {e}")
    print("\nPlease follow these steps:")
    print("1. Find your project ID at: https://console.cloud.google.com/")
    print("2. Replace 'your-project-id' in the script with your actual project ID")
    print("3. Or set it via: gcloud config set project YOUR_PROJECT_ID")
    raise

print("="*60)
print("MIMIC-IV Readmission Data Download")
print("="*60)

# Create output directory
output_dir = "mimic_readmission_data"
os.makedirs(output_dir, exist_ok=True)
print(f"\nOutput directory: {output_dir}/")

# =====================================================
# 1. Download Admissions with Readmission Labels
# =====================================================
print("\n[1/4] Fetching admissions with readmission labels...")

admissions_query = """
WITH admissions_with_next AS (
  SELECT
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    deathtime,
    admission_type,
    admission_location,
    discharge_location,
    insurance,
    language,
    marital_status,
    race,
    hospital_expire_flag,
    LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_admittime,
    LEAD(hadm_id) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_hadm_id
  FROM
    physionet-data.mimiciv_3_1_hosp.admissions
  WHERE
    dischtime IS NOT NULL
)
SELECT
  subject_id,
  hadm_id,
  admittime,
  dischtime,
  deathtime,
  admission_type,
  admission_location,
  discharge_location,
  insurance,
  language,
  marital_status,
  race,
  hospital_expire_flag,
  next_hadm_id,
  CASE
    WHEN next_admittime IS NOT NULL
      AND DATE_DIFF(DATE(next_admittime), DATE(dischtime), DAY) <= 30
      AND DATE_DIFF(DATE(next_admittime), DATE(dischtime), DAY) >= 0
    THEN 1
    ELSE 0
  END AS readmitted_30day,
  CASE
    WHEN next_admittime IS NOT NULL
      AND DATE_DIFF(DATE(next_admittime), DATE(dischtime), DAY) <= 30
      AND DATE_DIFF(DATE(next_admittime), DATE(dischtime), DAY) >= 0
    THEN DATE_DIFF(DATE(next_admittime), DATE(dischtime), DAY)
    ELSE NULL
  END AS days_to_readmission
FROM
  admissions_with_next
"""

df_admissions = client.query(admissions_query).to_dataframe()
print(f"   ✓ Downloaded {len(df_admissions)} admissions")
print(f"   - Readmitted within 30 days: {df_admissions['readmitted_30day'].sum()}")
print(f"   - Not readmitted: {(df_admissions['readmitted_30day'] == 0).sum()}")

# =====================================================
# 2. Download Discharge Notes
# =====================================================
print("\n[2/4] Fetching discharge notes...")

discharge_query = """
SELECT
  subject_id,
  hadm_id,
  note_type,
  note_seq,
  charttime,
  storetime,
  text
FROM
  physionet-data.mimiciv_note.discharge
WHERE
  text IS NOT NULL
  AND LENGTH(text) > 100
"""

df_discharge = client.query(discharge_query).to_dataframe()
print(f"   ✓ Downloaded {len(df_discharge)} discharge notes")
print(f"   - Unique admissions: {df_discharge['hadm_id'].nunique()}")

# =====================================================
# 3. Download Radiology Notes
# =====================================================
print("\n[3/4] Fetching radiology notes...")

radiology_query = """
SELECT
  subject_id,
  hadm_id,
  note_type,
  note_seq,
  charttime,
  storetime,
  text
FROM
  physionet-data.mimiciv_note.radiology
WHERE
  text IS NOT NULL
  AND LENGTH(text) > 50
"""

df_radiology = client.query(radiology_query).to_dataframe()
print(f"   ✓ Downloaded {len(df_radiology)} radiology notes")
print(f"   - Unique admissions: {df_radiology['hadm_id'].nunique()}")

# =====================================================
# 4. Filter to Only Overlapping Data
# =====================================================
print("\n[4/4] Filtering to overlapping data (admissions with notes)...")

# Combine all notes
df_all_notes = pd.concat([df_discharge, df_radiology], ignore_index=True)
print(f"   - Total notes: {len(df_all_notes)}")

# Get unique admission IDs that have notes
admissions_with_notes = df_all_notes['hadm_id'].unique()
print(f"   - Unique admissions with notes: {len(admissions_with_notes)}")

# Filter admissions to only those with notes
df_admissions_filtered = df_admissions[df_admissions['hadm_id'].isin(admissions_with_notes)].copy()
print(f"   ✓ Filtered to {len(df_admissions_filtered)} admissions with notes")
print(f"   - Readmitted: {df_admissions_filtered['readmitted_30day'].sum()}")
print(f"   - Not readmitted: {(df_admissions_filtered['readmitted_30day'] == 0).sum()}")

# Filter notes to only those admissions
hadm_ids_filtered = set(df_admissions_filtered['hadm_id'])
df_discharge_filtered = df_discharge[df_discharge['hadm_id'].isin(hadm_ids_filtered)].copy()
df_radiology_filtered = df_radiology[df_radiology['hadm_id'].isin(hadm_ids_filtered)].copy()

# =====================================================
# 5. Save to CSV files
# =====================================================
print("\n" + "="*60)
print("SAVING FILES")
print("="*60)

# Save admissions
admissions_file = os.path.join(output_dir, "admissions_with_readmission_labels.csv")
df_admissions_filtered.to_csv(admissions_file, index=False)
print(f"✓ Saved: {admissions_file}")
print(f"  Size: {os.path.getsize(admissions_file) / (1024*1024):.2f} MB")

# Save discharge notes
discharge_file = os.path.join(output_dir, "discharge_notes.csv")
df_discharge_filtered.to_csv(discharge_file, index=False)
print(f"✓ Saved: {discharge_file}")
print(f"  Size: {os.path.getsize(discharge_file) / (1024*1024):.2f} MB")

# Save radiology notes
radiology_file = os.path.join(output_dir, "radiology_notes.csv")
df_radiology_filtered.to_csv(radiology_file, index=False)
print(f"✓ Saved: {radiology_file}")
print(f"  Size: {os.path.getsize(radiology_file) / (1024*1024):.2f} MB")

# =====================================================
# 6. Create Summary Statistics
# =====================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

summary = {
    'Metric': [
        'Total Admissions',
        'Admissions with Notes',
        'Readmitted within 30 days',
        'Not Readmitted',
        'Readmission Rate',
        'Total Discharge Notes',
        'Total Radiology Notes',
        'Total Notes',
        'Avg Notes per Admission',
        'Unique Patients'
    ],
    'Value': [
        len(df_admissions),
        len(df_admissions_filtered),
        df_admissions_filtered['readmitted_30day'].sum(),
        (df_admissions_filtered['readmitted_30day'] == 0).sum(),
        f"{df_admissions_filtered['readmitted_30day'].mean():.2%}",
        len(df_discharge_filtered),
        len(df_radiology_filtered),
        len(df_discharge_filtered) + len(df_radiology_filtered),
        f"{(len(df_discharge_filtered) + len(df_radiology_filtered)) / len(df_admissions_filtered):.2f}",
        df_admissions_filtered['subject_id'].nunique()
    ]
}

df_summary = pd.DataFrame(summary)
summary_file = os.path.join(output_dir, "data_summary.csv")
df_summary.to_csv(summary_file, index=False)

for _, row in df_summary.iterrows():
    print(f"{row['Metric']:.<40} {row['Value']}")

print(f"\n✓ Saved: {summary_file}")

# =====================================================
# 7. Create README
# =====================================================
readme_content = f"""# MIMIC-IV Readmission Data with Clinical Notes

## Download Information
- Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Source: PhysioNet MIMIC-IV Database

## Files Included

### 1. admissions_with_readmission_labels.csv
Contains admission records with readmission labels (only admissions that have associated notes).

Columns:
- subject_id: Patient identifier
- hadm_id: Hospital admission identifier
- admittime: Admission timestamp
- dischtime: Discharge timestamp
- readmitted_30day: Binary label (1=readmitted within 30 days, 0=not readmitted)
- days_to_readmission: Number of days until readmission (NULL if not readmitted)
- Additional demographic and admission details

### 2. discharge_notes.csv
Discharge summary notes for the admissions above.

Columns:
- subject_id: Patient identifier
- hadm_id: Hospital admission identifier
- note_type: Type of note
- charttime: Time note was charted
- text: Full note text

### 3. radiology_notes.csv
Radiology reports for the admissions above.

Columns:
- subject_id: Patient identifier
- hadm_id: Hospital admission identifier
- note_type: Type of radiology report
- charttime: Time note was charted
- text: Full report text

### 4. data_summary.csv
Summary statistics of the downloaded dataset.

## Usage

To load the data in Python:
python
import pandas as pd

# Load admissions with labels
admissions = pd.read_csv('admissions_with_readmission_labels.csv')

# Load notes
discharge_notes = pd.read_csv('discharge_notes.csv')
radiology_notes = pd.read_csv('radiology_notes.csv')

# Merge admissions with notes
merged = admissions.merge(discharge_notes, on=['subject_id', 'hadm_id'], how='left')


## Important Notes
- This dataset only includes admissions that have associated clinical notes
- Readmission label is based on 30-day window from discharge
- Text fields may contain PHI-redacted information (e.g., [*Name*])
- All timestamps are in original MIMIC-IV format

## Citation
Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023).
MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
"""

readme_file = os.path.join(output_dir, "README.md")
with open(readme_file, 'w') as f:
    f.write(readme_content)
print(f"✓ Saved: {readme_file}")

print("\n" + "="*60)
print("DOWNLOAD COMPLETE!")
print("="*60)
print(f"\nAll files saved to: {output_dir}/")
print("\nNext steps:")
print("1. Use the admissions file for readmission labels")
print("2. Link with notes files using subject_id and hadm_id")
print("3. Run your logistic regression model on this data")