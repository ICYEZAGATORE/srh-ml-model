"""Generate the two Colab-compatible notebooks for the SRH language classifier
(Kinyarwanda vs English), matching the topic/safety classifier conventions:

  1. data/Language_Classifier_data/language_classifier_data_pipeline.ipynb
       Builds the labeled EN/RW dataset from open sources (KINNEWS [MIT],
       SIB-200 [CC-BY-SA-4.0], and the project's own English topic/safety text),
       cleans/balances/splits, and saves a composition figure.
  2. notebooks/language classifier.ipynb
       Trains LogReg / LinearSVM / XGBoost on CHARACTER n-gram TF-IDF (char_wb 2-5),
       evaluates on val, runs the best once on test, saves the pipeline + figures.
"""
import nbformat as nbf

def new():
    return nbf.v4.new_notebook()
def save(nb, path, cells):
    nb['cells'] = cells
    nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                      'language_info': {'name': 'python'}, 'colab': {'provenance': []}}
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print('wrote', path, 'with', len(cells), 'cells')

# ════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — DATA PIPELINE
# ════════════════════════════════════════════════════════════════════════════
p_cells = []
def pmd(s): p_cells.append(nbf.v4.new_markdown_cell(s))
def pcode(s): p_cells.append(nbf.v4.new_code_cell(s))

pmd("""# Language Classifier Dataset Pipeline
**Project:** SRH AI Platform — ALU Capstone (Kinyarwanda–English bilingual)
**Model:** Model 2 — Language Identification (Kinyarwanda vs English)
**Purpose:** Build one labeled EN/RW text dataset by combining open sources, the same
way the topic pipeline combined MedMCQA / AdaptLLM / HealthCareMagic / AfriMedQA.

## Classes
| ID | Language | Source(s) |
|---|---|---|
| 0 | `english` | Project SRH text — topic dataset + safety dataset (already domain-relevant) |
| 1 | `kinyarwanda` | KINNEWS (Kinyarwanda news, **MIT**) + SIB-200 `kin_Latn` (**CC-BY-SA-4.0**) |

**Code-switched / mixed text:** no open, labeled Kinyarwanda–English code-switch corpus
was found on the Hugging Face Hub, so this is a **binary** EN vs RW dataset and
code-switching detection is flagged as **future work** (see final summary).

## Output
`data/Language_Classifier_data/lang_{train,val,test}.csv` + `lang_labels_full.csv` +
`lang_label_map.json`, columns: `text`, `language`, `label` (int 0/1), `source`.""")

pmd("---\n## STEP 0 — Install & import")
pcode("!pip install -q datasets pandas scikit-learn matplotlib seaborn\nprint('deps ready')")
pcode("""import os, re, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

LANG_MAP   = {'english': 0, 'kinyarwanda': 1}
LANG_NAMES = {v: k for k, v in LANG_MAP.items()}

# Anchor all paths to the repo root (walk up for a marker) so the notebook works
# no matter which directory it is launched from (root, notebooks/, data/X/, Colab).
def find_repo_root(start='.'):
    p = os.path.abspath(start)
    while True:
        if (os.path.isdir(os.path.join(p, 'data', 'Topic_Classifier_data'))
                or os.path.isdir(os.path.join(p, '.git'))):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return os.path.abspath(start)
        p = parent

REPO_ROOT = find_repo_root()
DATA_ROOT = os.path.join(REPO_ROOT, 'data')
OUT_DIR   = os.path.join(DATA_ROOT, 'Language_Classifier_data')
os.makedirs(OUT_DIR, exist_ok=True)

MIN_LEN, MAX_LEN = 15, 300   # focus on query-length text; avoids long-article length confound

def clean_text(t):
    if not isinstance(t, str):
        return ''
    t = t.replace('\\ufffd', \"'\")                       # mangled apostrophe -> '
    t = re.sub(r'\\s+', ' ', t).strip()
    t = re.sub(r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]', '', t)
    return t

def prep(series, language, source):
    s = pd.Series(series).dropna().astype(str).map(clean_text)
    df = pd.DataFrame({'text': s})
    df = df[(df['text'].str.len() >= MIN_LEN) & (df['text'].str.len() <= MAX_LEN)]
    df['language'] = language
    df['source']   = source
    return df.drop_duplicates(subset='text').reset_index(drop=True)

print('Setup complete. DATA_ROOT =', DATA_ROOT)""")

pmd("""---
## STEP 1 — English from the project's own SRH text
Reuses the already-cleaned English text from the **topic** classifier dataset and the
**safety** classifier dataset (two domains: medical Q&A + general/conversational), which
also reduces an English=medical vs Kinyarwanda=news domain confound.""")
pcode("""en_parts = []

topic = pd.read_csv(os.path.join(DATA_ROOT, 'Topic_Classifier_data', 'topic_labels_full.csv'))
en_parts.append(prep(topic['text'], 'english', 'topic_dataset'))
print(f'  topic_dataset English: {len(en_parts[-1])}')

for cand in ['Safety/safety_classifier_final.csv', 'Safety/safety_classifier_clean.csv']:
    fp = os.path.join(DATA_ROOT, cand)
    if os.path.exists(fp):
        safety = pd.read_csv(fp)
        en_parts.append(prep(safety['text'], 'english', 'safety_dataset'))
        print(f'  safety_dataset English: {len(en_parts[-1])}')
        break

english = pd.concat(en_parts, ignore_index=True).drop_duplicates(subset='text').reset_index(drop=True)
print(f'English total (deduped): {len(english)}')""")

pmd("""---
## STEP 2 — Kinyarwanda from KINNEWS (news, MIT license)
Uses headline **titles** (short, query-like) plus **sentence-split article content** so
the Kinyarwanda side has the same short/medium length profile as the English queries.""")
pcode("""kin = load_dataset('kinnews_kirnews', 'kinnews_cleaned', trust_remote_code=True)['train'].to_pandas()
print(f'KINNEWS rows: {len(kin)}  cols: {list(kin.columns)}')

titles = prep(kin['title'], 'kinyarwanda', 'kinnews_title')

def sentence_split(texts):
    out = []
    for c in pd.Series(texts).dropna().astype(str):
        for part in re.split(r'(?<=[.!?])\\s+|\\n+', c):
            out.append(part)
    return pd.Series(out)

kin_sents = prep(sentence_split(kin['content']), 'kinyarwanda', 'kinnews_sentence')
print(f'  kinnews_title   : {len(titles)}')
print(f'  kinnews_sentence: {len(kin_sents)}')""")

pmd("---\n## STEP 3 — Kinyarwanda from SIB-200 (`kin_Latn`, CC-BY-SA-4.0)\nGeneral-domain Kinyarwanda sentences — adds topical variety beyond news.")
pcode("""sib = load_dataset('Davlan/sib200', 'kin_Latn')['train'].to_pandas()
sib_kin = prep(sib['text'], 'kinyarwanda', 'sib200')
print(f'SIB-200 kin: {len(sib_kin)}')

kinyarwanda = pd.concat([titles, kin_sents, sib_kin], ignore_index=True
                        ).drop_duplicates(subset='text').reset_index(drop=True)
print(f'Kinyarwanda total (deduped): {len(kinyarwanda)}')""")

pmd("""---
## STEP 4 — Merge, balance & clean
Balances the two classes with the topic pipeline's `TARGET_PER_CLASS` logic
(undersample the larger class; oversample the smaller **only if** it is short).""")
pcode("""combined = pd.concat([english, kinyarwanda], ignore_index=True)
combined = combined.drop_duplicates(subset='text').reset_index(drop=True)
print('Before balancing:'); print(combined['language'].value_counts().to_string())

counts = combined['language'].value_counts()
TARGET_PER_CLASS = int(min(counts.min(), 6000))
print(f'\\nTARGET_PER_CLASS = {TARGET_PER_CLASS}')

parts = []
for lang in LANG_MAP:
    sub = combined[combined['language'] == lang]
    if len(sub) >= TARGET_PER_CLASS:
        parts.append(sub.sample(n=TARGET_PER_CLASS, random_state=RANDOM_SEED))
        print(f'  {lang:<12}: {len(sub):>6} -> sampled {TARGET_PER_CLASS}')
    else:
        parts.append(sub.sample(n=TARGET_PER_CLASS, replace=True, random_state=RANDOM_SEED))
        print(f'  {lang:<12}: {len(sub):>6} -> oversampled to {TARGET_PER_CLASS}')

balanced = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
balanced['label'] = balanced['language'].map(LANG_MAP)
balanced['text_len'] = balanced['text'].str.len()
print(f'\\nBalanced dataset: {len(balanced)} rows')
print(balanced['language'].value_counts().to_string())""")

pmd("---\n## STEP 5 — Visualise composition")
pcode("""fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
fig.suptitle('Language Classifier Dataset — Composition', fontsize=13, fontweight='bold')

lang_counts = balanced['language'].value_counts()
lang_counts.plot(kind='barh', ax=axes[0], color=['#3B82F6', '#10B981'])
axes[0].set_title('Class distribution (balanced)'); axes[0].set_xlabel('Count')
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y()+p.get_height()/2),
                     ha='left', va='center', fontsize=9)

pivot = balanced.groupby(['language', 'source']).size().unstack(fill_value=0)
pivot.plot(kind='barh', stacked=True, ax=axes[1], colormap='tab10')
axes[1].set_title('Source mix per class'); axes[1].set_xlabel('Count'); axes[1].legend(fontsize=7)

for lang, color in [('english', '#3B82F6'), ('kinyarwanda', '#10B981')]:
    axes[2].hist(balanced[balanced['language'] == lang]['text_len'], bins=30,
                 alpha=0.6, label=lang, color=color)
axes[2].set_title('Text length by class'); axes[2].set_xlabel('Characters'); axes[2].legend()

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/lang_dataset_composition.png', dpi=150, bbox_inches='tight')
plt.show()

print('Text length by class:')
print(balanced.groupby('language')['text_len'].describe()[['mean','min','25%','50%','75%','max']].round(1).to_string())""")

pmd("---\n## STEP 6 — Train/val/test split (70/15/15, stratified) & save")
pcode("""final = balanced[['text', 'language', 'label', 'source']].copy()
train_df, temp_df = train_test_split(final, test_size=0.30, stratify=final['label'], random_state=RANDOM_SEED)
val_df,   test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=RANDOM_SEED)

train_df.to_csv(f'{OUT_DIR}/lang_train.csv', index=False)
val_df.to_csv(f'{OUT_DIR}/lang_val.csv', index=False)
test_df.to_csv(f'{OUT_DIR}/lang_test.csv', index=False)
final.to_csv(f'{OUT_DIR}/lang_labels_full.csv', index=False)
with open(f'{OUT_DIR}/lang_label_map.json', 'w') as f:
    json.dump({'language_to_int': LANG_MAP, 'int_to_language': LANG_NAMES}, f, indent=2)

# Leakage guard
def _keys(df): return set(df['text'].str.strip().str.lower())
ktr, kva, kte = _keys(train_df), _keys(val_df), _keys(test_df)
assert not (ktr & kva) and not (ktr & kte) and not (kva & kte), 'LEAKAGE'

print('=' * 50); print('LANGUAGE DATASET — COMPLETE'); print('=' * 50)
print(f'  Train: {len(train_df):>5}  Val: {len(val_df):>5}  Test: {len(test_df):>5}  Total: {len(final):>5}')
print('  Leakage check: PASSED')
for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    print(f'  {name:<6}', df['label'].value_counts().sort_index().to_dict())""")

pmd("---\n## STEP 7 — Inspect samples per class")
pcode("""for lang in LANG_MAP:
    sub = final[final['language'] == lang]
    print(f'\\n=== {lang.upper()} (label={LANG_MAP[lang]}) — {len(sub)} rows ===')
    for _, r in sub.sample(3, random_state=7).iterrows():
        print(f'  [{r[\"source\"]}] {r[\"text\"][:90]}')""")

save(new(), 'data/Language_Classifier_data/language_classifier_data_pipeline.ipynb', p_cells)

# ════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — TRAINING
# ════════════════════════════════════════════════════════════════════════════
t_cells = []
def tmd(s): t_cells.append(nbf.v4.new_markdown_cell(s))
def tcode(s): t_cells.append(nbf.v4.new_code_cell(s))

tmd("""# Language Classifier (Model 2) — Kinyarwanda vs English
**Project:** SRH AI Platform — ALU Capstone
**Input:** `data/Language_Classifier_data/lang_{train,val,test}.csv`
**Output:** `models/language_classifier.pkl` + comparison chart + confusion matrix

Train-from-scratch (no pretrained language-ID libraries). **Key difference from the topic
classifier:** features are **CHARACTER n-gram TF-IDF** (`analyzer='char_wb'`,
`ngram_range=(2,5)`) — language ID keys on spelling/morphology, not vocabulary. This
matters for Kinyarwanda's agglutinative morphology, where word-level features generalise
poorly to unseen word forms. Same model family as the safety and topic classifiers:
Logistic Regression / Linear SVM (calibrated) / XGBoost.""")

tmd("---\n## STEP 0 — Install dependencies")
tcode("!pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn joblib\nprint('Dependencies ready.')")

tmd("---\n## STEP 1 — Imports and config")
tcode("""import os, json, time, warnings, joblib, random, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

def find_repo_root(start='.'):
    p = os.path.abspath(start)
    while True:
        if (os.path.isdir(os.path.join(p, 'data', 'Topic_Classifier_data'))
                or os.path.isdir(os.path.join(p, '.git'))):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return os.path.abspath(start)
        p = parent

REPO_ROOT = find_repo_root()
DATA_DIR  = os.path.join(REPO_ROOT, 'data', 'Language_Classifier_data')
MODEL_DIR = os.path.join(REPO_ROOT, 'models'); os.makedirs(MODEL_DIR, exist_ok=True)
assert os.path.exists(os.path.join(DATA_DIR, 'lang_train.csv')), \\
    'Run language_classifier_data_pipeline.ipynb first.'
print(f'Repo root      : {REPO_ROOT}')
print(f'Data directory : {DATA_DIR}')
print(f'Model directory: {MODEL_DIR}')""")

tmd("---\n## STEP 2 — Load splits & confirm class distribution")
tcode("""train_df = pd.read_csv(os.path.join(DATA_DIR, 'lang_train.csv'))
val_df   = pd.read_csv(os.path.join(DATA_DIR, 'lang_val.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'lang_test.csv'))
with open(os.path.join(DATA_DIR, 'lang_label_map.json')) as f:
    LMAP = json.load(f)
INT_TO_LANG = {int(k): v for k, v in LMAP['int_to_language'].items()}
CLASS_NAMES = [INT_TO_LANG[i] for i in range(len(INT_TO_LANG))]
POS_LABEL = 1  # kinyarwanda = positive class for binary F1 / ROC-AUC

for df in (train_df, val_df, test_df):
    df['text'] = df['text'].fillna('').astype(str)
    df['label'] = df['label'].astype(int)

print(f'Splits  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}')
def _keys(df): return set(df['text'].str.strip().str.lower())
print('Leakage  train&val={}  train&test={}  val&test={}'.format(
    len(_keys(train_df)&_keys(val_df)), len(_keys(train_df)&_keys(test_df)), len(_keys(val_df)&_keys(test_df))))

dist = pd.DataFrame({s: d['label'].value_counts().sort_index()
                     for s, d in [('train', train_df), ('val', val_df), ('test', test_df)]}).fillna(0).astype(int)
dist.index = [INT_TO_LANG[i] for i in dist.index]
print('\\nClass distribution per split:'); print(dist.to_string())
ax = dist.plot(kind='bar', figsize=(8, 4), color=['#3B82F6', '#F59E0B', '#10B981'])
ax.set_title('Language class distribution per split', fontweight='bold'); ax.set_ylabel('Count')
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/fig_language_class_distribution.png', dpi=150, bbox_inches='tight'); plt.show()

X_train, y_train = train_df['text'], train_df['label'].values
X_val,   y_val   = val_df['text'],   val_df['label'].values
X_test,  y_test  = test_df['text'],  test_df['label'].values""")

tmd("""---
## STEP 3 — Feature extraction: CHARACTER n-gram TF-IDF
`char_wb` builds char n-grams within word boundaries (pads with spaces), so it captures
prefixes/suffixes/morphology. Fit on **train only**. Accents/apostrophes are kept (signal).""")
tcode("""TFIDF_PARAMS = dict(
    analyzer      = 'char_wb',
    ngram_range   = (2, 5),
    max_features  = 50_000,
    sublinear_tf  = True,
    min_df        = 2,
    strip_accents = None,   # keep diacritics/apostrophes as language signal
    lowercase     = True,
)
def make_pipeline(clf):
    return Pipeline([('tfidf', TfidfVectorizer(**TFIDF_PARAMS)), ('clf', clf)])
print('Character n-gram TF-IDF configured (char_wb, 2-5, max 50k).')""")

tmd("---\n## STEP 4 — Define the three models")
tcode("""def _xgb_device():
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True); return 'cuda'
    except Exception:
        return 'cpu'
XGB_DEVICE = _xgb_device(); print(f'XGBoost device: {XGB_DEVICE}')

lr_pipeline  = make_pipeline(LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs',
                                                class_weight='balanced', random_state=RANDOM_SEED))
svm_pipeline = make_pipeline(CalibratedClassifierCV(
    LinearSVC(C=0.5, max_iter=5000, class_weight='balanced', random_state=RANDOM_SEED), cv=3))
xgb_pipeline = make_pipeline(XGBClassifier(objective='binary:logistic', n_estimators=300,
    max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', tree_method='hist', device=XGB_DEVICE,
    random_state=RANDOM_SEED, n_jobs=-1))
MODELS = {'Logistic Regression': lr_pipeline, 'Linear SVM': svm_pipeline, 'XGBoost': xgb_pipeline}
print('Models defined:'); [print(f'  - {n}') for n in MODELS]""")

tmd("---\n## STEP 5 — Train & evaluate on the VALIDATION set\nTest set untouched until Step 7.")
tcode("""results = {}
for name, pipeline in MODELS.items():
    print('\\n' + '=' * 58); print(f'  Training: {name}'); print('=' * 58)
    fit_params = {}
    if name == 'XGBoost':
        fit_params['clf__sample_weight'] = compute_sample_weight('balanced', y_train)
    t0 = time.time(); pipeline.fit(X_train, y_train, **fit_params); train_time = time.time() - t0
    y_pred = pipeline.predict(X_val); y_prob = pipeline.predict_proba(X_val)[:, POS_LABEL]
    acc      = accuracy_score(y_val, y_pred)
    f1_bin   = f1_score(y_val, y_pred, pos_label=POS_LABEL, average='binary')
    f1_macro = f1_score(y_val, y_pred, average='macro')
    roc_auc  = roc_auc_score(y_val, y_prob)
    results[name] = {'accuracy': round(acc, 4), 'f1_binary': round(f1_bin, 4),
                     'f1_macro': round(f1_macro, 4), 'roc_auc': round(roc_auc, 4),
                     'train_sec': round(train_time, 1), 'pipeline': pipeline}
    print(f'  Train time : {train_time:.1f}s')
    print(f'  Accuracy   : {acc:.4f}')
    print(f'  F1 (kinya) : {f1_bin:.4f}   <- selection metric (binary, pos=kinyarwanda)')
    print(f'  F1 Macro   : {f1_macro:.4f}')
    print(f'  ROC-AUC    : {roc_auc:.4f}')
    print('\\n  Classification report (val):')
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, zero_division=0))
print('\\nAll models trained.')""")

tmd("---\n## STEP 6 — Model comparison & selection (by binary F1)")
tcode("""rows = [{'Model': n, 'Accuracy': r['accuracy'], 'F1 (kinya)': r['f1_binary'],
         'F1 Macro': r['f1_macro'], 'ROC-AUC': r['roc_auc'], 'Train (s)': r['train_sec']}
        for n, r in results.items()]
comparison = pd.DataFrame(rows).sort_values('F1 (kinya)', ascending=False).reset_index(drop=True)
print('Model comparison (sorted by binary F1):'); print(comparison.to_string(index=False))
best_name = comparison.iloc[0]['Model']; best_pipeline = results[best_name]['pipeline']
print(f'\\nBest model: {best_name}  (F1 = {results[best_name][\"f1_binary\"]})')

fig, ax = plt.subplots(figsize=(9, 4.5))
models = comparison['Model'].tolist(); x = np.arange(len(models)); w = 0.2
ax.bar(x-1.5*w, comparison['Accuracy'],   width=w, label='Accuracy',   color='#6366F1')
ax.bar(x-0.5*w, comparison['F1 (kinya)'], width=w, label='F1 (kinya)', color='#EF4444')
ax.bar(x+0.5*w, comparison['F1 Macro'],   width=w, label='F1 Macro',   color='#10B981')
ax.bar(x+1.5*w, comparison['ROC-AUC'],    width=w, label='ROC-AUC',    color='#F59E0B')
ax.set_xticks(x); ax.set_xticklabels(models, rotation=10, ha='right')
ax.set_ylim(0, 1.05); ax.set_ylabel('Score'); ax.set_title('Language Classifier — Model Comparison (val)')
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/fig_language_model_comparison.png', dpi=150, bbox_inches='tight'); plt.show()
print(f'Saved: {MODEL_DIR}/fig_language_model_comparison.png')""")

tmd("---\n## STEP 7 — Evaluate the best model on the held-out TEST set\n⚠️ **Run once.**")
tcode("""y_pred_test = best_pipeline.predict(X_test)
y_prob_test = best_pipeline.predict_proba(X_test)[:, POS_LABEL]
test_acc      = accuracy_score(y_test, y_pred_test)
test_f1_bin   = f1_score(y_test, y_pred_test, pos_label=POS_LABEL, average='binary')
test_f1_macro = f1_score(y_test, y_pred_test, average='macro')
test_auc      = roc_auc_score(y_test, y_prob_test)
print(f'TEST SET RESULTS — {best_name}')
print(f'  Accuracy   : {test_acc:.4f}')
print(f'  F1 (kinya) : {test_f1_bin:.4f}')
print(f'  F1 Macro   : {test_f1_macro:.4f}')
print(f'  ROC-AUC    : {test_auc:.4f}\\n')
print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES, zero_division=0))
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, display_labels=CLASS_NAMES,
    colorbar=False, cmap='Blues', ax=ax)
ax.set_title(f'{best_name} — Test Confusion Matrix'); plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/fig_language_confusion_matrix.png', dpi=150, bbox_inches='tight'); plt.show()
print(f'Saved: {MODEL_DIR}/fig_language_confusion_matrix.png')""")

tmd("---\n## STEP 8 — Save the winning pipeline + metadata")
tcode("""joblib.dump(best_pipeline, f'{MODEL_DIR}/language_classifier.pkl')
meta = {'model': 'Model 2 — Language Classifier (Kinyarwanda vs English)',
        'best_model': best_name, 'classes': CLASS_NAMES,
        'features': 'Character n-gram TF-IDF (char_wb, 2-5, max 50k)',
        'val_metrics': {k: results[best_name][k] for k in ['accuracy','f1_binary','f1_macro','roc_auc']},
        'test_metrics': {'accuracy': round(test_acc,4), 'f1_binary': round(test_f1_bin,4),
                         'f1_macro': round(test_f1_macro,4), 'roc_auc': round(test_auc,4)},
        'rows': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)}}
with open(f'{MODEL_DIR}/language_classifier_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'Saved: {MODEL_DIR}/language_classifier.pkl'); print(json.dumps(meta, indent=2))""")

tmd("---\n## STEP 9 — Quick inference + short-query stress test")
tcode("""loaded = joblib.load(f'{MODEL_DIR}/language_classifier.pkl')
samples = [
    'Where can I get free condoms in Kigali?',
    'Ni hehe nabona uburyo bwo kuboneza urubyaro?',
    'What are the symptoms of an STI?',
    'Indwara zandurira mu mibonano mpuzabitsina ni izihe?',
    'umubyibuho ukabije',          # very short Kinyarwanda
    'HIV test',                    # very short English
    'Murakoze cyane',              # short Kinyarwanda greeting
]
preds = loaded.predict(pd.Series(samples))
print(f'{\"Predicted\":<13} Query'); print('-' * 70)
for q, p in zip(samples, preds): print(f'{INT_TO_LANG[int(p)]:<13} {q}')""")

tmd("""---
## STEP 10 — Summary & caveats""")
tcode("""print('=' * 64)
print('LANGUAGE CLASSIFIER (Model 2) — COMPLETE')
print('=' * 64)
print(f'  Winning model : {best_name}  (selected by binary F1)')
print(f'  Rows          : train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}')
print('\\n  Validation comparison:'); print(comparison.to_string(index=False))
print('\\n  Held-out TEST metrics:')
print(f'    Accuracy   : {test_acc:.4f}')
print(f'    F1 (kinya) : {test_f1_bin:.4f}')
print(f'    F1 Macro   : {test_f1_macro:.4f}')
print(f'    ROC-AUC    : {test_auc:.4f}')
print('\\n  Caveats:')
print('   - BINARY EN/RW only: no open labeled code-switch corpus was found, so')
print('     mixed Kinyarwanda-English text is NOT handled (future work).')
print('   - Char n-grams are robust on short queries, but 1-2 word inputs are the')
print('     hardest case (see the short-query stress test in Step 9).')
print('   - Domain skew: English is SRH/medical+safety text, Kinyarwanda is news+')
print('     general sentences; char-level LID is largely domain-robust but worth noting.')
print('=' * 64)""")

save(new(), 'notebooks/language classifier.ipynb', t_cells)
print('done')
