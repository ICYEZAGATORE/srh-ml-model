"""
Topic Classifier Dataset Pipeline — REBUILD with labeling fixes.
Project: SRH AI Platform — ALU Capstone (Model 3: Multi-class SRH Topic Classifier)

Fixes vs. the original topic_classifier_data_pipeline.ipynb:
  1. WORD-BOUNDARY keyword matching (regex \\b...\\b) instead of naive substring
     matching. Stops 'implant' matching "implanted", 'disabled'/'blind' matching
     unrelated words, etc.
  2. Labels come from the TEXT CONTENT, not from blanket subject maps. Every kept
     row must genuinely match an SRH keyword. The old fallbacks (all Microbiology
     -> sti_hiv, all Pediatrics -> puberty) that injected non-SRH rows are removed.
  3. argmax-by-match-count: a row is assigned the topic with the MOST keyword hits
     (priority tie-break), instead of the first topic in dict order.
  4. disability_srh requires a disability term AND an SRH-context term (it is "SRH
     for persons with disabilities", not any mention of disability).
  5. NO oversampling-with-replacement. Global dedup, then STRATIFIED split on unique
     rows -> zero cross-split leakage. Minority classes stay smaller; the training
     notebook relies on class_weight='balanced'.

AfriMedQA v2 is a GATED dataset and cannot be downloaded without an HF token, so it
is skipped (it also contributed 0 rows to the original artifacts).
"""
import os, re, json, time
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
SEED = 42
OUT = os.path.dirname(os.path.abspath(__file__))

TOPIC_MAP = {
    'contraception': 0, 'sti_hiv': 1, 'pregnancy': 2, 'puberty': 3,
    'gbv_consent': 4, 'disability_srh': 5, 'general_srh': 6,
}
TOPIC_NAMES = {v: k for k, v in TOPIC_MAP.items()}

# Tie-break priority: more specific / higher-stakes classes win ties; general_srh is
# the catch-all and loses ties.
PRIORITY = ['gbv_consent', 'sti_hiv', 'contraception', 'pregnancy', 'puberty',
            'disability_srh', 'general_srh']

# Tightened keyword lists. Bare generic tokens that caused false positives
# ('pill','patch','injection','implant','discharge','consent') are made specific.
TOPIC_KEYWORDS = {
    'contraception': [
        'contraception', 'contraceptive', 'birth control', 'family planning',
        'condom', 'condoms', 'iud', 'intrauterine device', 'nexplanon',
        'depo provera', 'depo-provera', 'morning after pill', 'plan b',
        'emergency contraception', 'copper coil', 'contraceptive pill',
        'contraceptive injection', 'contraceptive implant', 'birth control pill',
        'diaphragm', 'vasectomy', 'tubal ligation', 'sterilization', 'sterilisation',
    ],
    'sti_hiv': [
        'hiv', 'aids', 'sti', 'stis', 'std', 'stds', 'sexually transmitted',
        'chlamydia', 'gonorrhea', 'gonorrhoea', 'syphilis', 'genital herpes',
        'herpes', 'hpv', 'genital wart', 'genital warts', 'trichomoniasis',
        'bacterial vaginosis', 'antiretroviral', 'antiretrovirals', 'arv', 'arvs',
        'prep', 'pep', 'viral load', 'cd4', 'hepatitis b', 'hepatitis c',
    ],
    'pregnancy': [
        'pregnant', 'pregnancy', 'prenatal', 'antenatal', 'postnatal',
        'trimester', 'fetus', 'foetus', 'fetal', 'childbirth', 'in labour',
        'in labor', 'miscarriage', 'abortion', 'stillbirth', 'ectopic pregnancy',
        'maternal', 'morning sickness', 'fertility', 'infertility', 'ivf',
        'ovulation', 'menstrual cycle', 'missed period', 'pregnancy test',
        'gestation', 'gestational', 'gravida',
    ],
    'puberty': [  # bare 'period'/'periods' removed (matched "period of time")
        'puberty', 'menstruation', 'menstrual', 'menstruating', 'menarche',
        'menstrual period', 'first period', 'late period', 'irregular period',
        'irregular periods', 'period pain', 'period cramps', 'my period',
        'adolescent', 'adolescence', 'breast development', 'body hair',
        'voice breaking', 'growth spurt', 'wet dream', 'dysmenorrhea',
        'pms', 'pmdd', 'pubescent',
    ],
    'gbv_consent': [
        'sexual violence', 'rape', 'raped', 'sexual assault', 'gbv',
        'gender based violence', 'gender-based violence', 'sexual consent',
        'sexual coercion', 'forced sex', 'domestic violence',
        'intimate partner violence', 'sexual abuse', 'sexual harassment',
        'sex trafficking', 'sexual exploitation', 'femicide', 'molested',
        'molestation',
    ],
    'disability_srh': [  # disability terms; only used in conjunction with SRH context
        'disability', 'disabled', 'wheelchair', 'visual impairment',
        'hearing impairment', 'cerebral palsy', 'intellectual disability',
        'physical disability', 'learning disability', 'down syndrome',
        'spinal cord injury', 'paraplegia', 'paraplegic', 'quadriplegia',
    ],
    'general_srh': [
        'sexual health', 'reproductive health', 'sex education', 'vagina',
        'vaginal', 'vulva', 'penis', 'testicle', 'testicles', 'ovary',
        'ovaries', 'uterus', 'cervix', 'cervical', 'pelvic', 'sexual intercourse',
        'virginity', 'hymen', 'libido', 'sex drive', 'masturbation', 'orgasm',
        'erection', 'erectile', 'ejaculation', 'genital', 'genitals',
        'reproductive system',
    ],
}
# SRH context for the disability conjunction rule = every non-disability SRH keyword.
SRH_CONTEXT = sorted({kw for t, kws in TOPIC_KEYWORDS.items()
                      if t != 'disability_srh' for kw in kws})

def _compile(words):
    # longest-first so multi-word phrases win; \b word boundaries
    pats = sorted(words, key=len, reverse=True)
    return re.compile(r'\b(' + '|'.join(re.escape(w) for w in pats) + r')\b',
                      re.IGNORECASE)

TOPIC_RE = {t: _compile(kws) for t, kws in TOPIC_KEYWORDS.items()}
DISABILITY_RE = TOPIC_RE['disability_srh']
SRH_CONTEXT_RE = _compile(SRH_CONTEXT)

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text

def assign_topic(text):
    """Content-based label: topic with the most word-boundary keyword hits."""
    t = text.lower()
    counts = {}
    for topic in ['contraception', 'sti_hiv', 'pregnancy', 'puberty',
                  'gbv_consent', 'general_srh']:
        n = len(TOPIC_RE[topic].findall(t))
        if n:
            counts[topic] = n
    # disability_srh only if a disability term co-occurs with SRH context
    if DISABILITY_RE.search(t) and SRH_CONTEXT_RE.search(t):
        counts['disability_srh'] = len(DISABILITY_RE.findall(t)) + 1  # nudge it in
    if not counts:
        return None
    best = max(counts.values())
    winners = [tp for tp, c in counts.items() if c == best]
    if len(winners) == 1:
        return winners[0]
    for tp in PRIORITY:           # priority tie-break
        if tp in winners:
            return tp
    return winners[0]

def load_with_retry(loader, tries=4):
    for i in range(tries):
        try:
            return loader()
        except Exception as e:
            print(f'    retry {i+1}/{tries} after error: {str(e)[:90]}')
            time.sleep(5 * (i + 1))
    raise RuntimeError('failed after retries')

frames = []

# ── MedMCQA ────────────────────────────────────────────────────────────────
print('Loading MedMCQA...')
mq = load_with_retry(lambda: load_dataset('openlifescienceai/medmcqa'))
mq_df = pd.concat([mq['train'].to_pandas(), mq['validation'].to_pandas()],
                  ignore_index=True)
SRH_SUBJECTS = ['Obstetrics and Gynecology', 'Microbiology',
                'Preventive & Social Medicine', 'Pediatrics', 'Pharmacology',
                'Medicine', 'Gynaecology & Obstetrics']
mq_df = mq_df[mq_df['subject_name'].isin(SRH_SUBJECTS)].copy()
mq_df['text'] = mq_df['question'].apply(clean_text)
mq_df['topic'] = mq_df['text'].apply(assign_topic)
mq_df = mq_df[mq_df['topic'].notna()].copy()
mq_df['source'] = 'medmcqa'
frames.append(mq_df[['text', 'topic', 'source']])
print(f'  MedMCQA kept: {len(mq_df)}')

# ── AdaptLLM med_knowledge_prob (SRHR specialties) ─────────────────────────
print('Loading AdaptLLM med_knowledge_prob...')
ADAPT = ['Gynaecology & Obstetrics', 'Microbiology',
         'Social & Preventive Medicine', 'Pediatrics']
adapt_parts = []
for cfg in ADAPT:
    try:
        ds = load_with_retry(lambda c=cfg: load_dataset('AdaptLLM/med_knowledge_prob', c))
        for split in ds.keys():
            adapt_parts.append(ds[split].to_pandas())
    except Exception as e:
        print(f'    skip {cfg}: {str(e)[:80]}')
adapt_df = pd.concat(adapt_parts, ignore_index=True)
tcol = next((c for c in ['input', 'question', 'text', 'prompt']
             if c in adapt_df.columns), None)
adapt_df['text'] = adapt_df[tcol].apply(clean_text)
adapt_df['topic'] = adapt_df['text'].apply(assign_topic)
adapt_df = adapt_df[adapt_df['topic'].notna()].copy()
adapt_df['source'] = 'adaptllm'
frames.append(adapt_df[['text', 'topic', 'source']])
print(f'  AdaptLLM kept: {len(adapt_df)} (text col: {tcol})')

# ── HealthCareMagic-100k ───────────────────────────────────────────────────
print('Loading HealthCareMagic-100k...')
hcm = load_with_retry(lambda: load_dataset(
    'Malikeh1375/medical-question-answering-datasets', 'chatdoctor_healthcaremagic'))
hcm_df = hcm['train'].to_pandas()
hcol = next((c for c in ['input', 'question', 'instruction']
             if c in hcm_df.columns), None)
hcm_df['text'] = hcm_df[hcol].apply(clean_text)
hcm_df['topic'] = hcm_df['text'].apply(assign_topic)
hcm_df = hcm_df[hcm_df['topic'].notna()].copy()
hcm_df['source'] = 'healthcaremagic'
frames.append(hcm_df[['text', 'topic', 'source']])
print(f'  HealthCareMagic kept: {len(hcm_df)} (text col: {hcol})')

# ── AfriMedQA v2 (gated; needs HF login + "Agree and access" on the dataset page)
print('Loading AfriMedQA v2 (gated)...')
try:
    afri = load_with_retry(lambda: load_dataset('intronhealth/afrimedqa_v2'), tries=2)
    afri_df = pd.concat([afri[s].to_pandas() for s in afri.keys()], ignore_index=True)
    acol = next((c for c in ['question_clean', 'question', 'text']
                 if c in afri_df.columns), None)
    afri_df['text'] = afri_df[acol].apply(clean_text)
    afri_df['topic'] = afri_df['text'].apply(assign_topic)
    afri_df = afri_df[afri_df['topic'].notna()].copy()
    afri_df['source'] = 'afrimedqa'
    frames.append(afri_df[['text', 'topic', 'source']])
    print(f'  AfriMedQA kept: {len(afri_df)} (text col: {acol})')
except Exception as e:
    print(f'  AfriMedQA SKIPPED (no access yet): {str(e)[:120]}')
    print('  -> Accept the gate at https://huggingface.co/datasets/intronhealth/afrimedqa_v2 and re-run.')

# ── Merge, clean, dedup ────────────────────────────────────────────────────
combined = pd.concat(frames, ignore_index=True)
combined['text'] = combined['text'].apply(clean_text)
combined = combined[(combined['text'].str.len() >= 15) &
                    (combined['text'].str.len() <= 1000)].copy()
before = len(combined)
combined['_k'] = combined['text'].str.strip().str.lower()
combined = combined.drop_duplicates(subset='_k').drop(columns='_k').reset_index(drop=True)
combined['label'] = combined['topic'].map(TOPIC_MAP)
print(f'\nCombined unique rows: {len(combined)} (removed {before - len(combined)} dups)')
print('Yield per class (content-labeled, deduped):')
print(combined['topic'].value_counts().to_string())
print('\nSource breakdown:')
print(combined['source'].value_counts().to_string())

# ── Balance: undersample majority classes, NO replacement ──────────────────
counts = combined['topic'].value_counts()
CAP = int(np.median(counts.values))           # balance toward the median class size
CAP = max(CAP, counts.min())                  # never below the smallest class
parts = []
for topic in TOPIC_MAP:
    sub = combined[combined['topic'] == topic]
    if len(sub) > CAP:
        sub = sub.sample(n=CAP, random_state=SEED)
    parts.append(sub)
balanced = pd.concat(parts, ignore_index=True)
balanced = balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f'\nCAP per class = {CAP}. Balanced dataset: {len(balanced)} rows')
print(balanced['topic'].value_counts().to_string())

# ── Stratified split on UNIQUE rows (no leakage) ───────────────────────────
final = balanced[['text', 'topic', 'label', 'source']].copy()
train_df, temp_df = train_test_split(final, test_size=0.30,
                                     stratify=final['label'], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50,
                                   stratify=temp_df['label'], random_state=SEED)

train_df.to_csv(os.path.join(OUT, 'topic_train.csv'), index=False)
val_df.to_csv(os.path.join(OUT, 'topic_val.csv'), index=False)
test_df.to_csv(os.path.join(OUT, 'topic_test.csv'), index=False)
final.to_csv(os.path.join(OUT, 'topic_labels_full.csv'), index=False)
with open(os.path.join(OUT, 'topic_label_map.json'), 'w') as f:
    json.dump({'topic_to_int': TOPIC_MAP, 'int_to_topic': TOPIC_NAMES}, f, indent=2)

# ── Leakage assertion ──────────────────────────────────────────────────────
def keyset(df): return set(df['text'].str.strip().str.lower())
ktr, kva, kte = keyset(train_df), keyset(val_df), keyset(test_df)
assert not (ktr & kva) and not (ktr & kte) and not (kva & kte), 'LEAKAGE!'

print('\n' + '=' * 55)
print('TOPIC CLASSIFIER DATASET — REBUILT (no leakage)')
print('=' * 55)
print(f'  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}  Total: {len(final)}')
print('  Leakage check: PASSED (0 shared texts across splits)')
