# Netflix Viewing Behavior — Personal Data Analysis

A full end-to-end data analysis project built on my own Netflix viewing history.
The goal was to reverse-engineer my viewing preferences from raw behavioral data,
using a combination of classical data science tools, external APIs, and LLMs.

---

## Project Structure
├── exploration.ipynb              # Full analysis notebook — iterative process,
│                                  # decisions, dead ends, and reasoning documented
├── analysis.ipynb                 # Clean production notebook — runs end-to-end
├── thematic_profiles_cache.json   # Cached LLM API responses (avoids re-billing)
├── viewing_patterns.png           # Visualization output
├── viewing_personality_summary.txt # LLM-generated personality portrait (see note in file)
└── requirements.txt

> **Note on structure**: This project was developed as a single iterative notebook
> rather than modular scripts. `exploration.ipynb` shows the actual development
> process — including intermediate decisions, threshold tuning, and model comparisons.
> `analysis.ipynb` is the cleaned version that runs sequentially from start to finish.
> The analysis covers the following stages in order:
>
> 1. Data collection & TMDB API enrichment
> 2. Feature engineering (engagement score, binge metrics)
> 3. Preference vectors (genre one-hot, keyword TF-IDF)
> 4. LLM enrichment via Claude API (thematic profiling)
> 5. Modeling & train/test evaluation
> 6. Clustering (K-Means + PCA)
> 7. Time analysis & visualization
> 8. LLM-generated natural language summary

---

## Data Sources

- **Netflix** — Personal viewing history export (`ViewingActivity.csv`, `Ratings.csv`)
  Available at: netflix.com → Account → Download your personal information
  *(Not included in this repo — personal data)*

- **TMDB API** — Metadata enrichment: genres, keywords, cast, episode count, runtime
  Free API key at: themoviedb.org/settings/api

- **Claude API (Anthropic)** — Thematic profiling of shows via natural language analysis

---

## Methodology

### 1. Data Preparation
Merged three data sources: viewing sessions, explicit ratings (Thumbs 0–3),
and TMDB metadata. Handled Hebrew-to-English title mapping for rating joins.
Filtered sessions under 3 minutes (trailers, misclicks).

### 2. Feature Engineering
Built behavioral features that don't exist in the raw data:

| Feature | Description |
|---|---|
| `engagement_score` | Weighted composite: 40% rating + 30% completion + 20% binge + 10% pace (unrated shows: 50% completion + 30% binge + 20% pace) |
| `completion_ratio` | Minutes watched / estimated total runtime |
| `binge_score` | Normalized: max episodes in one day + sustained binge days |
| `pace_norm` | Average daily episodes, capped at P90 of distribution |

All caps were set empirically from the actual data distribution (P90/P95),
not from arbitrary assumptions.

### 3. Preference Vectors
- **Genre vector**: One-hot encoding → weighted average by engagement score
- **Keyword vector**: TF-IDF (penalizes common keywords like "based on novel",
  rewards specific ones like "time travel") → weighted average by engagement

### 4. LLM Enrichment
Sent each show's Overview + Keywords to Claude Haiku and received structured
JSON with scores (1–5) on 6 thematic dimensions:
`pacing`, `darkness`, `complexity`, `emotional_intensity`, `mystery_level`, `realism`

Results cached in `thematic_profiles_cache.json` to avoid redundant API calls.

### 5. Modeling
Built and compared two model families:

**With behavioral features — not a valid result (target leakage):**
R² = 0.978, but the behavioral features (completion, binge, pace) are the same values
used to build the engagement score, so the model is partly predicting its own inputs.
This number only confirms the score is internally consistent; it is not predictive power.

**Content-only features (no leakage), 198 shows with completion and thematic data:**

| Model | R² | MAE |
|---|---|---|
| Ridge Regression (29 features) | 0.262 | 0.222 |
| Ridge Regression (6 thematic only) | 0.125 | 0.246 |
| Random Forest (29 features) | -0.010 | 0.253 |
| Cosine Similarity Engine | 0.199 | 0.219 |

5-fold cross-validation R² (Ridge): −153.6 ± 226.4 with 29 features and −160.8 ± 241.6
with 6 thematic features. The scores in the table do not hold up under cross-validation,
so they should not be read as real predictive performance.

**Result:** content metadata (genres, keywords, LLM thematic scores) does not reliably
predict my engagement on this dataset. The likely reasons are the small sample, the sparse
explicit ratings (see Limitations), and that taste depends on things metadata doesn't
capture (acting, mood, personal context). Testing that properly would need data from
other users.

### 6. Clustering
K-Means (K=5, selected via Elbow method) on thematic + genre feature matrix.
PCA to 2 components for visualization (34.5% variance explained).

| Cluster | Label | Avg Engagement |
|---|---|---|
| 3 | Dark Sci-Fi & Fantasy | 0.518 |
| 2 | Psychological Thriller | 0.486 |
| 0 | True Crime / Documentary | 0.399 |
| 4 | Reality / Light | 0.289 |
| 1 | Family / Comedy | 0.218 |

### 7. Time Analysis
Extracted viewing patterns by hour, day of week (Israeli Sunday-first week),
month, and device. Peak: Friday 18:00–19:00.

### 8. LLM Summary
Sent the full numerical profile (engagement stats, top shows, preference vectors,
cluster breakdown, model findings) to Claude Sonnet for a natural language
portrait of viewing personality. Output saved in `viewing_personality_summary.txt`.

---

## Key Findings

- **Metadata is a weak predictor**: genre, keywords and thematic scores did not reliably
  predict my engagement; the near-perfect behavioral model was caused by target leakage
- **Top content cluster**: Dark, complex Sci-Fi & Fantasy with high mystery
- **Peak viewing**: Friday evenings — deliberate end-of-week decompression pattern
- **Dominant keywords**: miniseries, thriller, murder, time travel, based on novel
- **Thematic profile**: High emotional intensity (3.9/5) and mystery (3.8/5), lower realism (3.1/5)

---

## Tools & Libraries

| Tool | Usage |
|---|---|
| pandas | Data wrangling, merging, aggregation |
| numpy | Vectorized computation, normalization |
| scikit-learn | TF-IDF, encoding, regression, clustering, PCA |
| matplotlib + seaborn | Custom Netflix-styled visualizations |
| requests | TMDB API calls |
| Anthropic Python SDK | Claude API for LLM enrichment and summary |
| PowerBI | Interactive dashboard (4 pages) |
| python-dotenv | Secure API key management |

---

## Limitations

- **No first-watch isolation**: Cannot separate first viewing from re-watches
- **Small dataset**: 246 unique shows (198 usable for modeling) limits model generalizability
- **Sparse ratings**: only 42 of the 246 shows have an explicit rating from me; for the
  other 204 the engagement score is built only from viewing behavior
- **Target leakage**: behavioral features are part of the engagement score, so they can't be
  used to predict it
- **Single user**: No collaborative filtering possible
- **TMDB coverage**: ~7 shows had no metadata (missing Overview)
- **Content ceiling**: Low content-only R² reflects that taste is shaped by
  factors outside metadata — performance, personal context, timing

---

## Setup

```bash
git clone https://github.com/omrisalem/netflix-viewing-analysis
cd netflix-viewing-analysis
pip install -r requirements.txt
```

Create a `.env` file:
TMDB_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

Export your Netflix data and place `ViewingActivity.csv` and `Ratings.csv`
in the project root, then run `analysis.ipynb` from top to bottom.

---

## Author
Omri Salem — Computer Science student, Israel
Built as a portfolio project demonstrating end-to-end data analysis skills.
