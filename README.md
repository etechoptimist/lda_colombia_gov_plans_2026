# Colombia 2026 Government Plans — LDA Topic Analysis

Unsupervised topic modelling of the presidential candidates' government plans using Latent Dirichlet Allocation (LDA). The pipeline extracts text from PDFs, lemmatizes with spaCy, discovers thematic topics, and compares candidates in a shared latent space.

## Candidates

| File | Candidate |
|---|---|
| `programa-gobierno-abelardo.pdf` | Abelardo |
| `programa-gobierno-cepeda.pdf` | Cepeda |
| `programa-gobierno-claudia.pdf` | Claudia |
| `programa-gobierno-fajardo.pdf` | Fajardo |
| `programa-gobierno-paloma.pdf` | Paloma |

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run python -m spacy download es_core_news_sm
```

Place PDFs in `./pdfs/` and run:

```bash
uv run lda_gov_plans.py
```

Open `results/topic_comparison.html` for the full report.

## Topics (k=7, coherence 0.4260)

| # | Label | Key words |
|---|---|---|
| 0 | Security, Justice & Anti-corruption | seguridad, justicia, corrupción, control |
| 1 | Social Policy | salud, educación, desarrollo, acceso, servicio |
| 2 | Political Identity & Rhetoric | político, histórico, lucha, violencia, cambio |
| 3 | Conflict, Rights & Victims | guerra, víctima, violencia, internacional |
| 4 | Gender & Social Rights | mujer, derecho, organización, violencia |
| 5 | Economic & Sustainable Development | productivo, sostenible, ambiental, inversión |
| 6 | Territory, Water & Rural Economy | agua, campesino, territorio, potable, rural |

## Candidate profiles

| Candidate | Primary topic | Secondary topic |
|---|---|---|
| Abelardo | Social Policy (0.52) | Security & Justice (0.38) |
| Cepeda | Political Identity & Rhetoric (0.62) | Territory & Rural (0.14) |
| Claudia | Social Policy (0.54) | Security & Justice (0.34) |
| Fajardo | Social Policy (0.76) | Security & Justice (0.15) |
| Paloma | Social Policy (0.46) | Security & Justice (0.33) |

## Latent space

Candidates are projected to 2D via PCA of their 7-dimensional topic distribution vectors. PC1 (88.9% of variance) separates Cepeda from all others. PC2 (10.5%) separates Fajardo from the Abelardo/Claudia/Paloma cluster.

## Outputs (`results/`)

| File | Description |
|---|---|
| `topic_comparison.html` | Main report (all charts + tables) |
| `words_per_topic.csv` | Top 20 words per topic for labeling |
| `topic_distribution_candidates.csv` | Topic weights per candidate |
| `heatmap_candidates_topics.png` | Heatmap: candidates × topics |
| `bars_candidates_topics.png` | Stacked bar chart |
| `coherence_scores.png` | Coherence sweep (k=5–8) |
| `latent_space_pca.png` | 2D PCA of topic vectors |
| `similarity_cosine.png/csv` | Pairwise cosine similarity |
| `similarity_js.png/csv` | Pairwise Jensen-Shannon similarity |
