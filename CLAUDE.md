# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the LDA analysis pipeline
uv run lda_gov_plans.py

# Install / sync dependencies
uv sync
```

`main.py` is a placeholder and not the analysis entry point.

## Architecture

The entire pipeline lives in `lda_gov_plans.py`. `main.py` is unused boilerplate.

**Pipeline flow (one script, top-to-bottom):**

1. **Text extraction** — `extract_text_from_pdf`: uses `pdfplumber` for native text; falls back to `pdf2image` + `pytesseract` OCR for image-only pages (threshold: <50 chars extracted).
2. **Tokenization + lemmatization** — `clean_and_tokenize`: strips non-Spanish characters (uppercase kept so spaCy lemmatizes with correct casing), runs `es_core_news_sm` (`ner` and `parser` disabled for speed), extracts `token.lemma_.lower()`, then filters by `MIN_TOKEN_LEN` and `ALL_STOPWORDS`. Accents are preserved in lemmas, so stopwords must include accented forms where needed. The spaCy model is loaded once at module level (`nlp = spacy.load(...)`); re-importing the module is expensive.
3. **Corpus** — `build_corpus`: builds a gensim `Dictionary`, filters extremes (`no_below=2`, `no_above=0.85`, `keep_n=5000`).
4. **Coherence sweep** — `find_optimal_topics`: trains LDA for each k in `NUM_TOPICS_RANGE`, saves `coherence_scores.png`.
5. **Final model** — `train_final_lda`: uses `FINAL_NUM_TOPICS` (overrides auto-selected best k).
6. **Distribution** — `get_candidate_topic_distribution`: aggregates all tokens per candidate into one BoW, then calls `get_document_topics`.
7. **Output** — heatmap PNG, stacked bar PNG, `words_per_topic.csv`, `topic_distribution_candidates.csv`, `topic_comparison.html`, saved LDA model.

**Inputs/outputs:**
- PDFs → `./pdfs/` (configured via `PDF_FOLDER`)
- All results → `./results/` (configured via `OUTPUT_FOLDER`)

## Key configuration (top of `lda_gov_plans.py`)

| Constant | Purpose |
|---|---|
| `CANDIDATE_NAMES` | Maps PDF filename → display label; unmapped files use stem |
| `TOPIC_LABELS` | Maps topic index → human label; update after inspecting `words_per_topic.csv` and re-run |
| `NUM_TOPICS_RANGE` | Range of k values evaluated in coherence sweep |
| `FINAL_NUM_TOPICS` | Fixed k for the final model (overrides sweep best) |
| `STOPWORDS_ES` | General Spanish stopwords (unaccented + accented forms both needed) |
| `STOPWORDS_DOMAIN` | Political/candidate-specific words that appear across all plans |
| `ALL_STOPWORDS` | Union of the two sets; used at tokenization time |

## Lemmatization and coherence

Adding spaCy lemmatization shifted the coherence curve significantly. Before lemmatization the sweep (k=5–11) peaked at **k=5** (score ~0.47). After lemmatization the peak moved to **k=9** (~0.47), with k=7 also strong (~0.46). This happens because lemmatization collapses inflected forms into a shared root, increasing vocabulary overlap across documents and letting LDA find finer-grained distinctions.

After additionally removing auxiliary verb lemmas (`haber`, `tener`, `estar`, `deber`, `decir`, `poder`, `querer`, `ser`, `ir`) the coherence curve flattened and the peak shifted to **k=7** (score ~0.47, almost tied with k=5 and k=6). The curve is now very flat in the 5–7 range, meaning topic quality is stable there. `FINAL_NUM_TOPICS` is set to **7** to match the coherence peak.

With k=7, auxiliary verbs removed, and Cepeda-specific rhetoric terms (`compañero`, `revolución`, `revolucionario`, `querido`) moved to `STOPWORDS_DOMAIN`, the final labeled topics are:

| Index | Label | Key signal words |
|---|---|---|
| 0 | Security, Justice & Anti-corruption | seguridad, justicia, público, corrupción, control, energético |
| 1 | Social Policy | salud, educación, público, desarrollo, acceso, servicio |
| 2 | Political Identity & Rhetoric | político, corrupción, histórico, lucha, violencia, cambio |
| 3 | Conflict, Rights & Victims | guerra, sociedad, derecho, víctima, violencia, internacional |
| 4 | Gender & Social Rights | mujer, derecho, lucha, organización, violencia |
| 5 | Economic & Sustainable Development | productivo, desarrollo, sostenible, ambiental, inversión, rural |
| 6 | Territory, Water & Rural Economy | agua, campesino, territorio, potable, economía, población |

**Candidate profiles:**

| Candidate | Primary | Secondary |
|---|---|---|
| Abelardo | Social Policy (0.52) | Security, Justice & Anti-corruption (0.38) |
| Cepeda | Political Identity & Rhetoric (0.62) | Territory, Water & Rural Economy (0.14) |
| Claudia | Social Policy (0.54) | Security, Justice & Anti-corruption (0.34) |
| Fajardo | Social Policy (0.76) | Security, Justice & Anti-corruption (0.15) |
| Paloma | Social Policy (0.46) | Security, Justice & Anti-corruption (0.33) |

Lemmatization also surfaces auxiliary verbs as high-weight tokens. These should be added to `STOPWORDS_ES` as they appear.

## Latent space & similarity

The pipeline computes pairwise cosine similarity and Jensen-Shannon similarity from the topic distribution vectors, and projects them to 2D via PCA (numpy SVD). Outputs: `similarity_cosine.png/csv`, `similarity_js.png/csv`, `latent_space_pca.png` — all embedded in `topic_comparison.html`.

**PCA interpretation (last run):**
- **PC1 (88.9% variance)** is the Cepeda axis: he sits far left (Political Identity & Rhetoric dominant), all others cluster right (Social Policy + Security).
- **PC2 (10.5% variance)** separates Fajardo (top — strongest Social Policy, least Security) from Abelardo, Paloma, and Claudia (bottom — more balanced between Social Policy and Security).
- Claudia, Paloma, and Abelardo are very close together — similar thematic profiles.
- Fajardo is the most distinct among the non-Cepeda candidates.

## Typical iteration loop

1. Add PDFs to `./pdfs/`, register labels in `CANDIDATE_NAMES`.
2. Run → inspect `results/words_per_topic.csv`.
3. Add noisy/ubiquitous words to `STOPWORDS_ES` or `STOPWORDS_DOMAIN`.
4. Update `TOPIC_LABELS` with descriptive names.
5. Adjust `FINAL_NUM_TOPICS` based on `coherence_scores.png`.
6. Re-run.
