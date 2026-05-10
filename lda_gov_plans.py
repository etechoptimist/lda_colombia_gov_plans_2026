"""
LDA Pipeline -- Topic Comparison of Colombia 2026 Government Plans
==================================================================
Usage:
    1. Place your PDFs in pdf_folder (default: ./pdfs/)
    2. Adjust CANDIDATE_NAMES for custom labels
    3. Run: python lda_gov_plans.py
    4. Open: results/topic_comparison.html

Requires: pdfplumber, pdf2image, pytesseract, gensim, matplotlib, seaborn
    pip install pdfplumber pdf2image pytesseract gensim matplotlib seaborn pandas
"""

import os
import re
import warnings
import spacy
import numpy as np
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from scipy.spatial.distance import cosine, jensenshannon

warnings.filterwarnings("ignore")

nlp = spacy.load("es_core_news_sm", disable=["ner", "parser"])

# --- CONFIGURATION -----------------------------------------------------------

PDF_FOLDER       = "./pdfs"        # folder containing the PDFs
OUTPUT_FOLDER    = "./results"     # output folder
NUM_TOPICS_RANGE = range(5, 9)     # topic range to evaluate
FINAL_NUM_TOPICS = 7               # final topic count (adjust after reviewing coherence)
LDA_PASSES       = 30              # more passes = more stable, slower
MIN_TOKEN_LEN    = 4               # minimum token length
OCR_LANG         = "spa"           # tesseract OCR language

# Map filename -> candidate label.
# If a file is not listed here, the filename stem is used as the label.
CANDIDATE_NAMES = {
    "PROPUESTAS-DEL-TIGRE.pdf":              "El Tigre",
    "Plan_Integrado_de_Gobierno_Final_Paloma.pdf": "Paloma Valencia",
    # add others as needed:
    # "plan_candidate3.pdf": "Candidate 3",
    # "plan_candidate4.pdf": "Candidate 4",
    # "plan_candidate5.pdf": "Candidate 5",
}

# Spanish stopwords -- manual list (no nltk/spacy dependency)
STOPWORDS_ES = set("""
a al algo algunas algunos ante antes como con contra cual cuando de del desde
donde durante e el ella ellas ellos en entre era eres es esa ese eso esta
este esto fue fue han hay he hecho hu ida igual la las le les lo los mas me
mi mis misma mismo muy na ni no nos nuestro o os para pero poco por que
quien quienes que se si sin sobre su sus tambien te ti tiene un una uno unos
ya yo colombia colombiano colombiana colombianos colombianas pais patria
gobierno nacional plan propuesta propuestas objetivo objetivos programa
programas politica politicas primer primera primero millones billones
traves manera forma parte parte sera seran hara haran debe deben puede
pueden hacer podra podran lograr lograremos trabajar trabajaremos
garantizar garantizaremos implementar implementaremos crear crearemos
establecer estableceremos reducir reduciremos aumentar aumentaremos
mejorar mejoraremos fortalecer fortaleceremos desarrollar desarrollaremos
apoyar apoyaremos promover promoveremos asegurar aseguraremos permitir
permitira generar generaremos buscar buscaremos incluir incluira
nuestro nuestra nuestros nuestras este esta estos estas aquel aquella
dicho dicha dichos dichas cada todo todos toda todas mismo misma
siempre nunca tambien ademas otro otra otros otras solo solo
sino gran grandes porque aqui aquí
está también hemos sido mediante necesitamos
haber tener estar deber decir poder querer ser ir
""".split())

# Political domain stopwords -- appear in ALL plans, add no discriminating value
STOPWORDS_DOMAIN = set("""
colombia colombiana colombiano colombianos pais estado republic nacion
gobierno nacional plan propuesta programa politica reforma sistema
ciudadano ciudadania pueblo gente familia familias vida
presidente presidenta candidato candidata
decreto ley articulo acuerdo contrato proceso recursos fondo
millones billones pesos presupuesto inversion inversion publico publica
ano anos cuatrienio cuadrenio periodo
ivan iván cepeda castro petro uribe gustavo abelardo claudia paloma fajardo
sergio lopez pacto historico colombia humana
compañero revolución revolucionario querido
""".split())

ALL_STOPWORDS = STOPWORDS_ES | STOPWORDS_DOMAIN

# Topic labels -- assign after inspecting the discovered words.
# Leave as defaults on first run; update once you see the output.
TOPIC_LABELS = {
    0: "Security, Justice & Anti-corruption",
    1: "Social Policy",
    2: "Political Identity & Rhetoric",
    3: "Conflict, Rights & Victims",
    4: "Gender & Social Rights",
    5: "Economic & Sustainable Development",
    6: "Territory, Water & Rural Economy",
}

# --- STEP 1: TEXT EXTRACTION -------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """Extract text page by page.
    Uses native text when available; falls back to OCR for image-only pages.
    Returns a list of strings, one per page.
    """
    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()

            if len(text) < 50:
                # page is likely a scanned image -- attempt OCR
                try:
                    from pdf2image import convert_from_path
                    import pytesseract
                    images = convert_from_path(
                        pdf_path,
                        first_page=i + 1,
                        last_page=i + 1,
                        dpi=300
                    )
                    text = pytesseract.image_to_string(images[0], lang=OCR_LANG)
                    print(f"    -> page {i+1}: OCR applied ({len(text)} chars)")
                except Exception as e:
                    print(f"    -> page {i+1}: OCR failed ({e}), skipping")
                    text = ""
            else:
                print(f"    -> page {i+1}: native text ({len(text)} chars)")

            if text.strip():
                pages_text.append(text)

    return pages_text


# --- STEP 2: CLEANING AND TOKENIZATION ---------------------------------------

def clean_and_tokenize(text: str) -> list[str]:
    """Clean text, lemmatize with spaCy, and return filtered tokens."""
    text = re.sub(r'[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_space
        and not token.is_digit
        and len(token.lemma_) >= MIN_TOKEN_LEN
        and token.lemma_.lower() not in ALL_STOPWORDS
    ]

    return tokens


# --- STEP 3: LOAD ALL PDFs ---------------------------------------------------

def load_all_candidates(pdf_folder: str) -> tuple[list, list, list]:
    """Load all PDFs from the folder.
    Returns:
        candidate_names   : list of labels
        all_pages         : list of token lists, one per page
        page_to_candidate : page index -> candidate index mapping
        doc_to_candidate  : full document tokens per candidate
    """
    pdf_files = sorted(Path(pdf_folder).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in '{pdf_folder}'.\n"
            f"Create the folder and place your PDFs there."
        )

    candidate_names = []
    all_pages = []
    page_to_candidate = []
    doc_to_candidate = []

    for pdf_path in pdf_files:
        name = CANDIDATE_NAMES.get(pdf_path.name, pdf_path.stem)
        print(f"\nProcessing: {name} ({pdf_path.name})")

        pages_text = extract_text_from_pdf(str(pdf_path))
        tokens_per_page = [clean_and_tokenize(t) for t in pages_text]
        tokens_per_page = [t for t in tokens_per_page if len(t) > 10]

        candidate_idx = len(candidate_names)
        candidate_names.append(name)

        for page_tokens in tokens_per_page:
            all_pages.append(page_tokens)
            page_to_candidate.append(candidate_idx)

        all_tokens_doc = [t for page in tokens_per_page for t in page]
        doc_to_candidate.append((name, all_tokens_doc))

        print(f"    OK: {len(tokens_per_page)} pages with text, "
              f"{sum(len(p) for p in tokens_per_page)} total tokens")

    return candidate_names, all_pages, page_to_candidate, doc_to_candidate


# --- STEP 4: BUILD GENSIM CORPUS ---------------------------------------------

def build_corpus(all_pages: list) -> tuple:
    """Build the gensim dictionary and corpus for LDA."""
    dictionary = corpora.Dictionary(all_pages)

    dictionary.filter_extremes(
        no_below=2,     # must appear in at least 2 pages
        no_above=0.85,  # must not appear in more than 85% of pages
        keep_n=5000
    )

    corpus = [dictionary.doc2bow(page) for page in all_pages]

    print(f"\nVocabulary: {len(dictionary)} unique words")
    print(f"Corpus: {len(corpus)} documents (pages)")

    return dictionary, corpus


# --- STEP 5: FIND OPTIMAL NUMBER OF TOPICS -----------------------------------

def find_optimal_topics(
    corpus, dictionary, texts, topic_range, output_folder
) -> int:
    """Train LDA for different values of k and plot the coherence score."""
    print(f"\nSearching for optimal number of topics ({min(topic_range)}-{max(topic_range)})...")

    coherence_scores = []

    for n in topic_range:
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n,
            passes=15,
            random_state=42,
            alpha="auto",
            eta="auto"
        )
        cm = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v"
        )
        score = cm.get_coherence()
        coherence_scores.append(score)
        print(f"   k={n}: coherence = {score:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(topic_range), coherence_scores, "o-", color="#2E75B6", linewidth=2)
    ax.set_xlabel("Number of topics")
    ax.set_ylabel("Coherence Score (c_v)")
    ax.set_title("Coherence Score by number of topics\n(choose the 'elbow' of the curve)")
    ax.grid(True, alpha=0.3)

    best_n = list(topic_range)[coherence_scores.index(max(coherence_scores))]
    ax.axvline(x=best_n, color="red", linestyle="--", alpha=0.7,
               label=f"Best: k={best_n} ({max(coherence_scores):.4f})")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(output_folder, "coherence_scores.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n   Saved: {out_path}")
    print(f"   Best k (auto): {best_n} -- review the chart and adjust FINAL_NUM_TOPICS if needed")

    return best_n


# --- STEP 6: TRAIN FINAL LDA MODEL -------------------------------------------

def train_final_lda(corpus, dictionary, num_topics: int):
    """Train the final LDA model with more passes for stability."""
    print(f"\nTraining final LDA with k={num_topics}, passes={LDA_PASSES}...")

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=LDA_PASSES,
        random_state=42,
        alpha="auto",
        eta="auto",
        minimum_probability=0.01
    )

    print("\nDiscovered topics (top 10 words):")
    print("=" * 60)
    for idx, topic in model.print_topics(num_words=10):
        label = TOPIC_LABELS.get(idx, f"Topic {idx}")
        print(f"\n  [{idx}] {label}")
        words = re.findall(r'"([^"]+)"', topic)
        print(f"       {', '.join(words)}")
    print("=" * 60)

    return model


# --- STEP 7: TOPIC DISTRIBUTION PER CANDIDATE --------------------------------

def get_candidate_topic_distribution(
    model, dictionary, doc_to_candidate: list, num_topics: int
) -> pd.DataFrame:
    """Compute the average topic distribution for each candidate."""
    results = {}

    for name, tokens in doc_to_candidate:
        bow = dictionary.doc2bow(tokens)
        topic_dist = dict(model.get_document_topics(bow, minimum_probability=0))

        row = {TOPIC_LABELS.get(i, f"Topic {i}"): topic_dist.get(i, 0.0)
               for i in range(num_topics)}
        results[name] = row

    df = pd.DataFrame(results).T
    df.index.name = "Candidate"
    df = df.div(df.sum(axis=1), axis=0)

    return df


# --- STEP 8: VISUALIZATIONS --------------------------------------------------

def plot_heatmap(df: pd.DataFrame, output_folder: str):
    """Heatmap: candidates vs topics."""
    fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 1.5), len(df) * 1.2 + 2))

    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Topic weight"}
    )

    ax.set_title(
        "Topic distribution by candidate\nHigher values = more emphasis on that topic",
        fontsize=14, pad=15
    )
    ax.set_xlabel("Topics", fontsize=11)
    ax.set_ylabel("Candidate", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(output_folder, "heatmap_candidates_topics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def plot_bars(df: pd.DataFrame, output_folder: str):
    """Stacked bar chart: alternative view to the heatmap."""
    fig, ax = plt.subplots(figsize=(max(12, len(df) * 2), 6))

    colors = plt.cm.Set3.colors[:len(df.columns)]
    bottom = pd.Series([0.0] * len(df), index=df.index)

    for col, color in zip(df.columns, colors):
        ax.bar(df.index, df[col], bottom=bottom, label=col,
               color=color, edgecolor="white", linewidth=0.5)
        bottom += df[col]

    ax.set_title("Thematic composition by candidate", fontsize=14)
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Relative weight")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out_path = os.path.join(output_folder, "bars_candidates_topics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def export_topic_words(model, num_topics: int, output_folder: str):
    """Export the top words per topic to CSV for manual labeling."""
    rows = []
    for idx in range(num_topics):
        words = [w for w, _ in model.show_topic(idx, topn=20)]
        rows.append({
            "topic_id": idx,
            "suggested_label": TOPIC_LABELS.get(idx, f"Topic {idx}"),
            "top20_words": ", ".join(words)
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_folder, "words_per_topic.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"   Saved: {out_path}")
    return df


def export_distribution_csv(df: pd.DataFrame, output_folder: str):
    """Export the topic distribution table to CSV."""
    out_path = os.path.join(output_folder, "topic_distribution_candidates.csv")
    df.round(4).to_csv(out_path, encoding="utf-8-sig")
    print(f"   Saved: {out_path}")


def generate_html_report(
    df: pd.DataFrame, topic_words_df: pd.DataFrame,
    cosine_sim: pd.DataFrame, js_sim: pd.DataFrame,
    output_folder: str
):
    """Generate an HTML report combining all outputs."""
    heatmap_path    = "heatmap_candidates_topics.png"
    bars_path       = "bars_candidates_topics.png"
    coherence_path  = "coherence_scores.png"
    cosine_path     = "similarity_cosine.png"
    js_path         = "similarity_js.png"
    pca_path        = "latent_space_pca.png"

    table_html = df.round(3).to_html(
        classes="table", border=0, float_format=lambda x: f"{x:.3f}"
    )

    topics_html    = topic_words_df.to_html(classes="table", border=0, index=False)
    cosine_html    = cosine_sim.round(3).to_html(classes="table", border=0, float_format=lambda x: f"{x:.3f}")
    js_html        = js_sim.round(3).to_html(classes="table", border=0, float_format=lambda x: f"{x:.3f}")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Colombia 2026 Government Plans -- Topic Comparison</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto;
         padding: 30px; background: #f5f5f5; color: #333; }}
  h1 {{ color: #1F2937; border-bottom: 3px solid #2E75B6; padding-bottom: 10px; }}
  h2 {{ color: #2E75B6; margin-top: 40px; }}
  .table {{ border-collapse: collapse; width: 100%; background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 13px; }}
  .table th {{ background: #2E75B6; color: white; padding: 10px 12px; text-align: left; }}
  .table td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
  .table tr:hover td {{ background: #f0f7ff; }}
  img {{ max-width: 100%; border-radius: 8px;
         box-shadow: 0 2px 8px rgba(0,0,0,0.15); margin: 20px 0; }}
  .note {{ background: #fff3cd; border-left: 4px solid #ffc107;
           padding: 12px 16px; border-radius: 4px; margin: 20px 0; }}
  .section {{ background: white; padding: 24px; border-radius: 8px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 30px; }}
</style>
</head>
<body>

<h1>Colombia 2026 Government Plans -- Topic Comparison</h1>
<p>LDA topic analysis applied to the presidential candidates' government plans.</p>

<div class="note">
  <strong>Next step:</strong> Open <code>words_per_topic.csv</code>,
  read the top words for each topic, and assign descriptive labels in the
  <code>TOPIC_LABELS</code> dict in the script. Re-run to get the final report
  with meaningful labels.
</div>

<div class="section">
  <h2>Topic distribution by candidate</h2>
  <p>Each value represents how much weight that topic has in the candidate's plan.
     Higher value = more emphasis on that theme.</p>
  {table_html}
</div>

<div class="section">
  <h2>Comparative heatmap</h2>
  <img src="{heatmap_path}" alt="Heatmap candidates vs topics">
</div>

<div class="section">
  <h2>Thematic composition (stacked bars)</h2>
  <img src="{bars_path}" alt="Stacked bars by candidate">
</div>

<div class="section">
  <h2>Coherence Score</h2>
  <p>Chart for choosing the optimal number of topics. The "elbow" of the curve
     indicates the best balance between interpretability and detail.</p>
  <img src="{coherence_path}" alt="Coherence score by number of topics">
</div>

<div class="section">
  <h2>Latent space (PCA)</h2>
  <p>Each candidate projected onto the first two principal components of their 7-dimensional
     topic distribution vector. Closer points = more similar thematic profiles.</p>
  <img src="{pca_path}" alt="Latent space PCA">
</div>

<div class="section">
  <h2>Candidate similarity (cosine)</h2>
  <p>Based on the topic distribution vectors. Values close to 1 = very similar thematic profile.</p>
  {cosine_html}
  <img src="{cosine_path}" alt="Cosine similarity heatmap">
</div>

<div class="section">
  <h2>Candidate similarity (Jensen-Shannon)</h2>
  <p>Jensen-Shannon similarity treats the topic distributions as probability distributions.
     More robust than cosine for comparing proportions.</p>
  {js_html}
  <img src="{js_path}" alt="Jensen-Shannon similarity heatmap">
</div>

<div class="section">
  <h2>Words per topic</h2>
  <p>Use these words to manually label each topic in the script
     (<code>TOPIC_LABELS</code>).</p>
  {topics_html}
</div>

</body>
</html>"""

    out_path = os.path.join(output_folder, "topic_comparison.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   Saved: {out_path}")


# --- STEP 9: SIMILARITY ------------------------------------------------------

def plot_latent_space(df: pd.DataFrame, output_folder: str):
    """Project topic distribution vectors to 2D via PCA and plot the latent space."""
    X = df.values
    X_centered = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    coords = X_centered @ Vt[:2].T

    # variance explained
    cov = X_centered.T @ X_centered
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    var_explained = eigvals[:2] / eigvals.sum() * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.Set1.colors

    for i, candidate in enumerate(df.index):
        x, y = coords[i]
        ax.scatter(x, y, color=colors[i], s=180, zorder=3)
        ax.annotate(
            candidate,
            xy=(x, y),
            xytext=(10, 6),
            textcoords="offset points",
            fontsize=10,
            color=colors[i],
            fontweight="bold"
        )

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)", fontsize=11)
    ax.set_title("Candidate latent space (PCA of topic distributions)", fontsize=13)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(output_folder, "latent_space_pca.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def compute_similarity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise cosine similarity and Jensen-Shannon similarity from topic distributions."""
    candidates = df.index.tolist()
    cosine_sim = pd.DataFrame(index=candidates, columns=candidates, dtype=float)
    js_sim     = pd.DataFrame(index=candidates, columns=candidates, dtype=float)

    for c1 in candidates:
        for c2 in candidates:
            v1, v2 = df.loc[c1].values, df.loc[c2].values
            cosine_sim.loc[c1, c2] = 1 - cosine(v1, v2)
            js_sim.loc[c1, c2]     = 1 - jensenshannon(v1, v2)

    return cosine_sim.astype(float), js_sim.astype(float)


def plot_similarity(sim_df: pd.DataFrame, title: str, filename: str, output_folder: str):
    """Heatmap for a pairwise similarity matrix."""
    fig, ax = plt.subplots(figsize=(len(sim_df) * 1.4 + 2, len(sim_df) * 1.2 + 2))
    sns.heatmap(
        sim_df, annot=True, fmt=".3f", cmap="YlGn",
        vmin=0, vmax=1, linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "Similarity (0–1)"}
    )
    ax.set_title(title, fontsize=13, pad=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path = os.path.join(output_folder, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


# --- MAIN --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  LDA -- Colombia 2026 Government Plans")
    print("=" * 60)

    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. load PDFs
    candidate_names, all_pages, page_to_candidate, doc_to_candidate = \
        load_all_candidates(PDF_FOLDER)

    print(f"\n{len(candidate_names)} candidates loaded: {candidate_names}")
    print(f"   Total pages with text: {len(all_pages)}")

    if len(all_pages) < 10:
        print("\nWARNING: very few pages -- LDA results may be unstable.")
        print("   Consider lowering MIN_TOKEN_LEN or checking text extraction.")

    # 2. build corpus
    dictionary, corpus = build_corpus(all_pages)

    # 3. coherence sweep to pick k
    best_k = find_optimal_topics(
        corpus, dictionary, all_pages, NUM_TOPICS_RANGE, OUTPUT_FOLDER
    )

    num_topics = FINAL_NUM_TOPICS if FINAL_NUM_TOPICS else best_k
    print(f"\n   Using k={num_topics} for the final model")

    # 4. train final LDA
    model = train_final_lda(corpus, dictionary, num_topics)

    # 5. topic distribution per candidate
    print("\nCalculating topic distribution per candidate...")
    df = get_candidate_topic_distribution(
        model, dictionary, doc_to_candidate, num_topics
    )

    print("\n" + "=" * 60)
    print("TOPIC DISTRIBUTION BY CANDIDATE")
    print("=" * 60)
    print(df.round(3).to_string())

    # 6. similarity + latent space
    print("\nCalculating pairwise similarity and latent space...")
    plot_latent_space(df, OUTPUT_FOLDER)
    cosine_sim, js_sim = compute_similarity(df)
    plot_similarity(cosine_sim, "Candidate similarity (cosine)", "similarity_cosine.png", OUTPUT_FOLDER)
    plot_similarity(js_sim,     "Candidate similarity (Jensen-Shannon)", "similarity_js.png", OUTPUT_FOLDER)
    cosine_sim.round(4).to_csv(os.path.join(OUTPUT_FOLDER, "similarity_cosine.csv"), encoding="utf-8-sig")
    js_sim.round(4).to_csv(os.path.join(OUTPUT_FOLDER, "similarity_js.csv"), encoding="utf-8-sig")
    print(f"   Saved: similarity_cosine.csv / similarity_js.csv")

    # 7. export and visualize
    print("\nSaving results...")
    plot_heatmap(df, OUTPUT_FOLDER)
    plot_bars(df, OUTPUT_FOLDER)
    topic_words_df = export_topic_words(model, num_topics, OUTPUT_FOLDER)
    export_distribution_csv(df, OUTPUT_FOLDER)
    generate_html_report(df, topic_words_df, cosine_sim, js_sim, OUTPUT_FOLDER)

    model_path = os.path.join(OUTPUT_FOLDER, "lda_model")
    model.save(model_path)
    print(f"   Saved: {model_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nAll results in: {OUTPUT_FOLDER}/")
    print(f"   -> topic_comparison.html          (main report)")
    print(f"   -> words_per_topic.csv             (for labeling topics)")
    print(f"   -> heatmap_candidates_topics.png")
    print(f"   -> coherence_scores.png")
    print()
    print("NEXT STEPS:")
    print("   1. Open words_per_topic.csv")
    print("   2. Read the top words for each topic")
    print("   3. Add labels to TOPIC_LABELS in the script")
    print("      e.g.  0: 'Security',  1: 'Economy', ...")
    print("   4. Re-run the script to get the final labeled report")
    print()


if __name__ == "__main__":
    main()
