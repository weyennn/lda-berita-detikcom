from src.preprocessing import load_data, preprocess_texts
from src.n_gram import build_ngrams
from src.vectorizer import vectorize_count
from src.lda_model import compute_coherence_values
from src.visualization import (
    plot_topic_distribution,
     visualize_interactive_lda,
    generate_wordclouds_per_topic
)

def main():
    df = load_data('data/data_berita_cleaned.xlsx')
    texts = df['title'].astype(str).tolist()

    cleaned_texts = preprocess_texts(texts, custom_stopword_path='data/stopwords.txt')
    ngrammed_texts = build_ngrams(cleaned_texts)
    doc_term_matrix, vectorizer = vectorize_count(ngrammed_texts)

    model_list, coherence_values = compute_coherence_values(
        texts=ngrammed_texts,
        vectorizer=vectorizer,
        doc_term_matrix=doc_term_matrix,
        start=2, limit=11, step=1
    )
    if not model_list:
        print(" Tidak ada model berhasil dibuat.")
        return

    best_idx = coherence_values.index(max(coherence_values))
    best_model = model_list[best_idx]
    best_topic_count = 2 + best_idx
    print(f" Topik terbaik: {best_topic_count}, coherence = {coherence_values[best_idx]:.4f}")

    generate_wordclouds_per_topic(
        model=best_model,
        feature_names=vectorizer.get_feature_names_out()
    )

    plot_topic_distribution(best_model, doc_term_matrix, save_path='figures/lda_topic_distribution.png')
    visualize_interactive_lda(best_model, doc_term_matrix, vectorizer)


if __name__ == '__main__':
    main()
