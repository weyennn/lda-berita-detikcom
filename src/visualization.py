import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import os
import pyLDAvis
import pyLDAvis.lda_model

def visualize_interactive_lda(model, doc_term_matrix, vectorizer, save_html='figures/pyldavis_lda.html'):
    panel = pyLDAvis.sklearn_utils.prepare(model, doc_term_matrix, vectorizer)
    pyLDAvis.save_html(panel, save_html)
    print(f"✅ pyLDAvis saved to: {save_html}")



def plot_topic_distribution(model, doc_term_matrix, save_path=None):
    topic_dist = model.transform(doc_term_matrix)
    topic_sums = topic_dist.sum(axis=0)
    topics = [f"Topik {i+1}" for i in range(len(topic_sums))]

    plt.figure(figsize=(8, 5))
    plt.bar(topics, topic_sums)
    plt.xlabel("Topik")
    plt.ylabel("Total Skor Distribusi")
    plt.title("Distribusi Topik LDA")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

def generate_wordclouds_per_topic(model, feature_names, save_dir='figures/wordclouds_per_topic'):
    import os
    os.makedirs(save_dir, exist_ok=True)

    for topic_idx, topic_weights in enumerate(model.components_):
        top_words_idx = topic_weights.argsort()[:-50 - 1:-1]
        top_words = {feature_names[i]: topic_weights[i] for i in top_words_idx}

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_words)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Topik {topic_idx + 1}")
        plt.tight_layout()
        path = f"{save_dir}/wordcloud_topic_{topic_idx + 1}.png"
        plt.savefig(path)
        print(f" Wordcloud disimpan: {path}")
        plt.close()

def visualize_interactive_lda(model, doc_term_matrix, vectorizer, save_html='figures/pyldavis_lda.html'):
    panel = pyLDAvis.lda_model.prepare(model, doc_term_matrix, vectorizer)
    pyLDAvis.save_html(panel, save_html)
    print(f"pyLDAvis saved to: {save_html}")
