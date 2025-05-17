from gensim.models import Phrases
from gensim.models.phrases import Phraser

def build_ngrams(texts, min_count=10, threshold=100):
    bigram = Phrases(texts, min_count=min_count, threshold=threshold)
    trigram = Phrases(bigram[texts], threshold=threshold)

    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    texts_bigrams = [bigram_mod[doc] for doc in texts]
    texts_trigrams = [trigram_mod[bigram_mod[doc]] for doc in texts]

    return texts_trigrams
