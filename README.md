# LDA Topik Modeling Berita Indonesia

Proyek ini bertujuan untuk melakukan analisis topik otomatis terhadap kumpulan judul berita berbahasa Indonesia menggunakan algoritma Latent Dirichlet Allocation (LDA). Pipeline ini mencakup preprocessing teks, pembentukan n-gram, evaluasi topic coherence, dan visualisasi hasil dalam bentuk wordcloud per topik serta perceptual map.

## Alur Analisis

1. Input Data  
   Format: Excel (`.xlsx`) — kolom utama: `title`

2. Preprocessing
   - Case folding
   - Stopword removal (Sastrawi + stopwords manual)
   - Tokenisasi

3. Bigram & Trigram  
   Menggabungkan frasa populer seperti `korupsi_anggaran`, `bantuan_sosial`.

4. Vectorization  
   Menggunakan CountVectorizer untuk menghasilkan dokumen-term matrix.

5. Topic Modeling (LDA)  
   Pemilihan jumlah topik terbaik berdasarkan coherence score (c_v).

6. Visualisasi
   - Wordcloud per topik
   - Perceptual Map (PCA 2D scatter)
   - Visualisasi interaktif menggunakan pyLDAvis

## Struktur Folder

```
lda-berita/
├── data/
│   ├── data_berita_cleaned.xlsx
│   └── stopwords.txt
├── figures/
│   ├── lda_topic_distribution.png
│   ├── pyldavis_lda.html
│   └── wordclouds_per_topic/
│       ├── wordcloud_topic_1.png
│       ├── wordcloud_topic_2.png
│       └── ...
├── src/
│   ├── preprocessing.py
│   ├── ngram.py
│   ├── vectorizer.py
│   ├── lda_model.py
│   └── visualization.py
├── main.py
└── requirements.txt
```

## Cara Menjalankan

```bash
# 1. Buat dan aktifkan virtual environment
python3 -m venv env
source env/bin/activate

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan analisis
python main.py
```

## Dependensi Utama

- pandas
- matplotlib
- scikit-learn
- wordcloud
- gensim
- Sastrawi
- pyLDAvis

### Deskripsi Dataset

Dataset yang digunakan dalam proyek ini terdiri dari kumpulan **judul berita berbahasa Indonesia** yang diperoleh dari kanal **detikNews** di situs [https://news.detik.com](https://news.detik.com).

- **Populasi**: Semua judul berita pada kanal detikNews
- **Sumber data**: [https://news.detik.com](https://news.detik.com)
- **Rentang waktu pengambilan data**: 16 Desember 2021 hingga 24 Maret 2022
- **Jumlah data**: 1980 judul berita
- **Format data**: Excel (`.xlsx`) dengan satu kolom utama yaitu `title` yang berisi judul berita

Tujuan pengumpulan data ini adalah untuk menganalisis kecenderungan topik yang muncul dalam periode waktu tertentu menggunakan pendekatan unsupervised topic modeling.
### Deskripsi Model LDA

Model utama yang digunakan dalam proyek ini adalah **Latent Dirichlet Allocation (LDA)** yang merupakan metode *probabilistic topic modeling* berbasis unsupervised learning.

- **Model**: LDA dari `scikit-learn` (`LatentDirichletAllocation`)
- **Vectorizer**: CountVectorizer
- **Pemilihan jumlah topik**: Otomatis berdasarkan nilai **coherence score (c_v)** menggunakan `gensim.models.CoherenceModel`
- **Preprocessing**:
  - Case folding (huruf kecil)
  - Penghapusan angka, tanda baca
  - Stopword removal (Sastrawi + custom list)
  - Tokenisasi dan pembentukan n-gram (bigram & trigram)

- **Visualisasi hasil**:
  - Wordcloud per topik
  - Perceptual map (PCA)
  - Visualisasi interaktif via pyLDAvis

## Output

- Wordcloud per topik (tersimpan di folder `figures/wordclouds_per_topic/`)
- Distribusi Topik (tersimpan sebagai `lda_topic_distribution.png`)
- Visualisasi interaktif HTML (tersimpan sebagai `pyldavis_lda.html`)

## Author

Yayang Matira — A master's student in Computer Science at Universitas Gadjah Mada