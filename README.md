# Smart Document Insights

Smart Document Insights adalah aplikasi berbasis web yang didukung oleh AI untuk membantu pengguna memahami dokumen berukuran besar atau kompleks, seperti kontrak hukum, makalah penelitian, dan laporan keuangan. Aplikasi ini menggunakan teknologi **RAG (Retrieval-Augmented Generation)** dengan LangChain dan OpenAI.

## Fitur Utama

- **Pemrosesan Dokumen Otomatis**: Unggah file PDF, dan aplikasi akan secara otomatis mengekstrak, memecah, dan membuat representasi vektor (embeddings) dari teks dokumen.
- **Pencarian Cerdas**: Menggunakan ChromaDB (Vector Database) untuk mencari bagian dokumen yang paling relevan dengan pertanyaan Anda.
- **Tanya Jawab Berbasis AI**: Memanfaatkan model bahasa besar (LLMs) dari OpenAI untuk memberikan jawaban yang komprehensif, terstruktur, dan akurat berdasarkan isi dokumen.

## Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Orkestrasi AI**: [LangChain](https://python.langchain.com/)
- **LLM & Embeddings**: [OpenAI API](https://openai.com/api/)
- **Vector Database**: [Chroma](https://www.trychroma.com/)
- **Pemrosesan PDF**: `pypdf`

## Prasyarat

Sebelum menjalankan aplikasi, pastikan Anda telah menginstal:

- Python 3.9 atau lebih baru
- [OpenAI API Key](https://platform.openai.com/account/api-keys)

## Instalasi

1. Clone repositori ini atau unduh kode sumbernya:
   ```bash
   git clone <url-repo-anda>
   cd smart-document-insights
   ```

2. Buat virtual environment (opsional namun disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate     # Untuk Windows
   ```

3. Instal dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan

1. Jalankan aplikasi Streamlit dengan perintah:
   ```bash
   streamlit run app.py
   ```

2. Buka browser Anda dan akses URL lokal yang ditampilkan (biasanya `http://localhost:8501`).

3. Di bilah samping (sidebar):
   - Masukkan **OpenAI API Key** Anda.
   - Unggah dokumen PDF yang ingin dianalisis.

4. Tunggu beberapa saat hingga proses pembuatan vector database selesai.

5. Mulai ajukan pertanyaan terkait dokumen di kolom obrolan yang tersedia!

## Catatan Keamanan

Harap berhati-hati saat mengunggah dokumen yang bersifat rahasia (seperti kontrak hukum atau data finansial). Karena aplikasi ini menggunakan API dari pihak ketiga (OpenAI) untuk embeddings dan LLM, pastikan Anda mematuhi kebijakan privasi dan keamanan organisasi Anda.
