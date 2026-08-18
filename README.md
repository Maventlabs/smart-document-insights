# Smart Document Insights 📄

Smart Document Insights adalah aplikasi berbasis web yang didukung oleh AI untuk membantu pengguna memahami dokumen berukuran besar atau kompleks, seperti kontrak hukum, makalah penelitian, dan laporan keuangan. Aplikasi ini menggunakan teknologi **RAG (Retrieval-Augmented Generation)** dengan LangChain dan **NVIDIA NIM** (100+ model AI gratis).

## ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **💬 Tanya Jawab** | Ajukan pertanyaan bebas terhadap dokumen yang diunggah |
| **📋 Ringkasan Dokumen** | Generate ringkasan otomatis dengan satu klik |
| **🔍 Key Insights** | Ekstrak wawasan kunci: temuan, data, risiko, rekomendasi |
| **📁 Multi-Dokumen** | Unggah beberapa file sekaligus (PDF, TXT, DOCX) |
| **📊 Statistik Dokumen** | Lihat jumlah halaman, kata, karakter, dan chunks |
| **📥 Export** | Export chat history, ringkasan, dan insights ke Markdown |
| **🤖 Pilihan Model** | Pilih model OpenAI: GPT-3.5 Turbo, GPT-4, atau GPT-4 Turbo |
| **🔧 Pengaturan Lanjut** | Atur chunk size, overlap, dan jumlah konteks (k) |

## 🛠 Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Orkestrasi AI**: [LangChain](https://python.langchain.com/) + LangChain Classic
- **LLM & Embeddings**: [NVIDIA NIM](https://build.nvidia.com) (100+ model gratis)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Pemrosesan File**: `pypdf`, `docx2txt`

## 📋 Prasyarat

- Python 3.9 atau lebih baru
- [NVIDIA NIM API Key](https://build.nvidia.com) (Gratis! Tanpa kartu kredit)

## 🚀 Instalasi

1. Clone repositori:
   ```bash
   git clone <url-repo-anda>
   cd smart-document-insights
   ```

2. Buat virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Cara Menjalankan

```bash
streamlit run app.py
```

Buka browser → `http://localhost:8501`

### Langkah Penggunaan:

1. **Masukkan NVIDIA NIM API Key** → di sidebar (gratis di [build.nvidia.com](https://build.nvidia.com))
2. **Unggah Dokumen** → PDF, TXT, atau DOCX (bisa beberapa sekaligus)
3. **Pilih Model** → Llama 3.1 (cepat) atau Nemotron 340B (terkuat) atau DeepSeek R1 (reasoning)
4. **Mulai Berinteraksi**:
   - **Tab 💬 Tanya Jawab** → Ajukan pertanyaan bebas
   - **Tab 📋 Ringkasan** → Generate ringkasan otomatis
   - **Tab 🔍 Key Insights** → Ekstrak wawasan kunci
5. **Export Hasil** → Download sebagai Markdown

## 🧪 Menjalankan Test

```bash
python -m unittest test_app -v
```

## 🤖 Model yang Tersedia (Gratis via NVIDIA NIM)

| Model | Keunggulan |
|-------|------------|
| Meta Llama 3.1 8B | Cepat, ringan |
| Meta Llama 3.1 70B | Seimbang, kuat |
| Meta Llama 3.3 70B | Versi terbaru |
| NVIDIA Nemotron 4 340B | Model terkuat NVIDIA |
| Mistral Large 2 | Multilingual excellent |
| DeepSeek R1 | Reasoning expert |
| Qwen 3 235B | Multilingual, context panjang |

## ⚙️ Pengaturan Lanjut

| Setting | Default | Deskripsi |
|---------|---------|-----------|
| Chunk Size | 1000 | Ukuran potongan teks. Lebih kecil = lebih detail |
| Chunk Overlap | 200 | Tumpang tindih antar chunks untuk menjaga konteks |
| Jumlah Konteks (k) | 5 | Jumlah chunk yang diambil untuk menjawab |
| Temperature | 0.0 | Semakin rendah = semakin fokus |

## 🔒 Catatan Keamanan

- API key Anda tidak disimpan secara permanen
- Dokumen diproses di memori dan tidak disimpan
- Hati-hati saat mengunggah dokumen rahasia — data dikirim ke NVIDIA NIM API
- NVIDIA NIM menyediakan 1.000+ inference credits gratis tanpa kartu kredit
- Patuhi kebijakan privasi dan keamanan organisasi Anda

## 📂 Struktur Proyek

```
smart-document-insights/
├── app.py                         # Entry point (streamlit run app.py)
├── src/
│   └── smart_doc/
│       ├── __init__.py            # Package init
│       ├── config.py              # Konfigurasi & constants
│       ├── core/                  # Business logic
│       │   ├── document.py        # Document loading & stats
│       │   ├── embeddings.py      # Embeddings & vector store (NVIDIA NIM)
│       │   ├── rag.py             # RAG chain logic
│       │   └── prompts.py         # System prompts
│       ├── ui/                    # Streamlit UI components
│       │   ├── sidebar.py         # Sidebar config
│       │   ├── chat.py            # Chat tab
│       │   ├── summary.py         # Summary tab
│       │   ├── insights.py        # Key insights tab
│       │   └── stats.py           # Document stats
│       └── utils/                 # Utilities
│           ├── file.py            # File handling
│           └── export.py          # Export to Markdown
├── tests/                         # Unit tests (20 tests)
│   ├── test_imports.py
│   ├── test_core.py
│   └── test_utils.py
├── pyproject.toml                 # Project config
├── requirements.txt               # Dependencies
├── README.md
└── .gitignore
```
