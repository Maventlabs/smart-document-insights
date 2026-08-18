# Panduan Testing - Smart Document Insights

## Cara Menjalankan App

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan
streamlit run app.py

# 3. Buka browser → http://localhost:8501
# 4. Masukkan NVIDIA NIM API Key (gratis di https://build.nvidia.com)
# 5. Upload dokumen → mulai tanya!
```

---

## Dokumen Sample Gratis (Download)

### 1. Research Paper (Ilmiah)
| Dokumen | Link | Tipe |
|---------|------|------|
| Attention Is All You Need (Transformer) | https://arxiv.org/pdf/1706.03762 | PDF, 15 halaman |
| Bitcoin Whitepaper | https://bitcoin.org/bitcoin.pdf | PDF, 9 halaman |
| BERT Paper | https://arxiv.org/pdf/1810.04805 | PDF, 16 halaman |

### 2. Laporan Keuangan
| Dokumen | Link | Tipe |
|---------|------|------|
| Apple 10-K Annual Report (contoh) | https://investor.apple.com/sec-filings/default.aspx | PDF |
| Tesla 10-K Annual Report | https://ir.tesla.com/sec-filings/annual-reports | PDF |

### 3. Dokumen Hukum / Kontrak
| Dokumen | Link | Tipe |
|---------|------|------|
| NDA Template (Stanford) | https://e.gزrge.io/pdf/n.pdf | PDF |
| Creative Commons License | https://creativecommons.org/licenses/by/4.0/legalcode | Web → PDF |

### 4. Dokumen Umum
| Dokumen | Link | Tipe |
|---------|------|------|
| UN Declaration of Human Rights | https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/eng.pdf | PDF |
| Wikipedia Article (export as PDF) | Export dari Wikipedia | PDF |

---

## Prompt Testing per Kategori

### A. Tanya Jawab Dasar (Tab: Tanya Jawab)

#### Research Paper (Attention Is All You Need)
```
1. Apa topik utama dari paper ini?
2. Siapa saja penulis dari paper ini?
3. Apa itu mekanisme "attention" yang dijelaskan dalam paper?
4. Bagaimana arsitektur Transformer berbeda dari model RNN/LSTM?
5. Apa dataset yang digunakan untuk evaluasi?
6. Berapa BLEU score yang dicapai model Transformer?
7. Apa keunggulan utama Transformer dibandingkan model sebelumnya?
8. Jelaskan konsep "multi-head attention" secara sederhana
```

#### Bitcoin Whitepaper
```
1. Siapa yang menulis whitepaper ini (nama samaran)?
2. Apa masalah yang ingin diselesaikan oleh Bitcoin?
3. Bagaimana cara kerja blockchain menurut paper ini?
4. Apa itu "proof of work" dan bagaimana cara kerjanya?
5. Berapa lama rata-rata waktu untuk mengkonfirmasi transaksi?
6. Apa risiko atau kelemahan yang disebutkan dalam paper?
7. Jelaskan tentang "double spending problem" dan solusinya
8. Apa yang terjadi jika sebuah node jaringan offline?
```

#### BERT Paper
```
1. Apa perbedaan utama antara BERT dan GPT?
2. Apa itu "masked language modeling"?
3. Bagaimana BERT dilatih (pre-training)?
4. Apa tugas NLP yang bisa diselesaikan oleh BERT?
5. Berapa jumlah parameter dalam model BERT?
```

### B. Ringkasan Dokumen (Tab: Ringkasan Dokumen)

```
# Klik tombol "Generate Ringkasan" lalu cek hasilnya
# Harusnya menghasilkan:
- Ringkasan dalam bahasa yang sama dengan dokumen
- Poin-poin utama dalam format bullet point
- Struktur yang jelas dan terorganisir
```

### C. Key Insights (Tab: Key Insights)

#### Tanpa Topik Spesifik
```
# Klik "Ekstrak Insights" tanpa mengisi topik
# Harusnya menghasilkan:
- Temuan utama dari dokumen
- Angka atau data penting
- Potensi risiko atau peluang
- Rekomendasi (jika ada)
```

#### Dengan Topik Spesifik
```
# Isi topik spesifik lalu klik "Ekstrak Insights":

# Untuk paper AI:
topik: "comparative analysis with previous approaches"

# Untuk whitepaper:
topik: "technical architecture and consensus mechanism"

# Untuk laporan keuangan:
topik: "revenue growth and risk factors"

# Untuk kontrak:
topik: "key obligations and liability clauses"
```

### D. Multi-Document Testing

Upload 2-3 dokumen sekaligus, lalu tanya:
```
1. Apa kesamaan antara dokumen-dokumen ini?
2. Bandingkan pendekatan yang digunakan dalam dokumen-dokumen ini
3. Mana yang lebih relevan untuk topik X?
4. Buat ringkasan gabungan dari semua dokumen
```

### E. Stress Testing

```
# Test batas kemampuan:
1. "Jelaskan SEMUA konsep teknis dalam paper ini satu per satu"
2. "Buat daftar lengkap semua angka dan statistik yang disebutkan"
3. "Apa saja referensi yang dikutip dalam dokumen?"
4. "Jelaskan bagian yang paling sulit dalam dokumen ini"
```

---

## Checklist Testing

- [ ] App bisa dijalankan (`streamlit run app.py`)
- [ ] API key validation bekerja (warning jika kosong)
- [ ] File upload berhasil (PDF, TXT, DOCX)
- [ ] File type validation bekerja (reject .exe, .jpg, dll)
- [ ] Document stats ditampilkan dengan benar
- [ ] Chat Q&A berfungsi dengan benar
- [ ] Source context bisa di-expand dan dibaca
- [ ] Ringkasan dokumen dihasilkan dengan benar
- [ ] Key insights bisa diekstrak
- [ ] Export chat history ke Markdown berfungsi
- [ ] Export ringkasan ke Markdown berfungsi
- [ ] Export insights ke Markdown berfungsi
- [ ] Chat history di-clear saat upload dokumen baru
- [ ] Model selection berfungsi
- [ ] Temperature slider berfungsi
- [ ] Chunk size & overlap settings berfungsi
- [ ] Error handling untuk API key salah
- [ ] Error handling untuk dokumen kosong
- [ ] App tidak crash saat network error

---

## Tips Testing

1. **Mulai dari yang sederhana** — upload dokumen pendek dulu (Bitcoin whitepaper)
2. **Test satu fitur pada satu waktu** — jangan campur aduk
3. **Catat hasil** — screenshot jika perlu
4. **Test edge case** — upload file kosong, file corrupt, dll
5. **Bandungkan model** — coba dengan Nemotron 3.5 Lightning vs Nemotron 3 Ultra
