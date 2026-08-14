# 📄 NLP Legal Clause Extractor

Extract **termination**, **payment**, **penalty**, and **risk** clauses from legal contracts using NLP.

## 🧠 NLP Pipeline

| Stage | Library | Purpose |
|-------|---------|---------|
| Document ingestion | `pypdf`, `python-docx` | Extract text from PDF/DOCX/TXT |
| Segmentation | `spaCy` | Split into sentences & sections |
| Entity extraction | `spaCy NER` | Extract money, dates, percentages |
| Clause classification | `DistilBERT zero-shot` | Categorize clauses |
| Summarization | `spaCy` + templates | Generate summaries |


## 🚀 Quick Start
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run application
python app.py
Open http://localhost:3000 in your browser.
```

📁 Project Structure
text
nlp-legal-clause-extractor/
├── app.py                 # FastAPI server
├── nlp/
│   ├── segmentation.py    # Sentence splitting
│   ├── classification.py  # Zero-shot classification
│   └── summarization.py   # Summary generation
├── demo_nlp.py           # CLI demo
├── clause-extractor.html # Web interface
├── requirements.txt      # Dependencies
└── README.md            # This file


🔌 API Endpoints
Method	Endpoint	Description
POST	/api/extract	Upload file or text for analysis
GET	/api/health	Check service status

API Examples
# Upload file
curl -X POST -F "file=@contract.pdf" http://localhost:3000/api/extract
# Send text
curl -X POST -F "text=Contract terms..." http://localhost:3000/api/extract


💻 CLI Demo
bash
# Run demo with sample text
python demo_nlp.py
# Analyze specific file
python demo_nlp.py --file contract.pdf


📦 Dependencies
fastapi - Web framework
spaCy - NLP processing
transformers - Zero-shot classification
pypdf - PDF parsing
python-docx - DOCX parsing

🎯 Features
✅ Multi-format support (PDF, DOCX, TXT, HTML)
✅ Zero-shot classification (no training needed)
✅ Entity extraction (money, dates, percentages)
✅ Web interface & REST API
✅ Structured clause summaries

📄 License
MIT License - Free for personal and commercial use.

🤝 Contributing
Fork the repository
Create feature branch (git checkout -b feature/amazing)
Commit changes (git commit -m 'Add amazing feature')

Push (git push origin feature/amazing)

Open Pull Request

