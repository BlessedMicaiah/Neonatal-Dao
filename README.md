# Neonatal Dao

## Overview
Neonatal Dao is an advanced AI assistant specializing in neonatal healthcare. It leverages cutting-edge language models, retrieval-augmented generation (RAG), and specialized medical knowledge to provide accurate and reliable information to healthcare professionals working with newborns.

## Features
- **Medical Q&A**: Provides accurate answers to neonatal healthcare questions
- **Research Paper Integration**: Analyzes and incorporates insights from latest medical research papers
- **Knowledge Base Retrieval**: Uses RAG to improve response quality with relevant medical literature
- **Web Search Capability**: Supplements knowledge with real-time information from trusted sources
- **Document Analysis**: Processes and extracts information from medical documents
- **Modern Web Interface**: User-friendly React frontend for seamless interaction

## Project Structure
```
Neonata Dao/
├── data/ # Training and knowledge base data
│ ├── medical_qa.json # Q&A pairs for fine-tuning
│ ├── vocab.txt # Medical vocabulary for tokenizer
│ ├── research_papers/ # Directory for health-related research papers
│ │ ├── raw/ # Raw paper files (e.g., PDFs, XML)
│ │ ├── processed/ # Preprocessed text or embeddings
│ │ └── metadata.json # Metadata for papers (e.g., title, authors, DOI)
│ └── knowledge_base/ # RAG data (e.g., PDFs, text files)
│     ├── articles/
│     ├── research_papers/ # Symlink or copied papers for RAG
│     └── index.faiss # Vector index for retrieval
├── model/ # Model files
│ ├── base_model/ # Pre-trained base model
│ └── fine_tuned/ # Fine-tuned weights (LoRA adapters)
├── src/ # Backend source code
│ ├── tokenizer.py # Custom medical tokenizer
│ ├── fine_tune.py # Fine-tuning script
│ ├── inference.py # Inference engine with RAG
│ ├── rag/ # RAG-specific components
│ │ ├── retriever.py # Retrieval logic
│ │ └── indexer.py # Knowledge base indexing
│ ├── tools/ # Additional tools
│ │ ├── web_search.py
│ │ ├── doc_analyzer.py
│ │ └── paper_processor.py # Script to process research papers
│ ├── training/ # Training-related scripts
│ │ └── paper_trainer.py # Script to train on research papers
│ └── api.py # FastAPI backend
├── frontend/ # React frontend
│ ├── public/ # Static assets
│ │ ├── index.html # HTML template
│ │ └── favicon.ico # Icon (optional)
│ ├── src/ # React source code
│ │ ├── components/ # Reusable UI components
│ │ │ ├── Chat.js # Chat window component
│ │ │ ├── Message.js # Single message component
│ │ │ └── Input.js # Text input component
│ │ ├── App.js # Main app component
│ │ ├── App.css # Styles
│ │ ├── index.js # Entry point
│ │ └── api.js # API client for backend calls
│ ├── package.json # Dependencies and scripts
│ └── README.md # Frontend-specific docs
├── config.yaml # Configuration file
├── requirements.txt # Backend dependencies
└── README.md # Project-wide documentation
```

## Installation

### Prerequisites
- Python 3.9+
- Node.js 16+
- GPU with CUDA support (recommended for training and inference)

### Backend Setup
1. Clone the repository:
   ```
   git clone https://github.com/your-organization/neonatal-dao.git
   cd neonatal-dao
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the base model (if not included):
   ```
   python src/tools/model_downloader.py
   ```

5. Set up the knowledge base:
   ```
   python src/rag/indexer.py
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Usage

### Running the Backend
1. Start the backend API server:
   ```
   python src/api.py
   ```
   The API will be available at `http://localhost:8000`

### Running the Frontend
1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Start the development server:
   ```
   npm start
   ```
   The frontend will be available at `http://localhost:3000`

### Training and Fine-tuning
To fine-tune the model on your own data:
```
python src/fine_tune.py --config config.yaml
```

### Processing Research Papers
To process and index new research papers:
```
python src/tools/paper_processor.py --input data/research_papers/raw --output data/research_papers/processed
```

## API Documentation
The API documentation is available at `http://localhost:8000/docs` when the backend server is running.

## Contributing
Contributions to Neonatal Dao are welcome! Please refer to our contributing guidelines for more information.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
- Thanks to the medical professionals who provided expert knowledge and validation
- Research papers and datasets that have been integrated into our knowledge base
- Open-source AI and NLP community for their invaluable tools and libraries
