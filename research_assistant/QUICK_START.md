# Quick Start Guide

## 1. Setup (One-time)

The virtual environment has been created and dependencies installed! 

**To activate the virtual environment:**

```bash
cd research_assistant
source venv/bin/activate
```

Or use the convenience script:
```bash
cd research_assistant
source activate.sh
```

**If you need to reinstall dependencies:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Set API Key

```bash
export PERPLEXITY_API_KEY="your-key-here"
```

Or create a `.env` file:
```
PERPLEXITY_API_KEY=your-key-here
```

## 3. Add PDFs

Place your research papers in the `pdfs/` folder:
```bash
mkdir -p pdfs
# Copy your PDF files to pdfs/
```

## 4. Run the App

```bash
streamlit run main.py
```

## 5. Use the App

1. In the sidebar, enter your PDF folder path (default: `./pdfs`)
2. Click **"Build Index"** - wait for it to complete
3. Enter a research question in the main area
4. Click **"Research"** and wait for the answer!

## Example Research Questions

Try these to get started:

- **"Summarize the main contributions of these papers"**
- **"What are the key limitations mentioned?"**
- **"Compare the different methods proposed"**
- **"What datasets were used in these studies?"**
- **"What are the future research directions suggested?"**

## Troubleshooting

- **First run is slow**: The embedding model downloads on first use (~80MB)
- **API key error**: Make sure `PERPLEXITY_API_KEY` is set
- **No PDFs found**: Check the folder path in the sidebar

## Testing Without PDFs

If you want to test the system but don't have PDFs yet, you can:
1. Download a sample research paper (e.g., from arXiv)
2. Save it as `pdfs/sample.pdf`
3. Build the index and try a question!

