# Banking Chatbot Feedback Analysis Pipeline

A comprehensive Python toolkit for analyzing banking chatbot feedback data, generating detailed reports, and converting documentation to various formats.

## 🚀 Features

### Core Analysis Pipeline (`feedback_analyze_pipeline.py`)
- **Question Categorization**: Automatically categorizes banking questions into predefined categories
- **Feedback Comment Analysis**: Analyzes and categorizes user feedback comments
- **Scenario Mapping**: Maps questions to specific banking scenarios
- **Async Processing**: High-performance async processing with configurable concurrency
- **Comprehensive Reporting**: Generates detailed analysis summaries

### Report Generation (`feedback_report_generator.py`)
- **Automated Report Generation**: Creates comprehensive markdown reports from analyzed data
- **Temporal Analysis**: Analyzes feedback patterns over time
- **Category Performance**: Evaluates performance across different banking categories
- **Content Quality Analysis**: Assesses response quality and user satisfaction
- **Negative Feedback Analysis**: Identifies common complaint patterns
- **Actionable Recommendations**: Generates data-driven improvement suggestions

### Document Conversion (`md_to_docx_converter.py`)
- **Markdown to Word**: Converts generated markdown reports to Word documents
- **Simple CLI Interface**: Easy-to-use command-line tool
- **Error Handling**: Robust error handling and validation

## 📋 Prerequisites

- Python 3.8+
- DeepSeek API access (for AI-powered analysis)
- Required Python packages (see Installation)

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install required packages:**
```bash
pip install openai python-dotenv pandas numpy matplotlib seaborn pypandoc

3. **Install pypandoc system dependency:**
```bash
# macOS
brew install pandoc

# Ubuntu/Debian
sudo apt-get install pandoc

# Windows
# Download and install from: https://pandoc.org/installing.html
```

## ⚙️ Configuration

1. **Create a `.env` file in the project directory:**
```env
# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Processing Configuration
MAX_CONCURRENT_REQUESTS=5
REQUEST_DELAY=1.0
```

2. **Copy the example configuration:**
```bash
cp .env.example .env
# Edit .env with your actual API key
```

## 📊 Data Requirements

Place your data files in the `data/` directory:

### Required Files:
- `pws_chatbot_qa_feedbacks.csv` - Main feedback data
- `mapped_questions.csv` - Question mapping data

### Expected CSV Format:
```csv
timestamp,question,response,feedback_rating,feedback_comment
2025-01-01 10:00:00,"How do I reset my password?","You can reset...",THUMBS_UP,"Very helpful"
```

## 🚀 Usage

### 1. Run Complete Analysis Pipeline
```bash
python feedback_analyze_pipeline.py
```

**Interactive Menu Options:**
- `1`: Process comments only
- `2`: Process categories only  
- `3`: Process feedback comment categories only
- `4`: Process scenarios only
- `5`: Merge processed files
- `6`: Run full analysis (recommended)

### 2. Generate Comprehensive Reports
```bash
python feedback_report_generator.py
```

This generates a detailed markdown report including:
- Executive summary with key metrics
- Temporal analysis and trends
- Category performance breakdown
- Content quality assessment
- Negative feedback analysis
- Actionable recommendations

### 3. Convert Reports to Word Documents
```bash
python md_to_docx_converter.py feedback_analysis_report.md
```

### 4. Complete Workflow Example
```bash
# Step 1: Run full analysis
python feedback_analyze_pipeline.py
# Choose option 6 (Run full analysis)

# Step 2: Generate report
python feedback_report_generator.py

# Step 3: Convert to Word document
python md_to_docx_converter.py feedback_analysis_report.md
```

## 📁 Output Files

### Analysis Pipeline Outputs:
- `pws_chatbot_qa_feedbacks_with_categories.csv` - Data with question categories
- `pws_chatbot_qa_feedbacks_with_feedback_categories.csv` - Data with feedback categories
- `pws_chatbot_qa_feedbacks_with_scenarios.csv` - Data with scenario mappings
- `pws_chatbot_qa_feedbacks_final.csv` - Complete processed dataset
- `analysis_summary.txt` - Comprehensive analysis summary

### Report Generator Outputs:
- `feedback_analysis_report.md` - Detailed markdown report

### Document Converter Outputs:
- `feedback_analysis_report.docx` - Word document version

## 🔧 Configuration Options

### Environment Variables:
- `DEEPSEEK_API_KEY`: Your DeepSeek API key (required)
- `DEEPSEEK_BASE_URL`: API base URL (default: https://api.deepseek.com)
- `MAX_CONCURRENT_REQUESTS`: Concurrent API requests (default: 5)
- `REQUEST_DELAY`: Delay between requests in seconds (default: 1.0)

### Performance Tuning:
- **High-volume processing**: Increase `MAX_CONCURRENT_REQUESTS` to 10-15
- **Rate limiting issues**: Increase `REQUEST_DELAY` to 2.0+
- **Memory constraints**: Process data in smaller batches

## 📈 Report Features

The generated reports include:

### Executive Summary
- Total feedback entries processed
- Overall satisfaction rates
- Key performance indicators

### Temporal Analysis
- Feedback trends over time
- Peak activity periods
- Seasonal patterns

### Category Performance
- Performance by banking service category
- Success rates and common issues
- Category-specific recommendations

### Content Quality Analysis
- Response effectiveness metrics
- User satisfaction correlation
- Quality improvement suggestions

### Negative Feedback Analysis
- Common complaint patterns
- Root cause identification
- Priority improvement areas

## 🛠️ Troubleshooting

### Common Issues:

**API Key Errors:**
```
ValueError: API key is required
```
- Ensure `.env` file exists with valid `DEEPSEEK_API_KEY`
- Check API key format and permissions

**File Not Found Errors:**
```
FileNotFoundError: pws_chatbot_qa_feedbacks.csv not found
```
- Verify data files are in the `data/` directory
- Check file names match exactly

**Memory Issues:**
- Reduce `MAX_CONCURRENT_REQUESTS`
- Process data in smaller chunks
- Close other applications

**Conversion Errors:**
```
Error: pypandoc is not installed
```
- Install pypandoc: `pip install pypandoc`
- Install pandoc system dependency

### Performance Tips:
- Use SSD storage for faster file I/O
- Ensure stable internet connection for API calls
- Monitor API rate limits
- Use appropriate concurrency settings

## 🔒 Security Notes

- **Never commit `.env` files** to version control
- Store API keys securely
- Use environment-specific configurations
- Regularly rotate API keys
- Review data privacy requirements

## 📝 File Structure
```
feedback_analyze/
├── feedback_analyze_pipeline.py    # Main analysis pipeline
├── feedback_report_generator.py    # Report generation
├── md_to_docx_converter.py         # Document conversion
├── .env                            # Environment configuration
├── .env.example                    # Example configuration
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
└── data/                          # Data directory
    ├── pws_chatbot_qa_feedbacks.csv
    ├── mapped_questions.csv
    └── [output files]
```

## 🤝 Contributing

When contributing:
1. Follow existing code style
2. Add appropriate error handling
3. Update documentation
4. Test with sample data
5. Ensure security best practices

## Additional Files to Create

### `.env.example`
```env:/Users/qucy/Projects/PythonCodePractise/pws-chatbot-scripts/feedback_analyze/.env.example
# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Processing Configuration (Optional)
MAX_CONCURRENT_REQUESTS=5
REQUEST_DELAY=1.0
```

## Summary

The `feedback_report_generator.py` file is clean and doesn't require any API key removal. I've created a comprehensive README that incorporates all three Python files in your project:

1. **`feedback_analyze_pipeline.py`** - Main analysis with AI categorization
2. **`feedback_report_generator.py`** - Report generation and analytics
3. **`md_to_docx_converter.py`** - Document format conversion

The README now provides:
- Complete workflow documentation
- Usage examples for all three tools
- Comprehensive troubleshooting guide
- Security best practices
- Performance optimization tips

All files are ready for sharing with proper environment variable configuration and no hardcoded sensitive information.
        
