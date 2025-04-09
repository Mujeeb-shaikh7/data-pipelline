# Excel Transformation Pipeline

A tool for automated Excel data improvement using LLaMA via Groq and LangChain.

## Overview

This pipeline analyzes Excel files using the LLaMA language model (accessed through Groq's API) to identify potential improvements, automatically generates Python code to implement these improvements, and then executes the generated code to transform the data.

## Features

- Automated analysis of Excel data structure and content
- AI-powered identification of improvement opportunities 
- Automated Python code generation for data transformations
- Safe code execution environment
- Detailed reporting of changes made
- Support for various Excel formats

## Installation

### Prerequisites

- Python 3.8 or higher
- A Groq API key (sign up at https://console.groq.com/)

### Setup


2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages
   ```bash
   pip install langchain langchain-groq pandas openpyxl xlrd
   ```

4. Set your Groq API key
   ```bash
   # On Windows
   set GROQ_API_KEY=your-groq-api-key
   # On macOS/Linux
   export GROQ_API_KEY=your-groq-api-key
   ```
   
   Alternatively, you can modify the script to include your API key:
   ```python
   os.environ["GROQ_API_KEY"] = "your-groq-api-key"  # Replace with your actual API key
   ```

## Usage

Run the pipeline with:

```bash
python excel_pipeline.py --input your_data.xlsx --output transformed_data.xlsx
```

### Command-line arguments

- `--input`: Path to the input Excel file (required)
- `--output`: Path to save the transformed Excel file (required)
- `--model`: Llama model to use via Groq (default: "llama3-8b-8192", options: "llama3-8b-8192", "llama3-70b-8192")

### Example

```bash
python excel_pipeline.py --input customer_data.xlsx --output improved_customer_data.xlsx --model llama3-70b-8192
```

## How It Works

1. **Data Ingestion**: The pipeline reads the Excel file and extracts its structure and basic statistics.
2. **Analysis**: The LLM analyzes the data and identifies potential improvements.
3. **Action Planning**: The suggestions are parsed into discrete action items.
4. **Code Generation**: For each action item, Python code is generated using pandas.
5. **Execution**: The generated code is safely executed against the data.
6. **Reporting**: Changes are tracked and documented in a comprehensive Markdown report.

## Output

The pipeline produces two files:
1. The transformed Excel file at the specified output path
2. A detailed Markdown report (`output_file_report.md`) listing all actions taken, code executed, and changes made

## Customization

You can modify the prompts in `create_analysis_chain()` and `create_code_generation_chain()` to better suit your specific data needs.

## Limitations

- The pipeline can only process Excel files (xlsx, xls)
- Complex transformations may require manual review of the generated code
- The quality of suggestions depends on the LLM's understanding of your data

## Troubleshooting

- **Excel Engine Errors**: If you encounter issues reading Excel files, make sure you have both `openpyxl` and `xlrd` installed
- **API Key Errors**: Verify your Groq API key is set correctly
- **Memory Issues**: For large Excel files, consider processing in batches

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses [LangChain](https://github.com/langchain-ai/langchain) for LLM integration
- LLaMA models are accessed through [Groq's API](https://console.groq.com/)