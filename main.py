import os
import re
import pandas as pd
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

# Set your Groq API key
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key

def ingest_excel(file_path):
    """
    Read Excel file and extract metadata with explicit engine specification
    """
    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Choose appropriate engine based on file extension
    if file_extension == '.xlsx':
        engine = 'openpyxl'  # For .xlsx files
    elif file_extension == '.xls':
        engine = 'xlrd'      # For .xls files
    else:
        # For other formats, try openpyxl first
        engine = 'openpyxl'
    
    try:
        # Read the Excel file with explicit engine
        df = pd.read_excel(file_path, engine=engine)
    except Exception as e:
        if engine == 'openpyxl':
            # If openpyxl fails, try xlrd
            print(f"Failed to read with openpyxl, trying xlrd: {str(e)}")
            try:
                df = pd.read_excel(file_path, engine='xlrd')
            except Exception as e2:
                raise ValueError(f"Could not read file with any Excel engine: {str(e2)}")
        else:
            raise ValueError(f"Failed to read Excel file: {str(e)}")
    
    # Get basic statistics for context
    stats = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isna().sum().to_dict(),
        "data_types": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "sample_data": df.head(5).to_dict('records')
    }
    
    return df, stats
def setup_llama_model():
    """
    Initialize Llama model through Groq
    """
    llm = ChatGroq(
        model_name="llama3-70b-8192",  # or "llama3-70b-8192" for better performance
        temperature=0.1,
        max_tokens=2000,
    )
    
    return llm

def create_analysis_chain(llm):
    """
    Create LLM chain for data analysis
    """
    analysis_prompt = ChatPromptTemplate.from_template("""
    I have an Excel dataset with the following properties:
    
    Columns: {columns}
    Shape: {shape}
    Missing values: {missing_values}
    Data types: {data_types}
    
    Sample data:
    {sample_data}
    
    Please provide a list of data improvement actions I could take to enhance this dataset 
    for business purposes. For each action, explain why it would be beneficial.
    Format your response as a numbered list of specific actions.
    """)
    
    analysis_chain = LLMChain(
        llm=llm,
        prompt=analysis_prompt,
        verbose=True
    )
    
    return analysis_chain

def parse_action_items(llm_response):
    """
    Extract action items from LLM response
    """
    action_items = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)', llm_response, re.DOTALL)
    cleaned_items = [item.strip() for item in action_items]
    return cleaned_items

def create_code_generation_chain(llm):
    """
    Create LLM chain for code generation
    """
    code_prompt = ChatPromptTemplate.from_template("""
    I have an Excel dataset with columns: {columns}
    
    I want to perform this action: "{action_item}"
    
    Generate Python code using pandas that will perform this action on a DataFrame called 'df'.
    The code should be complete, well-commented, and handle edge cases.
    Return only the Python code without any additional explanations.
    ```python
    # Your code here
    ```
    """)
    
    code_generation_chain = LLMChain(
        llm=llm,
        prompt=code_prompt,
        verbose=True
    )
    
    return code_generation_chain

def extract_code(response):
    """
    Extract Python code from LLM response
    """
    code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return response.strip()

def execute_action(df, code):
    """
    Execute generated code on DataFrame
    """
    namespace = {"df": df.copy(), "pd": pd}
    
    try:
        exec(code, namespace)
        result_df = namespace["df"]
        return result_df, None
    except Exception as e:
        return df, str(e)

def detect_changes(original_df, new_df):
    """
    Detect changes between original and modified DataFrames
    """
    changes = {
        "columns_added": list(set(new_df.columns) - set(original_df.columns)),
        "columns_removed": list(set(original_df.columns) - set(new_df.columns)),
        "row_count_before": len(original_df),
        "row_count_after": len(new_df),
        "null_values_before": original_df.isna().sum().sum(),
        "null_values_after": new_df.isna().sum().sum()
    }
    return changes

def generate_report(input_file, output_file, results):
    """
    Generate markdown report of actions taken
    """
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count
    
    report = f"""
    # Excel Transformation Report
    
    Input file: {input_file}
    Output file: {output_file}
    Total actions: {len(results)}
    Successful actions: {success_count}
    Failed actions: {failed_count}
    
    ## Action Details
    
    """
    
    for i, result in enumerate(results):
        report += f"### Action {i+1}: {result['action']}\n"
        report += f"Status: {result['status']}\n\n"
        
        if result["status"] == "success":
            report += "Changes:\n"
            for key, value in result["changes"].items():
                report += f"- {key}: {value}\n"
        else:
            report += f"Error: {result['error']}\n"
        
        report += f"\nCode:\n```python\n{result['code']}\n```\n\n"
    
    with open(output_file.replace('.xlsx', '_report.md'), 'w') as f:
        f.write(report)

def run_pipeline(input_file, output_file):
    """
    Run the full data transformation pipeline
    """
    # Step 1: Ingest data
    df, stats = ingest_excel(input_file)
    print(f"Loaded Excel file with {stats['shape'][0]} rows and {stats['shape'][1]} columns")
    
    # Step 2: Setup LLM via Groq
    llm = setup_llama_model()
    
    # Step 3: Create chains
    analysis_chain = create_analysis_chain(llm)
    code_generation_chain = create_code_generation_chain(llm)
    
    # Step 4: Get analysis from LLM
    print("Analyzing data with LLM...")
    analysis_result = analysis_chain.run(
        columns=stats['columns'],
        shape=stats['shape'],
        missing_values=stats['missing_values'],
        data_types=stats['data_types'],
        sample_data=stats['sample_data']
    )
    
    # Step 5: Parse action items
    action_items = parse_action_items(analysis_result)
    print(f"Identified {len(action_items)} action items:")
    for i, action in enumerate(action_items):
        print(f"{i+1}. {action}")
    
    # Step 6: Generate and execute code for each action
    results = []
    current_df = df.copy()
    
    for i, action in enumerate(action_items):
        print(f"\nProcessing action {i+1}/{len(action_items)}: {action}")
        
        # Generate code
        code_result = code_generation_chain.run(
            columns=current_df.columns.tolist(),
            action_item=action
        )
        
        # Extract code from response
        code = extract_code(code_result)
        print(f"\nGenerated code:\n{code}\n")
        
        # Execute the code
        new_df, error = execute_action(current_df, code)
        
        if error:
            print(f"Error executing action {i+1}: {error}")
            results.append({
                "action": action,
                "code": code,
                "status": "failed",
                "error": error
            })
        else:
            changes = detect_changes(current_df, new_df)
            print(f"Action completed. Changes: {changes}")
            current_df = new_df
            results.append({
                "action": action,
                "code": code,
                "status": "success",
                "changes": changes
            })
    
    # Save final result
    current_df.to_excel(output_file, index=False)
    
    # Generate summary report
    generate_report(input_file, output_file, results)
    
    return results, current_df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Excel Data Transformation Pipeline')
    parser.add_argument('--input', required=True, help='Input Excel file path')
    parser.add_argument('--output', required=True, help='Output Excel file path')
    parser.add_argument('--model', default="llama3-70b-8192", 
                        choices=["llama3-70b-8192", "llama3-70b-8192"],
                        help='Which Llama model to use via Groq')
    
    args = parser.parse_args()
    
    try:
        results, transformed_df = run_pipeline(args.input, args.output)
        
        print(f"\nPipeline completed. Transformed file saved to {args.output}")
        print(f"Report saved to {args.output.replace('.xlsx', '_report.md')}")
        
    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()