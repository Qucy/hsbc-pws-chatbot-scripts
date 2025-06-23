import os
import sys
from pathlib import Path
try:
    import pypandoc
except ImportError:
    print("Error: pypandoc is not installed. Please install it using: pip install pypandoc")
    sys.exit(1)

def convert_md_to_docx(input_path):
    """
    Convert a Markdown file to a Word document (.docx)
    
    Args:
        input_path (str): Path to the input Markdown file
    
    Returns:
        str: Path to the generated .docx file
    """
    # Convert input path to Path object for easier manipulation
    input_file = Path(input_path)
    
    # Check if input file exists
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check if input file has .md extension
    if input_file.suffix.lower() not in ['.md', '.markdown']:
        raise ValueError(f"Input file must be a Markdown file (.md or .markdown): {input_path}")
    
    # Generate output path by changing extension to .docx
    output_path = input_file.with_suffix('.docx')
    
    try:
        # Convert MD to DOCX using pypandoc
        pypandoc.convert_file(
            str(input_file),
            'docx',
            outputfile=str(output_path)
            # Removed problematic --reference-doc= and --toc arguments
        )
        
        print(f"Successfully converted: {input_file} -> {output_path}")
        return str(output_path)
        
    except Exception as e:
        raise RuntimeError(f"Error during conversion: {str(e)}")

def main():
    """
    Main function to handle command line arguments
    """
    if len(sys.argv) != 2:
        print("Usage: python md_to_docx_converter.py <path_to_markdown_file>")
        print("Example: python md_to_docx_converter.py document.md")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    try:
        output_path = convert_md_to_docx(input_path)
        print(f"Conversion completed successfully!")
        print(f"Output file: {output_path}")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()