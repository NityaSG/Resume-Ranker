# Resume Ranker

This Python script uses the BERT NLP model to find the most suitable resume/candidate based on a description provided by you.

## Requirements

- Python 3.x
- numpy, torch, PyPDF2, transformers, and scikit-learn libraries

## Installation

1. Install the required dependencies by running `pip install -r requirements.txt` in your command prompt or terminal.

## Usage

1. Update the `folder_path` variable in the script with the path to your folder of PDF files.
2. Update the `description` variable in the script with your desired description.
3. Run the script by running `python pdf_similarity_search.py` in your command prompt or terminal.

The script will extract text from each PDF file in the specified folder and use the BERT NLP model to generate embeddings for your description and the text from each PDF file. It will then calculate the cosine similarity between the embeddings to determine how relevant each PDF file is to your description and rank the PDF files based on their similarity scores.
