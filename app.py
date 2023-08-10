import numpy as np
import torch
import os
import PyPDF2
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Set the path to your folder of PDF files and your description
folder_path = './data'
description = 'NextJs Tailwind Css Typescript Git & Github Good understanding of UI/UXResponsiveness (mobile view for websites'

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Preprocess your description
description_tokens = tokenizer.encode(description, add_special_tokens=True)
description_input_ids = torch.tensor([description_tokens])
with torch.no_grad():
    description_output = model(description_input_ids)
description_embedding = description_output[0][:,0,:].numpy()

# Initialize a list to store the similarity scores for each PDF file
similarity_scores = []

# Loop through each PDF file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.pdf'):
        # Extract text from the PDF file
        pdf_file = open(os.path.join(folder_path, filename), 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pdf_text = ''
        for page in range(len(pdf_reader.pages)):
            page_obj = pdf_reader.pages[page]
            pdf_text += page_obj.extract_text()
        pdf_file.close()
        count=0
        keywords = ['react', 'javascript', 'HTML']
        if not any(keyword.lower() in pdf_text.lower() for keyword in keywords):
            continue
        pdf_tokens = tokenizer.encode(pdf_text, add_special_tokens=True)
        pdf_chunks = [pdf_tokens[i:i+512] for i in range(0, len(pdf_tokens), 512)]

        # Generate embeddings for each chunk
        pdf_embeddings = []
        for chunk in pdf_chunks:
            chunk_input_ids = torch.tensor([chunk])
            with torch.no_grad():
                chunk_output = model(chunk_input_ids)
            chunk_embedding = chunk_output[0][:,0,:].numpy()
            pdf_embeddings.append(chunk_embedding)

        # Average the embeddings for each chunk to get a single embedding for the entire text
        pdf_embedding = np.mean(pdf_embeddings, axis=0)

        # Calculate the similarity between the description and the PDF text
        similarity = cosine_similarity(description_embedding, pdf_embedding)[0][0]
        similarity_scores.append((filename, similarity))

# Sort the PDF files by their similarity scores
similarity_scores.sort(key=lambda x: x[1], reverse=True)

# Print the ranked list of PDF files
for filename, similarity in similarity_scores:
    print(f'{filename}: {similarity}')
