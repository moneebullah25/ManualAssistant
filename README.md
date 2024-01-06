# Manual Assistant

## Overview

Manual Assistant is a tool that combines document embedding, vector search, and a question-answering chain to provide users with accurate answers based on uploaded documents. This project integrates Streamlit for a user-friendly interface.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Techniques Used](#techniques-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Manual Assistant is designed to assist users in extracting meaningful information from uploaded documents. Whether it's querying specific details from a manual or finding relevant content, the tool streamlines the process through document embedding and advanced question-answering capabilities.

## Features

- **Document Embedding:** Utilizes PyPDFLoader, text splitting, and Hugging Face Embeddings for efficient document representation.
- **Vector Search:** Implements Chroma from LangChain to perform similarity searches among embedded documents.
- **Question-Answering Chain:** Sets up a robust question-answering chain with HuggingFacePipeline and PromptTemplate for accurate responses.
- **Streamlit Integration:** Leverages Streamlit for a seamless user experience, allowing document upload and question-answering in a single interface.

## Techniques Used

### Document Embedding

The project employs PyPDFLoader for document loading, text splitting with CharacterTextSplitter, and Hugging Face Embeddings (sentence-transformers/all-mpnet-base-v2) for effective document embedding.

### Vector Search

Chroma from LangChain is utilized for vector search, enabling the tool to perform similarity searches among embedded documents efficiently.

### Question-Answering Chain

The question-answering chain is built using HuggingFacePipeline and PromptTemplate, providing a structured approach to handle user queries.

### Streamlit Integration

Streamlit is integrated into the project, offering a user-friendly interface for document upload and seamless interaction with the question-answering capabilities.

## Getting Started

To get started with the Manual Assistant, follow these steps:

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the main python file using `python manualassistant_tinyllama.py`
4. Run the Streamlit app using `streamlit run app.py`.

## Usage

1. Upload a PDF document using the provided interface.
2. Ask your questions in the input field.
3. Click "Get Answer" to receive accurate responses based on the uploaded document.

## License

This project is licensed under the [MIT License](LICENSE).
