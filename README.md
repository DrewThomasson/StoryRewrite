# StoryRewrite
Use Tiny LLM's to change the writing style of any given ebook

## Features

*   **File Conversion:** Accepts various document types (e.g., `.txt`, `.pdf`, `.epub`, `.docx`, `.html`) by converting them to plain text using Calibre.
*   **Sentence-by-Sentence Processing:** Splits the text into sentences and processes each one individually.
*   **LLM Rewriting:** Uses a specified Ollama model (default: `gemma3:1b`) to rewrite each sentence based on a customizable prompt.
*   **Web Interface:** Provides an easy-to-use Gradio interface for uploading files, setting the prompt, and viewing results.
*   **Progress Tracking:** Shows a progress bar during the rewriting process.

## Prerequisites

Before running the application, ensure you have the following installed and configured:

1.  **Python:** Python 3.7+
2.  **Calibre:** The full Calibre application must be installed. Crucially, the `ebook-convert` command-line tool included with Calibre needs to be accessible in your system's PATH environment variable.
    *   [Download Calibre](https://calibre-ebook.com/download)
    *   Verify by opening a terminal and typing `ebook-convert --version`.
3.  **Ollama:** Ollama must be installed and **running** in the background.
    *   [Download Ollama](https://ollama.com/)
4.  **Ollama Model:** The specific LLM model used by the script must be pulled. By default, this is `gemma3:1b`.
    *   Run: `ollama pull gemma3:1b`
5.  **Python Packages:** Listed in `requirements.txt`.
6.  **NLTK Data:** The NLTK 'punkt' tokenizer data is required for sentence splitting. The script attempts to download this automatically on first run if missing. If it fails, you can manually download it:
    ```bash
    python -m nltk.downloader punkt
    ```

## Installation

1.  **Clone the repository (or download the files):**
    ```bash
    # git clone <repository-url> # If you make this a git repo
    # cd <repository-directory>
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

1.  **Ensure Ollama is running** in the background.
2.  **Navigate to the directory** containing `app.py`.
3.  **Run the script:**
    ```bash
    python app.py
    ```
4.  The script will perform pre-launch checks for `ebook-convert` and Ollama connectivity. Check the console output for any errors or warnings.
5.  Open the URL provided in the terminal (usually `http://127.0.0.1:7860`) in your web browser.

## Usage

1.  **Upload File:** Click the upload box and select the document you want to process.
2.  **Customize Prompt (Optional):** Modify the default prompt in the textbox to instruct the LLM on *how* it should rewrite each sentence.
3.  **Rewrite:** Click the "Rewrite Sentences" button.
4.  **Wait:** Observe the progress bar. Processing time depends on the document length and LLM speed.
5.  **View Output:** The rewritten text will appear in the output text box. You can copy it using the copy button.

## Configuration

*   **Ollama Model:** You can change the `OLLAMA_MODEL` variable near the top of `app.py` to use a different Ollama model (ensure it's pulled first).
*   **ebook-convert Path:** If `ebook-convert` is not in your system PATH, you can set the full path in the `EBOOK_CONVERT_PATH` variable in `app.py`.
