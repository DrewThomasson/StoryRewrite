# -*- coding: utf-8 -*-
import gradio as gr
import ollama
import nltk
import subprocess
import os
import tempfile
import shutil
import sys
import traceback # For printing detailed errors

# --- Configuration ---
DEFAULT_PROMPT = "Make this given sentence written simple, respond with the modifed sentence and nothign else"
OLLAMA_MODEL = "gemma3:1b"
# Path to ebook-convert. Adjust if not in system PATH.
# Linux/macOS might just be 'ebook-convert'.
# Windows might be 'C:\\Program Files\\Calibre2\\ebook-convert.exe' (use double backslashes or raw string r'...')
EBOOK_CONVERT_PATH = 'ebook-convert' 

# --- NLTK Setup ---
try:
    # Check if 'punkt' is already available
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer found.")
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Attempting download...")
    try:
        nltk.download('punkt', quiet=True) # Download quietly
        nltk.data.find('tokenizers/punkt') # Verify download
        print("NLTK 'punkt' downloaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to download NLTK 'punkt' tokenizer: {e}")
        print("Please try running 'python -m nltk.downloader punkt' manually in your terminal.")
        # Consider exiting if sentence tokenization is absolutely critical
        # sys.exit(1) 
except Exception as e:
     print(f"An unexpected error occurred during NLTK check: {e}")
     # Optionally exit or proceed with caution
     # sys.exit(1)

# --- Core Processing Function ---
def rewrite_file_sentences(uploaded_file, rewrite_prompt, progress=gr.Progress(track_tqdm=True)):
    """
    Converts an uploaded file to TXT, splits into sentences, rewrites each
    sentence using Ollama, and returns the rewritten text.
    """
    if uploaded_file is None:
        return "Error: No file uploaded. Please upload a file."

    # Use default prompt if user clears the box
    if not rewrite_prompt:
        rewrite_prompt = DEFAULT_PROMPT
        print(f"Warning: Empty prompt provided. Using default: '{DEFAULT_PROMPT}'")


    temp_dir = tempfile.mkdtemp()
    rewritten_sentences = []
    original_file_path = None
    converted_txt_path = None
    print(f"Using temporary directory: {temp_dir}")

    try:
        # 1. Save uploaded file temporarily
        # Gradio's File component gives a temp file path directly in .name
        original_file_path = uploaded_file.name 
        original_filename = os.path.basename(original_file_path) # Get filename for naming output
        print(f"Processing uploaded file: {original_file_path}")

        # It's safer to copy it to our controlled temp dir in case Gradio cleans up early
        temp_original_path = os.path.join(temp_dir, original_filename)
        shutil.copy(original_file_path, temp_original_path)
        original_file_path = temp_original_path # Use the path inside our temp dir from now on
        print(f"Copied uploaded file to: {original_file_path}")


        # 2. Convert file to TXT using Calibre's ebook-convert
        base_name, _ = os.path.splitext(original_filename)
        converted_txt_path = os.path.join(temp_dir, f"{base_name}_converted.txt") # Make name slightly different
        
        progress(0, desc="Converting file to TXT...")
        print(f"Attempting conversion: '{original_file_path}' -> '{converted_txt_path}'")
        
        convert_command = [EBOOK_CONVERT_PATH, original_file_path, converted_txt_path]
        
        try:
            # Run conversion, capture output, don't raise exception on failure immediately
            result = subprocess.run(
                convert_command, 
                capture_output=True, 
                text=True, 
                check=False, # Manually check returncode
                encoding='utf-8', # Specify encoding for stdout/stderr
                errors='replace'  # Handle potential encoding errors in output
            )
            
            # Print Calibre output for debugging
            if result.stdout:
                print(f"ebook-convert stdout:\n---\n{result.stdout}\n---")
            if result.stderr:
                print(f"ebook-convert stderr:\n---\n{result.stderr}\n---")

            # Check for errors
            if result.returncode != 0:
                # Try to give a more helpful error based on stderr
                stderr_lower = result.stderr.lower()
                if "command not found" in stderr_lower or \
                   "'ebook-convert' is not recognized" in stderr_lower or \
                   ("no such file or directory" in stderr_lower and EBOOK_CONVERT_PATH in stderr_lower) or \
                   result.returncode == 127: # Common code for command not found on Linux
                     error_msg = (f"Error: '{EBOOK_CONVERT_PATH}' command failed.\n"
                                  f"Ensure Calibre is installed AND '{os.path.basename(EBOOK_CONVERT_PATH)}' "
                                  f"is in your system's PATH environment variable.\n"
                                  f"Stderr: {result.stderr.strip()}")
                elif "unsupported input format" in stderr_lower:
                     error_msg = (f"Error: Calibre could not convert the input file format.\n"
                                  f"Please provide a text-based file (e.g., TXT, DOCX, EPUB, HTML, PDF).\n"
                                  f"Stderr: {result.stderr.strip()}")
                else:
                    error_msg = f"Error during Calibre conversion (Code {result.returncode}). Check console logs.\nStderr: {result.stderr.strip()}"
                raise RuntimeError(error_msg)

            # Verify the output file was actually created
            if not os.path.exists(converted_txt_path) or os.path.getsize(converted_txt_path) == 0:
                 # Sometimes conversion finishes with code 0 but produces no output for certain errors
                 if not result.stderr and not result.stdout:
                    print("Warning: ebook-convert finished with code 0 but produced no output and no logs. Input might be empty or corrupted.")
                 elif not os.path.exists(converted_txt_path):
                     raise RuntimeError(f"Error: Calibre conversion seemed to succeed (Code 0), but the output file '{converted_txt_path}' was not created. Check Calibre logs/output.")
                 else: # File exists but is empty
                     print(f"Warning: Calibre conversion resulted in an empty output file: '{converted_txt_path}'. The original file might have no text content.")
                     # Proceed, might result in "no sentences found" later

            print(f"File converted successfully to: {converted_txt_path}")

        except FileNotFoundError:
             # This catches if EBOOK_CONVERT_PATH itself doesn't exist
             raise RuntimeError(f"Error: '{EBOOK_CONVERT_PATH}' command not found. Please ensure Calibre is installed and '{os.path.basename(EBOOK_CONVERT_PATH)}' is in your system's PATH.")
        except Exception as e:
             # Catch other potential subprocess errors or RuntimeErrors raised above
             # Avoid printing the full exception 'e' directly to the user if it contains sensitive paths
             print(f"Conversion failed: {e}") # Log detailed error
             raise RuntimeError(f"An error occurred during file conversion. Check console logs for details. Message: {e}")


        # 3. Read the converted TXT file
        progress(0.1, desc="Reading converted text...")
        text = ""
        if os.path.exists(converted_txt_path):
            try:
                with open(converted_txt_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
            except Exception as e:
                raise RuntimeError(f"Error reading converted TXT file '{converted_txt_path}': {e}")

        if not text.strip():
            print("Warning: Converted file is empty or contains only whitespace.")
            # Return message instead of proceeding? Or let it find 0 sentences?
            # return "Warning: The converted file appears to be empty." 

        # 4. Split text into sentences
        progress(0.2, desc="Splitting text into sentences...")
        sentences = []
        if text.strip(): # Only tokenize if there's non-whitespace text
            try:
                sentences = nltk.sent_tokenize(text)
            except Exception as e:
                 raise RuntimeError(f"Error during sentence tokenization with NLTK: {e}")
             
        if not sentences:
            print("Warning: No sentences found in the text after conversion.")
            return "Warning: No sentences could be extracted from the provided file after conversion."
            
        print(f"Found {len(sentences)} sentences.")

        # 5. Initialize Ollama Client
        progress(0.25, desc="Connecting to Ollama...")
        try:
            client = ollama.Client()
            # Quick check if the client can connect and list models
            client.list() 
            print(f"Successfully connected to Ollama. Using model: {OLLAMA_MODEL}")
        except Exception as e:
            raise RuntimeError(f"Error connecting to Ollama or listing models. Is the Ollama server running? Is '{OLLAMA_MODEL}' pulled? Error: {e}")


        # 6. Process each sentence with Ollama
        total_sentences = len(sentences)
        progress(0.3, desc=f"Rewriting sentences (0/{total_sentences})...")

        for i, sentence in enumerate(sentences):
            current_progress = 0.3 + (0.7 * ((i + 1) / total_sentences))
            progress(current_progress, desc=f"Rewriting sentence {i+1}/{total_sentences}")
            
            sentence = sentence.strip()
            if not sentence: # Skip empty strings that might result from splitting unusual text
                continue

            # Construct the prompt for Ollama
            full_prompt = f"{rewrite_prompt}\n\nSentence: \"{sentence}\""
            
            try:
                # Use chat endpoint for better instruction following
                response = client.chat(
                    model=OLLAMA_MODEL,
                    messages=[{'role': 'user', 'content': full_prompt}],
                    options={'temperature': 0.5} # Adjust temperature as needed (0=deterministic, 1=creative)
                )
                
                rewritten = response['message']['content'].strip()
                
                # Basic cleaning: Remove potential model prefixes/artifacts
                rewritten = rewritten.removeprefix("Modified sentence:").strip()
                rewritten = rewritten.removeprefix("Rewritten sentence:").strip()
                rewritten = rewritten.removeprefix("Simple sentence:").strip()
                # Remove potential surrounding quotes added by the model
                if rewritten.startswith('"') and rewritten.endswith('"'):
                   rewritten = rewritten[1:-1]
                if rewritten.startswith("'") and rewritten.endswith("'"):
                   rewritten = rewritten[1:-1]


                rewritten_sentences.append(rewritten)
                # print(f"Original [{i+1}]: {sentence}") # Uncomment for verbose debugging
                # print(f"Rewritten[{i+1}]: {rewritten}") # Uncomment for verbose debugging

            except ollama.ResponseError as e:
                print(f"Ollama API Error for sentence {i+1}: Status Code: {e.status_code}, Error: {e.error}")
                # Append original sentence with an error marker
                rewritten_sentences.append(f"[OLLAMA ERROR: {e.error}] {sentence}") 
            except Exception as e:
                print(f"Generic error during Ollama call for sentence {i+1}: {type(e).__name__}: {e}")
                traceback.print_exc() # Print full traceback for debugging
                rewritten_sentences.append(f"[PROCESSING ERROR] {sentence}")


        # 7. Combine rewritten sentences
        final_text = "\n".join(rewritten_sentences) # Join with newlines for better readability
        progress(1.0, desc="Finished!")
        print("Processing complete.")
        return final_text

    except RuntimeError as e:
        # Catch errors raised explicitly within the try block (e.g., conversion, NLTK)
        print(f"Caught RuntimeError: {e}")
        # Return the specific error message to Gradio
        return f"Error: {e}" 
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Caught unexpected exception in 'rewrite_file_sentences':")
        traceback.print_exc() # Print full traceback to console
        return f"An unexpected error occurred during processing. Please check the console logs. Error type: {type(e).__name__}"
    finally:
        # 8. Cleanup temporary directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Failed to delete temporary directory '{temp_dir}': {e}")


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo: # Added a theme for better visuals
    gr.Markdown(
        """
        # Simple Sentence Rewriter using Ollama (`gemma3:1b`)
        
        Upload almost any document (e.g., `.txt`, `.pdf`, `.epub`, `.docx`, `.html`). 
        The app will use **Calibre's `ebook-convert`** to convert it to plain text first. 
        
        Then, it splits the text into sentences and rewrites each one using a local **Ollama LLM** 
        (model: `""" + OLLAMA_MODEL + """`) based on the prompt you provide below.
        
        **Important Prerequisites:**
        1.  **Calibre:** Must be installed, and the `ebook-convert` command needs to be accessible in your system's PATH. ([Download Calibre](https://calibre-ebook.com/download))
        2.  **Ollama:** Must be installed and **running** in the background. ([Download Ollama](https://ollama.com/))
        3.  **Ollama Model:** The specific model (`""" + OLLAMA_MODEL + """`) must be pulled via `ollama pull """ + OLLAMA_MODEL + """`.
        4.  **Python Packages:** Run `pip install gradio ollama nltk`
        5.  **NLTK Data:** The first run might download the 'punkt' tokenizer data from NLTK.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(
                label="Upload File",
                file_count="single",
                # file_types=['text', '.pdf', '.epub', '.mobi', '.docx', '.doc', '.html'] # You can suggest types, but ebook-convert handles many
            )
            prompt_box = gr.Textbox(
                label="Rewrite Prompt per Sentence",
                info="Instruct the LLM on how to rewrite EACH sentence. It gets one sentence at a time.",
                value=DEFAULT_PROMPT,
                lines=3
            )
            rewrite_button = gr.Button("✍️ Rewrite Sentences", variant="primary") # Added emoji and variant

        with gr.Column(scale=2):
            output_textbox = gr.Textbox(
                label="Rewritten Text Output",
                lines=25, # Increased lines
                interactive=False, # Output is read-only
                show_copy_button=True
            )

    # Connect button click to the processing function
    rewrite_button.click(
        fn=rewrite_file_sentences,
        inputs=[input_file, prompt_box],
        outputs=output_textbox
        # api_name="rewrite" # Optional: for programmatic access
    )

# --- Launch the App ---
if __name__ == "__main__":
    # --- Pre-launch Check: ebook-convert ---
    print("-" * 60)
    print(f"Checking for Calibre's '{EBOOK_CONVERT_PATH}' command...")
    try:
        # Use '--version' as a quick, non-destructive check
        result = subprocess.run([EBOOK_CONVERT_PATH, '--version'], capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
        print(f"✅ Found ebook-convert: {result.stdout.strip().splitlines()[0]}") # Print just the first line of version info
    except FileNotFoundError:
        print(f"❌ ERROR: Command '{EBOOK_CONVERT_PATH}' not found.")
        print("   Please ensure Calibre is installed and its directory is added to your system's PATH.")
        print("   App will likely fail during file conversion.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: Command '{EBOOK_CONVERT_PATH}' executed but returned an error (Code {e.returncode}).")
        print(f"   Stderr: {e.stderr.strip()}")
        print("   This might indicate an issue with the Calibre installation, but the app will attempt to proceed.")
    except Exception as e:
        print(f"❓ An unexpected error occurred while checking for '{EBOOK_CONVERT_PATH}': {e}")
        print("   The app will attempt to proceed, but conversion might fail.")
    print("-" * 60)

    # --- Pre-launch Check: Ollama ---
    print("Checking Ollama connection...")
    try:
        client = ollama.Client()
        models = client.list()
        print(f"✅ Successfully connected to Ollama.")
        available_models = [m['name'] for m in models['models']]
        if OLLAMA_MODEL in available_models:
             print(f"   Model '{OLLAMA_MODEL}' is available.")
        else:
             print(f"⚠️ WARNING: Model '{OLLAMA_MODEL}' not found in Ollama models: {available_models}")
             print(f"   Please ensure you have run 'ollama pull {OLLAMA_MODEL}'")
             print("   The app will likely fail when trying to rewrite text.")

    except Exception as e:
        print(f"❌ ERROR: Could not connect to Ollama or list models.")
        print(f"   Ensure the Ollama application/server is running.")
        print(f"   Details: {e}")
    print("-" * 60)


    print("Launching Gradio interface... Access it at the URL provided below.")
    # Share=True creates a public link (useful for testing/sharing temporarily) - remove if not needed
    # Set debug=True for more detailed Gradio logs during development
    demo.launch() 