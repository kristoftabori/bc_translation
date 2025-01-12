# SafeTranslate

SafeTranslate is a Streamlit-based application that translates text or CSV files while preserving placeholders in the input. It supports interactive text translation and batch translation from CSV files. The application ensures placeholders remain untouched during translation.

## Features
- **Interactive Translation:** Translate individual strings to multiple languages with placeholders preserved.
- **Batch Translation:** Upload a CSV file and translate multiple strings simultaneously, retaining placeholder integrity.
- **Highlighting:** Highlight cells in the output where placeholders match the original text.

## Requirements
- Python 3.12+
- Docker and Docker Compose (optional for containerized deployment)
- An OpenAI API key for using the language models.

## Installation

You can start the application either using Docker Compose or Poetry:

### Option 1: Using Docker Compose
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Build and start the application:
   ```bash
   docker compose up --build
   ```
3. Open the application in your browser at [http://localhost:8501](http://localhost:8501).

### Option 2: Using Poetry
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Install dependencies:
   ```bash
   poetry install
   ```
3. Run the application:
   ```bash
   poetry run streamlit run app/main.py
   ```
4. Open the URL displayed in the terminal to access the app.

## Usage

### 1. Setting Up the Environment
- Enter your OpenAI API key in the sidebar under the **API Key** field.
- Select your preferred OpenAI model (e.g., `gpt-4o`).

### 2. Interactive Translation
1. Navigate to the **Interactive** tab.
2. Enter the text you want to translate in the **Text Input** field.
3. Select one or more target languages from the dropdown.
4. Click **Translate**.
5. View the translated results below the input form.

### 3. Batch Translation
1. Navigate to the **CSV** tab.
2. Upload a CSV file containing the text to translate (one string per row, no header).
3. Select one or more target languages from the dropdown.
4. Click **Translate**.
5. View the translated results as a styled DataFrame in the app.

### 4. Highlighting
- In the batch translation results, cells where placeholders match **DO NOT** between the original and translated strings are highlighted with a semi-transparent color.

## Supported Placeholders
The application preserves placeholders defined in specific formats (e.g., `[placeholder]`). Ensure your placeholders conform to the expected format for accurate preservation.

## Contributing
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Create a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [Streamlit](https://streamlit.io/) for the app framework.
- [OpenAI](https://openai.com/) for the language models.

## Support
For questions or issues, please open an issue in the repository or contact the maintainer.

