# WebSummarizer

## Goal of the Project
WebSummarizer is a LangChain Web Summarization project that aims to optimize your web searches based on a specific goal. Instead of just presenting you with a list of search results, the program employs AI to summarize content from multiple web sources. It then ranks and selects the summary that best aligns with your predefined search goal, giving you more targeted and useful information.

## Requirements

### System Requirements

#### MacOS
-   M1 chip with 16GB RAM should suffice

#### Windows / Linux
-   CUDA-compatible GPU is optional but can enhance performance
-   At least 16GB RAM

Zephyr is a fine-tuned Llama2 model that is not particularly heavy, so even systems without a dedicated GPU should be able to run it without significant performance issues.


### Software Requirements
To utilize Google Search API through LangChain, the project requires you to have a Google account, an API key, and a Custom Search Engine ID (CSE ID).

#### To obtain a GOOGLE_API_KEY:
1. Visit the [Google Cloud Console](https://console.developers.google.com/).
2. Create a new project or select an existing one.
3. Navigate to `APIs & Services > Credentials`.
4. Click on `Create Credentials` and choose `API Key`.

#### To obtain a GOOGLE_CSE_ID:
1. Visit [Google Custom Search](https://cse.google.com/cse/create/new).
2. Create a new search engine and configure it according to your needs.
3. After creation, you will find your Custom Search Engine ID (CSE ID) on the setup page.

#### To install ollama:
1. Visit the [ollama GitHub repository](https://github.com/jmorganca/ollama).
2. Follow the installation instructions provided in the README file.


## Python Application Setup

### Installation
Install the required packages by running the following command:
```bash
pip install -r requirements.txt
```

### .env File Configuration
Create a `.env` file in the root directory of the project and add the following variables:

```bash
GOOGLE_CSE_ID=your_cse_id
GOOGLE_API_KEY=your_api_key
# Optionally, you can specify other environment variables like LLM_MODEL, MIN_SCORE, and PROMPT_TEMPLATE
```

## How to Run

Execute the main program with the following command:
```bash
python main.py
```


## Code Structure

-   `parse_string_to_json_v2(input_str: str) -> dict`: Converts a string containing key-value pairs formatted as 'key="value"', possibly embedded within other text, into a dictionary.
-   `fetch_and_summarize(n_results: int, search_query: str, search_objective: str) -> Tuple[dict, str]`: Conducts a web search based on the given query and number of results, summarizes relevant content using large language models, and ranks the best match according to the specificity of your search goal.
-   `main(n_results: int, search_query: str, search_objective: str) -> dict`: Accepts search parameters as arguments and calls `fetch_and_summarize()` to perform the goal-based web search and summarization. Returns a dictionary containing the results.


## Testing

### Unit Testing

To run the unit tests, navigate to the test directory and execute the test files using unittest. For example:

```bash
python -m unittest test_main.py
```

### Functional Testing

To run the functional tests, navigate to the test directory and execute the functional test file using unittest. For example:

```bash
python -m unittest test_functional.py
```
Note: the functional tests require a working internet connection and a valid GOOGLE_API_KEY and GOOGLE_CSE_ID in the .env file.
The results of the test might vary depending on the search results returned by google at the time of testing.

## Contact
For more details or issues, feel free to contact the maintainers.

By using this software, you are agreeing to the terms and conditions as defined by the license.