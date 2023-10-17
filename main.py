import os
import re

import dotenv
import jinja2
from fuzzywuzzy import fuzz
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import WebBaseLoader
from langchain.llms import Ollama
from langchain.utilities import GoogleSearchAPIWrapper

# Load environment variables
dotenv.load_dotenv()
j2env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
# Initialize Google credentials
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "zephyr")
MIN_SCORE = int(os.getenv("MIN_SCORE", "70"))
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", "prompts/zephyr.j2")

# Initialize Google Search API Wrapper
search = GoogleSearchAPIWrapper(search_engine="iltoga", google_cse_id=GOOGLE_CSE_ID, google_api_key=GOOGLE_API_KEY)


def parse_string_to_json_v2(input_str):
    """
    Parses a string containing key-value pairs formatted as 'key="value"',
    possibly embedded within other text, into a JSON-like dictionary.

    Args:
        input_str (str): The input string to parse.

    Returns:
        dict: The parsed JSON-like dictionary.
    """
    output_dict = {}

    # Use regular expression to find all occurrences of 'key="value"' in the string
    key_value_pairs = re.findall(r'(\w+)="([^"]*)"', input_str)

    for key, value in key_value_pairs:
        # Try converting value to integer, if possible
        try:
            value = int(value)
        except ValueError:
            pass

        # Add the key-value pair to the output dictionary
        output_dict[key] = value

    return output_dict


def fetch_and_summarize(num_results, query, search_goal):
    # Fetch search results
    search_results = search.results(query, num_results)

    # Split the query into individual terms
    query_terms = query.split()

    # Perform fuzzy search and filter out results with a score less than 80%
    best_matches = []
    highest_score = 0
    for result in search_results:
        snippet = result["snippet"]
        # Perform fuzzy matching for each term against the snippet
        term_scores = [fuzz.partial_ratio(term, snippet) for term in query_terms]
        # Compute the average score across all terms
        avg_score = sum(term_scores) / len(term_scores) if term_scores else 0
        if avg_score >= MIN_SCORE:
            result["score"] = avg_score  # Add the average score to the result dictionary
            best_matches.append(result)
            highest_score = max(highest_score, avg_score)

    llm = Ollama(model=LLM_MODEL, num_ctx=4096)

    # Evaluate the best match using Ollama if there are multiple best matches
    if len(best_matches) == 1:
        loader = WebBaseLoader(best_matches[0]["link"])
        doc = loader.load()
        # Summarize the document
        chain = load_summarize_chain(llm, chain_type="stuff")
        selection = chain.run(doc)

        return selection

    for match in best_matches:
        loader = WebBaseLoader(match["link"])
        doc = loader.load()
        # Summarize the document
        chain = load_summarize_chain(llm, chain_type="stuff")
        selection = chain.run(doc)

        # Update best_matches with the summary
        match["summary"] = selection
        match["ranking_score"] = match["score"] / highest_score

    # Prepare the data for the template by iterating over best_matches
    data = {
        "summaries": [(idx + 1, match["summary"]) for idx, match in enumerate(best_matches)],
        "search_goal": search_goal,
    }

    # Load and render the template
    template = j2env.get_template(PROMPT_TEMPLATE)
    prompt = template.render(data)

    # Submit the prompt to Ollama
    answers = llm.generate([prompt])
    first_answer = answers.generations[0][0].text
    json_object = parse_string_to_json_v2(first_answer)

    selected_summary_index = int(json_object["idx"]) - 1
    selection = best_matches[selected_summary_index]
    reason = json_object["description"]
    return selection, reason


def main(n_results: int, search_query: str, search_objective: str) -> dict:
    # Clamp n_results to a maximum of 10
    n_results = min(n_results, 10)

    # Fetch and summarize
    res, desc = fetch_and_summarize(n_results, search_query, search_objective)

    result_dict = {
        "link": res.get("link"),
        "snippet": res.get("snippet"),
        "score": res.get("score"),
        "ranking_score": res.get("ranking_score"),
        "ai_content": {"summary": res.get("summary"), "reason": desc},
    }

    return result_dict


if __name__ == "__main__":
    num_results = int(input("Number of results (max 10): "))
    query = input("Query: ")
    search_goal = input("Goal of your search: ")

    results = main(num_results, query, search_goal)

    print("Search Results:")
    print(f"Link: {results.get('link')}")
    print(f"Snippet: {results.get('snippet')}")
    print(f"Score: {results.get('score')}")
    print(f"Ranking Score: {results.get('ranking_score')}")
    print("AI Content:")
    print(f"Summary: {results['ai_content'].get('summary')}")
    print(f"Reason: {results['ai_content'].get('reason')}")
