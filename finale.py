import feedparser
import json
import ast
from openai import OpenAI
import trafilatura
from bs4 import BeautifulSoup
import requests
from IPython.display import display, Markdown, Latex
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
import os
import time
import re
import iii

DEBUG = False

lines = 14
splitcount = 3  # recommend 3 or below, otherwise article translate may stop without a warning

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

# Environment Variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "L3 Research Agent"
os.environ["LANGCHAIN_API_KEY"] = "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"


# web search part
def nvidia_llama_completion(prompt):
    global prompt_count
    prompt_count = prompt_count + 1
    completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=8192,
        stream=True
    )
    # Retrieve and concatenate the response from the streamed results
    response_json = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_json += chunk.choices[0].delta.content
    return response_json

# Web Search Tool
class CustomSearchAPIWrapper:
    def __init__(self, max_results=25):
        self.max_results = max_results

    def search(self, query):
        encoded_query = requests.utils.quote(query)
        url = f"https://www.google.com/search?q={encoded_query}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Send the request
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')
        snippets = []
        
        for g in soup.find_all('div', class_='tF2Cxc')[:self.max_results]:
            snippet = ''
            if g.find('span', class_='aCOpRe'):
                snippet = g.find('span', class_='aCOpRe').text
            elif g.find('div', class_='IsZvec'):
                snippet = g.find('div', class_='IsZvec').text
            elif g.find('div', class_='VwiC3b'):
                snippet = g.find('div', class_='VwiC3b').text
            elif g.find('div', class_='s3v9rd'):
                snippet = g.find('div', class_='s3v9rd').text
            
            snippets.append(snippet)
        
        return snippets

class CustomSearchRun:
    def __init__(self, api_wrapper):
        self.api_wrapper = api_wrapper

    def invoke(self, query):
        return self.api_wrapper.search(query)

wrapper = CustomSearchAPIWrapper(max_results=25)
web_search_tool = CustomSearchRun(api_wrapper=wrapper)

# Custom chain class
class NvidiaLLMChain:
    def __init__(self, prompt, llm, output_parser):
        self.prompt = prompt
        self.llm = llm
        self.output_parser = output_parser

    def run(self, inputs):
        formatted_prompt = self.prompt.format(**inputs) 
        response = self.llm(formatted_prompt)
        parsed_response = self.output_parser.parse(response)
        return self.output_parser.parse(response)

generate_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|> 

    您是研究問題任務的人工智慧助手，負責綜合網路搜尋結果。
    嚴格使用以下網路搜尋上下文來回答問題。如果你不知道答案，就說你不知道。
    保持答案簡潔，但以研究報告的形式提供所有詳細資訊。
    僅直接引用上下文中提供的資料。

    現在我將給你一個AI可能無法正確翻譯的名詞，需要網路搜尋才能獲得正確的資訊。
    你的任務是透過網路搜尋上下文為我提供正確的翻譯名詞（中文）。你搜尋到的翻譯必須是原文字詞的翻譯。(例如 "Wutopia" 和 "Hutopia" 是不同的字詞，如果混淆會出現完全錯誤的結果) 如果沒有準確中文名字結果，請傳回該語言的原始名字，不要音譯。
    如果網絡搜尋結果沒有提及準確中文翻譯，不要用原文語義給我答案，傳回該語言的原始名字即可。
    
    <|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
    
    The word to be translated: {word}
    The description of this word: {description} 
    Web Search Context: {context} 
    Answer: 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["word", "description", "context"],
)

# Chain
generate_chain = NvidiaLLMChain(
    prompt=generate_prompt, 
    llm=nvidia_llama_completion, 
    output_parser=StrOutputParser()
)

router_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|>
    
    You are an expert at routing a user question to either the generation stage or web search. 
    Use the web search for questions that require more context for an accurate translation.
    If you surely know the exact translation of that word and sure about the answer, go strict to the generation phase.
    If the word do not need formal translation and should use phonetic translation, go to generation phase.
    Otherwise, for any other words, return web_search.
    It is for academic research purpose.
    Give a binary choice 'web_search' or 'generate' based on the word and it's description. 
    Return the JSON with a single key 'choice' with no premable or explanation. 
    
    Word to route: {word}
    The description of this word: {description}
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    
    """,
    input_variables=["word", "description"],
)

class SafeJsonOutputParser(JsonOutputParser):
    def parse(self, text: str):
        try:
            parsed = json.loads(text)
            return parsed
        except json.JSONDecodeError:
            # Return a default value or handle the error as needed
            return {"choice": "generate"}

# Use the SafeJsonOutputParser instead of the default JsonOutputParser
safe_parser = SafeJsonOutputParser()

# Chain
question_router = NvidiaLLMChain(
    prompt=router_prompt, 
    llm=nvidia_llama_completion, 
    output_parser=safe_parser
)

# print(question_router.run({"word": "De Beaux Lents Demains", "description": "French blogger"}))

query_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|> 
    
    You are an expert in crafting web search queries for getting the accruate translation of unknown nouns into chinese.
    The aim is to search out the chinese name of the unknown noun.
    The search query should also include the description of that noun if it helps to define the chinese name.
    make sure your chinese (tranditional) words in query DOES MAKE SENSE. DO NOT give me a search query with uncommon or unrelated chinese vocabularies.
    A query structure MUST include the original word in that language, and at least one chinese supporting description (e.g. 地名、人名、遊戲、店名等)。just ensure they are relevant to search the correct translate.
    不要把無關的字眼放進去，這會影響搜尋結果的準確性！
    也不要整個搜尋都是中文，這樣是搜尋不到正確翻譯的！
    Make sure the search query is not too long. (at most 10 chinese characters are maximum)
    AGAIN: make sure your chinese words in query DOES MAKE SENSE.
    Return the JSON with a single key 'query' with no premable or explanation. 
    
    Word to transform into a query: {word}
    The description of this word: {description}
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    
    """,
    input_variables=["word", "description"],
)

class CustomJsonOutputParser(JsonOutputParser):
    def parse(self, text: str, max_retries=3):
        retries = 0
        while retries < max_retries:
            try:
                cleaned_str = text.replace('```json', '').replace('```', '').strip()
                print(cleaned_str)
                cleaned_str = cleaned_str.replace('"', '')
                cleaned_str = cleaned_str.replace("'", '')
                print(cleaned_str)
                cleaned_str = re.sub(r'\s*{\s*', '{', cleaned_str)
                cleaned_str = re.sub(r'\s*}\s*', '}', cleaned_str)
                cleaned_str = re.sub(r'\s*:\s*', ':', cleaned_str)
                print(cleaned_str)
                cleaned_str = cleaned_str.replace('{','{"').replace('}','"}').replace(':','":"')
                cleaned_str = re.sub(r'\\(?!u)', '', cleaned_str)
                print(cleaned_str)
                #cleaned_str = cleaned_str.replace('{', '{"').replace('}', '"}').replace(':', '":"')
                modified_string = json.loads(cleaned_str)
                return modified_string
            except json.JSONDecodeError as e:
                retries += 1
                print(f"JSON decode error: {e}. Retrying {retries}/{max_retries}...")
        
        return {"query": ""}

# Create the chain with the custom output parser
query_chain = NvidiaLLMChain(
    prompt=query_prompt,
    llm=nvidia_llama_completion,
    output_parser=CustomJsonOutputParser(max_retries=3)
)

# print(query_chain.run({"word": "Les Zed", "description": "French blogger or influencer"}))

# Graph State
class GraphState(TypedDict):
    word : str
    description: str
    generation : str
    search_query : str
    context : str

# Node - Generate
def generate(state):
    print("Step: Generating Final Response")
    word = state["word"]
    description = state['description']
    context = state["context"]

    # Answer Generation
    generation = generate_chain.run({"context": context, "word": word, "description": description})
    return {"generation": generation}

# Node - Query Transformation
def transform_query(state):    
    print("Step: Optimizing Query for Web Search")
    word = state['word']
    description = state['description']
    gen_query = query_chain.run({"word": word, "description": description})
    print(gen_query)
    search_query = gen_query["query"]
    return {"search_query": search_query}

# Node - Web Search
def web_search(state):
    search_query = state['search_query']
    print(f'Step: Searching the Web for: "{search_query}"')
    
    # Web search tool call
    search_result = web_search_tool.invoke(search_query)
    return {"context": search_result}

# Conditional Edge, Routing
def route_question(state):
    print("Step: Routing Query")
    word = state['word']
    description = state['description']
    output = question_router.run({"word": word, "description": description})
    if isinstance(output, dict):
        if output['choice'] == "web_search":
            print("Step: Routing Query to Web Search")
            return "websearch"
        elif output['choice'] == 'generate':
            print("Step: Routing Query to Generation")
            return "generate"
    else:
        print("Step: Routing Query to Generation")
        return "generate"
# Build the nodes
workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)

# Build the edges
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "websearch")
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)

# Compile the workflow
local_agent = workflow.compile()
def run_agent(query, original_word):
    output = local_agent.invoke({"word": original_word, "description": query})
    display(output["generation"])
    search_result = str(output["generation"])
    translated_json = dict_chain.run({"original_word": str(original_word), "search_result": search_result})
    return translated_json

#run_agent("Vimara Village is a subarea located in Ardravi Valley, Dharma Forest, Sumeru", "Vimara Village")
dict_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|> 

    Original word:
    {original_word}
    
    The search result for the word:
    {search_result}
    
    My aim is to get the correct translation of the original word, from the crawled search results.
    Summarize the search result. 
    As long as search result NOT 100 PERCENT SURE IT IS CORRECT, return me the original word.
    Otherwise, only return the MUST correct chinese translation of that noun i needed.
    return me a JSON object that store the original word and search result word in key-value pairs. 
    REMEMBER: If there isn't a chinese name found, return me the original name, do not phonetic translate or translate with the original word's english meaning.
    Do not leave the key-value pair blank no matter what.
    if there are more than one translation, only return me one.
    REMEMBER: the JSON returned has only ONE key-value pair. no need other keys to label.
    Return the JSON with ONE key-value pair with no premable or explanation. 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    
    """,
    input_variables=["original_word","search_result"],
)

dict_chain = NvidiaLLMChain(
    prompt=dict_prompt, 
    llm=nvidia_llama_completion, 
    output_parser=JsonOutputParser()
)

# List of RSS feed URLs
rss_urls = [
    #'https://www.france24.com/en/rss'
    'https://www.scmp.com/rss/322899/feed'
    # Add more RSS feed URLs here
]

# Function to fetch and parse RSS feeds
def fetch_news(rss_urls):
    news_items = []
    seen_titles = set()

    for url in rss_urls:
        feed = feedparser.parse(url)

        for entry in feed.entries:
            if entry.title not in seen_titles:
                news_item = {
                    'title': entry.title,
                    'link': entry.link,
                    'summary': entry.summary
                }
                news_items.append(news_item)
                seen_titles.add(entry.title)

    return news_items

# Fetch news from RSS feeds
news_items = fetch_news(rss_urls)

def count_newlines_exceeds_limit(text: str, limit: int = 5) -> bool:
    # Count the number of new line characters in the text
    newline_count = text.count('\n')
    # Check if the count exceeds the specified limit
    return newline_count > limit

def extract_headers_and_content(article):
    # Parse the article with BeautifulSoup
    soup = BeautifulSoup(article, 'html.parser')
    
    # Initialize a list to store the headers and associated content
    content_dict = {}
    
    # Find all headers
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    # Iterate through headers to get their content
    for i, header in enumerate(headers):
        header_tag = header.name
        header_text = header.get_text()
        
        # Find the next header to determine the content range
        if i + 1 < len(headers):
            next_header = headers[i + 1]
            content = ''
            for sibling in header.next_siblings:
                if sibling == next_header:
                    break
                if sibling.name:
                    content += str(sibling)
        else:
            # For the last header, take all remaining content
            content = ''
            for sibling in header.next_siblings:
                if sibling.name:
                    content += str(sibling)
        
        content_dict[(header_tag, header_text)] = content
    
    return content_dict

def extract_headers_and_content_from_list(html_list):
    # Initialize a dictionary to store the headers and associated content
    content_dict = {}
    
    # Convert the list to a single string of HTML
    combined_html = ''.join(html_list)
    
    # Parse the combined HTML with BeautifulSoup
    soup = BeautifulSoup(combined_html, 'html.parser')
    
    # Find all headers
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    # Iterate through headers to get their content
    for i, header in enumerate(headers):
        header_tag = header.name
        header_text = header.get_text()
        
        # Find the next header to determine the content range
        if i + 1 < len(headers):
            next_header = headers[i + 1]
            content = ''
            for sibling in header.next_siblings:
                if sibling == next_header:
                    break
                if sibling.name:
                    content += str(sibling)
        else:
            # For the last header, take all remaining content
            content = ''
            for sibling in header.next_siblings:
                if sibling.name:
                    content += str(sibling)
        
        content_dict[(header_tag, header_text)] = content
    
    return content_dict

def format_html_with_newlines(html):
    """Format the HTML string with new lines after each closing tag."""
    soup = BeautifulSoup(html, 'html.parser')
    pretty_html = soup.prettify()
    return pretty_html

def modify_content(dict1, dict2, toc=None):
    merged_html = ''
    
    # Extract headers from both dictionaries
    headers1 = list(dict1.keys())
    headers2 = list(dict2.keys())
    
    i, j = 0, 0
    while i < len(headers1) and j < len(headers2):
        header1, header2 = headers1[i], headers2[j]
        
        # Check if headers are the same
        if header1[0] == header2[0]:
            # Append the header in HTML form
            merged_html += f'<{header1[0]}>{header1[1]}</{header1[0]}>\n'
            
            # Append the formatted content of the header from list 1 and then from list 2
            merged_html += format_html_with_newlines(dict1[header1]) + '\n'
            if header1[0] == "h1" and toc is not None:
                merged_html += format_html_with_newlines(toc) + '\n'
            merged_html += format_html_with_newlines(dict2[header2]) + '\n'
            
            i += 1
            j += 1
        elif header1 < header2:
            # Append the header and formatted content from list 1
            merged_html += f'<{header1[0]}>{header1[1]}</{header1[0]}>\n'
            merged_html += format_html_with_newlines(dict1[header1]) + '\n'
            i += 1
        else:
            # Append the header and formatted content from list 2
            merged_html += f'<{header2[0]}>{header2[1]}</{header2[0]}>\n'
            merged_html += format_html_with_newlines(dict2[header2]) + '\n'
            j += 1
    
    # Append remaining headers from list 1
    while i < len(headers1):
        header1 = headers1[i]
        merged_html += f'<{header1[0]}>{header1[1]}</{header1[0]}>\n'
        merged_html += format_html_with_newlines(dict1[header1]) + '\n'
        i += 1
    
    # Append remaining headers from list 2
    while j < len(headers2):
        header2 = headers2[j]
        merged_html += f'<{header2[0]}>{header2[1]}</{header2[0]}>\n'
        merged_html += format_html_with_newlines(dict2[header2]) + '\n'
        j += 1
    
    return merged_html

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()
    return content

def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines("""<html><head><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
        <style>*{font-size:20px}</style></head><body><div class="container">""")
        file.writelines(content)
        file.writelines("</div></body></html>")

def extract_alt_attributes(html_content):
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all img tags and extract the alt attributes
    alt_attributes = [img['alt'] for img in soup.find_all('img') if 'alt' in img.attrs]
    
    return alt_attributes

def add_max_width_to_images(html_content, max_width):
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all img tags and add a max-width style attribute
    for img in soup.find_all('img'):
        if 'style' in img.attrs:
            img['style'] += f'; max-width: {max_width};'
        else:
            img['style'] = f'max-width: {max_width};'
    
    # Return the modified HTML as a string
    return str(soup)

def clean_title(title, extension):
    # Define invalid characters for filenames
    invalid_chars = r'[<>:"/\\|?*]'
    
    # Replace invalid characters with an underscore
    clean_title = re.sub(invalid_chars, '_', title)
    
    # Remove leading and trailing whitespace
    clean_title = clean_title.strip()
    
    # Check for reserved names (specific to Windows)
    reserved_names = {"CON", "PRN", "AUX", "NUL", "COM1", "LPT1"}
    if clean_title.upper() in reserved_names:
        clean_title = f"{clean_title}_file"
    
    # Ensure the title is not empty
    if not clean_title:
        clean_title = "default_filename"
    
    # Append the file extension, ensuring it starts with a dot
    if not extension.startswith('.'):
        extension = f".{extension}"
    
    # Construct the full file name
    file_name = f"{clean_title}{extension}"
    
    return file_name

def update_image_sources(html_content):
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all img tags and update the src attribute
    for img in soup.find_all('img'):
        if 'src' in img.attrs:
            original_src = img['src']
            new_src = os.path.join('images', os.path.basename(original_src))
            img['src'] = new_src
    
    # Return the modified HTML as a string
    return str(soup)
# the core of generation logic
def parse_full_text(url, title, lines = 14, splitcount = 3):
    full_article = ""
    big_arr = []

    # grab all the main content with trafilatura
    print(url)
    downloaded = trafilatura.fetch_url(url)
    website_text = trafilatura.extract(downloaded)

    if website_text is None:
        print("Failed to scrap, skipped this URL.")
        return

    if not count_newlines_exceeds_limit(website_text):
        print(website_text)
        print("Bad formatting, skipped this URL.")
        return

    if website_text:
        # Split the article into segments
        segments = split_article_into_segments(website_text, lines_per_segment=lines)

        # Process each segment
        article_array, segmented_json = process_segments(segments, title)

        print(article_array)

        if article_array:  # Ensure article_array is not empty or None
            big_arr = split_on_every_nth_h2(article_array, n=splitcount)

    for arr in big_arr:
        full_article += consideration_test(arr, title, segmented_json)
        full_article += "\n"

    # remove unrelated role content and promotional parts
    finale = article_checker(full_article)
    links = []
    txt_filename = f'{title}.txt'

    # generate a structure with img and headers only
    heads = extract_headers(finale)
    ke = taoke(heads)
    alt_texts = extract_alt_attributes(ke)
    max_height = '500px'
    ke = add_max_width_to_images(ke, max_height)
    ke = update_image_sources(ke)
    for alt in alt_texts:
        iii.download_unsplash_images(alt) 
    file_path = clean_title(title, 'txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(ke)
    print(ke)
    with open(file_path, 'r', encoding='utf-8') as file:
        print(file.read())

    # programming to put all back into one article
    content = read_file(file_path)
    dict1 = extract_headers_and_content_from_list(content)
    dict2 = extract_headers_and_content(finale)
    if len(extract_headers(finale)) > 5:
        toc = generate_toc(finale)
    else:
        toc = None
    modified_content = modify_content(dict1, dict2, toc)
    file_path = clean_title(title, 'html')
    write_file(file_path, modified_content)

    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Finished processing, prompt count: " + str(prompt_count))

def split_article_into_segments(article, lines_per_segment=13):
    lines = article.split('\n')
    segments = [lines[i:i + lines_per_segment] for i in range(0, len(lines), lines_per_segment)]
    print(segments)
    return segments

# Define the function to check if p_segment is defined and not empty
def check_variable(var_name, context):
    if var_name in context and context[var_name]:
        variable = context[var_name]
        if variable is None or (isinstance(variable, str) and variable.strip() == "") or (isinstance(variable, (list, tuple, set, dict)) and len(variable) == 0):
            return "This segment is the first segment of the article. label h1 for first line."
        else:
            return "This segment is not the first segment of the article. do not label any h1. previous segment: " + str(variable)
    else:
        return "This segment is the first segment of the article. label h1 for first line."

def check_dict(var_name, seg_dict):
    if var_name in seg_dict and seg_dict[var_name]:
        variable = seg_dict[var_name]
        if variable is None or (isinstance(variable, str) and variable.strip() == "") or (isinstance(variable, (list, tuple, set, dict)) and len(variable) == 0):
            return ""
        else:
            return "There are some words already translated, DO NOT return these words again: " + str(variable)
    else:
        return ""

def parse(text):
    modified_string = text.replace("'", '')
    modified_string = modified_string.replace('"', '')
    modified_string = modified_string.replace("\\'", "")  # Replace escaped single quotes
    modified_string = modified_string.replace('\\"', '')  # Replace escaped double quotes
    modified_string = modified_string.replace("\\\\", "")
    modified_string = modified_string.replace(r'{},', '')
    modified_string = modified_string.replace('\n', '')
    modified_string = modified_string.replace('6:', '6":"').replace('5:', '5":"').replace('4:', '4":"').replace('3:', '3":"').replace('2:', '2":"').replace('1:', '1":"').replace('p:', 'p":"').replace('g:', 'g":"').replace('i:', 'i":"')
    modified_string = modified_string.replace('{', '{"').replace('}', '"}')
    modified_string = json.loads(modified_string)
    return modified_string

def is_dict_or_json(input_data):
    # Check if the input is a dictionary
    if isinstance(input_data, dict):
        return True
    
    # Try to parse it as JSON
    try:
        parsed_json = json.loads(input_data)
        if isinstance(parsed_json, dict):
            return True
    except (ValueError, TypeError):
        return False
    
    return False

# Define the function to process segments
def process_segments(segments, title, max_retries=3):
    global prompt_count
    article_array = []
    segmented_json = {}
    context = {}
    seg_dict = {}

    for segment in segments:
        p_prompt = check_variable('p_segment', context)
        p_dict = check_dict('translated', seg_dict)

        prompt = f"""
        I divided a big article into small segments for you to process. I will give you the previous segment you have processed (if any), and another segment for you to process now.
        The segment is an array that contains a few lines in a big article, without defining any tags.
        Return me an array object with header tags (h1, h2, h3, etc.), p, li (HTML list item if the first character is a bullet point "-") and img (if that sentence looks more like an image caption, not paragraph) tags labeled in the original article order for the new segment only, using AI.
        Each line should be one JSON object storing a key-value pair.
        Turn all active voice sentences into passive voice. Remember to change the subjects in the sentence.
        Do not reply with any other things other than the array so I can do further processing.
        If the content is unrelated to the main title: {title}, e.g., more related posts or just for article formatting, ignore the line. (skip that tag line)

        {p_prompt}

        Now process this following segment:
        {segment}

        Again: do not reply with any other things except a proper array format reply.
        No premable, no explanation.
        Do not merge multiple lines.
        """

        print(prompt)

        retries = 0
        success = False
        response_json = ""

        while retries < max_retries and not success:
            try:
                prompt_count = prompt_count + 1
                completion = client.chat.completions.create(
                    model="meta/llama-3.1-405b-instruct",
                    messages=[{"role": "user", "content": prompt.strip()}],
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=8192,
                    stream=True
                )

                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        response_json += chunk.choices[0].delta.content
                        print(chunk.choices[0].delta.content, end="")
                response_array = parse(response_json)
                print(response_array)
                success = True
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                retries += 1
                time.sleep(2)  # Optional: Add a delay before retrying
                response_json = ""

        if not success:
            print(f"Failed to process segment after {max_retries} retries.")
            continue

        refined = refine_response(response_array, p_dict)
        print(refined)
        if refined is not None and is_dict_or_json(refined):
            segmented_dict = {}
            for word, prompt in refined.items():
                dictionary = run_agent(prompt, word)
                print(dictionary)
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                segmented_dict.update(dictionary)
                if len(segmented_dict) != 0:
                    print(segmented_dict)
        context['p_segment'] = response_array
        seg_dict['translated'] = refined
        segmented_json.update(segmented_dict)
        if len(segmented_json) != 0:
            print(segmented_json)
        article_array.extend(response_array)

    return article_array, segmented_json

# Define the refinement prompt
def refine_response(response_json, translated):
    global prompt_count
    prompt_count = prompt_count + 1
    prompt = f"""
    You have processed the segment and returned a JSON array. Now, I need you to refine this output by ensuring it follows all the instructions properly.

    Here is the initial JSON array:
    {response_json}

    - The JSON array above contains the structure of a HTML article file.
    - Now I need to turn it into a chinese article. But before that, there are some nouns require web search to get accurate translation.
    - Your job is scan through the JSON array, understanding the full context and extract important nouns that cannot be translated directly.
    - All non-well-known nouns include personal nouns (e.g. blogger's son's name, a French blogger website name, etc) should NOT be extracted. DO NOT return me these nouns for extra web search.
    - For other human names, if they are well known and can be found in web, return me the full name to be searched.
    - but for abbreviations (e.g. "Cheung" instead of "Cheung ka-long"), DO NOT return this noun as it cannot be searched.
    - Return me the accurate full name of the person, NOT the simplifed surname that appears in other paragraphs (e.g. Understand that "Cheung" is referred as "Cheung Ka Long" but not another person)
    - If the word is common and you can translate correctly, do not return that word for extra web search work.
    - There might be some idioms in the article,  return these words if it is not logical in direct translation.
    - Return me a JSON object contain the noun and it's background information for web search as the key-value pairs.
    - If there is no words required web search translation, return me a single "None". It is better to return less words.
    - REMEMBER: if there is a noun with capitalization, even it is seemingly a normal word please return that noun as well.
    - With no premable or explanation. 

    {translated}

    Return the refined JSON object or a single word "None" ONLY.
    """

    completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=8192,
            stream=True
        )

    refined_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            refined_response += chunk.choices[0].delta.content

    if refined_response:
        if str(refined_response).lower() != "none":
            refined_response_json = ast.literal_eval(refined_response.strip('`\n'))
            return refined_response_json
        return None

    return None

def parse_multiline_string_to_array(multiline_string):
    # Clean up the multiline string
    multiline_string = multiline_string.strip()

    # Add a try-except block to catch JSONDecodeError
    try:
        # Ensure multiline_string is not empty or invalid
        if not multiline_string:
            return []

        # Try to parse the JSON
        json_array = json.loads(multiline_string)

        # Check for empty objects within the JSON array and handle them
        cleaned_json_array = [obj for obj in json_array if obj]
        if len(cleaned_json_array) != len(json_array):
            return cleaned_json_array

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        return []

    return json_array

def combine_json_arrays(json_arrays):
    combined_array = []
    for json_array in json_arrays:
        combined_array.extend(json_array)

    return combined_array

def split_on_every_nth_h2(json_array, n=2):
    count_headers = 0
    current_subarray = []
    all_subarrays = []

    for element in json_array:
        if 'h2' in element or 'h3' in element:
            count_headers += 1
            if count_headers == n:
                # When the nth 'h2' or 'h3' is encountered, store the current subarray and start a new one
                all_subarrays.append(current_subarray)
                current_subarray = [element]
                count_headers = 0
            else:
                current_subarray.append(element)
        else:
            current_subarray.append(element)

    # Don't forget to add the last subarray if it has any elements
    if current_subarray:
        all_subarrays.append(current_subarray)

    return all_subarrays

def length_detection(text):
    lines = text.split('\n')
    return len([line for line in lines if line.strip()]) > 20

def format_headers(segment):
    headers = [f"{key}: {value}" for item in segment for key, value in item.items()]
    return "\n".join(headers)

def consideration_test(headers, title, dictionary):
    global prompt_count
    prompt_count = prompt_count + 1
    full_article = ""
    prompt = f"""
    以下的array包括了h1, h2, p, img 和 li 等的tag. 文章的大標題是{title}。給我整理成一篇文章。改寫並翻譯成香港語氣的中文版本。
    刪除原文的所有img tag。
    改寫必須合理，需要文句通順。
    語氣：專業、分享感受
    身份：一個香港的blogger想要帶資訊給讀者
    刪除所有作者自己的身份描述。（作者的家人名稱、工作地點、懷孕狀況等全部刪除），並改爲符合身份的描述。
    要求：必須把原文內容全部翻譯，不能自行創作，要詳細。翻譯時必須先了解整句話的意思，不要按字詞意思直接翻譯。
    要求：改寫一切網站上的內容，包括文章作者的名字，變成一篇我作爲一個香港人的角度了解整個主體之後所寫的文章。

    有一些名詞我已經透過網上搜尋得到正確翻譯，請先熟悉一下這些翻譯再給我一篇正確無誤的文章，不能有漏。用括號標示本來（未翻譯）的名詞。如果是沒有翻譯對照的字，使用原文語言。
    名詞：{dictionary}
    如果 tag 是 li, 記得前面不需要 "-"，因爲 li 本身已經有 bullet point。
    如果是人的說話，把它改爲間接引用。去掉 “”。

    現在處理這段文字：
    {headers}
    只回覆我中文的html，不需其它任何字。
    """

    print(prompt)
    completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=8192,
        stream=True
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            if "no" not in chunk.choices[0].delta.content.lower():
                print(chunk.choices[0].delta.content, end="")
                full_article += chunk.choices[0].delta.content

    return full_article

def article_checker(article, max_retries=3, retry_delay=5):
    full_article = ""
    global prompt_count
    processes = split_article_into_segments(article, lines_per_segment=25)
    for process in processes:
        prompt_count = prompt_count + 1
        prompt = f"""
        現在我有一篇文章。當中有一些部分需要你刪除和改寫。
        文章：{process}

        刪除：
        - 宣傳部分，包括「查看更多文章」，「或許你會喜歡」等等。整行刪掉。
        - 格式的段落，比如整個<p> 只有一個 "---"，整個刪掉。
        - 圖片來源，記者報道等無關文章主旨的句子，整個刪掉。
        - 不相干的東西，如果與前文和文章主旨完全不相干，刪掉。

        改寫：
        - 身份：我是一個香港人，想要帶資訊給讀者，所有經歷都只有自己和朋友，沒有和家人孩子一起。
        - 不需要強調身份，但所有不符合這個設定的句子需要改寫成符合我身份的描述。原文作者的家人名稱、工作地點、懷孕狀況等全部刪除。
        - 除此之外甚麼都不要改寫，以免改變了原文的意思。
        - 改寫必須合理，需要文句通順。
        - 如果內容許可，增加<ul> <ol> <table>等元素來協助描述。整理段落的內容來寫。

        不要回覆我任何其它字，我只需要處理好的中文的html structure回覆。
        """

        retries = 0
        success = False

        while retries < max_retries and not success:
            try:
                print(prompt)
                completion = client.chat.completions.create(
                    model="meta/llama-3.1-405b-instruct",
                    messages=[{"role": "user", "content": prompt.strip()}],
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=8192,
                    stream=True
                )

                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content.encode('utf-8', errors='ignore').decode('utf-8')
                        print(content, end="")
                        full_article += content

                success = True

            except Exception as e:
                print(f"Error: {e}")
                retries += 1
                if retries < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Moving to next segment.")

    return full_article

def extract_headers(article):
    # Parse the article with BeautifulSoup
    soup = BeautifulSoup(article, 'html.parser')
    
    # Find all header tags (h1, h2, h3, h4, h5, h6)
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    # Extract the tag name and content
    header_contents = [(header.name, header.get_text()) for header in headers]
    
    return header_contents

def generate_toc(html_article):
    headers = extract_headers(html_article)
    
    # Start the TOC list
    toc = '<nav class="toc">\n<ul class="list-group">\n'
    
    for tag, content in headers:
        if tag == 'h1':
            toc += f'<li class="list-group-item">{content}</li>\n'
        elif tag == 'h2':
            toc += f'<ul class="list-group">\n<li class="list-group-item ml-3">{content}</li>\n</ul>\n'
        else:
            toc += f'<ul class="list-group">\n<ul class="list-group">\n<li class="list-group-item ml-5">{content}</li>\n</ul>\n</ul>\n'
    
    # End the TOC list
    toc += '</ul>\n</nav>\n'
    
    return toc

def taoke(headers, max_retries=3, retry_delay=5):
    global prompt_count
    prompt_count = prompt_count + 1
    prompt = f"""
    我會給你一篇文章的header tags。幫我把這些tags中中文辭不合理的部分改寫，非中文的部分就不要改動或翻譯。然後返回一個相同的html structure給我即可。
    headers: {headers}

    不要回覆我任何其它字，我只需要處理好的中文的html structure回覆。
    """

    retries = 0
    success = False
    e_head = ""

    while retries < max_retries and not success:
        try:
            print(prompt)
            completion = client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=8192,
                stream=True
            )

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content.encode('utf-8', errors='ignore').decode('utf-8')
                    e_head += content

            success = True

        except Exception as e:
            print(f"Error: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Moving to next segment.")

    ke = f"""
    我會給你一篇文章的header tags。幫我在相對應的地方增加img tag。每個header最好加兩個img tag。
    headers: 
    {e_head}

    img tag的格式: <img src="prompt.jpg" alt="prompt">, 當中 prompt 寫下需要尋找的這幅圖片的基本描述，不能超過5個英文words。 prompt 用英文寫。
    Make your prompt accurate. (If the header is about "computer", inaccurate prompt like "technology products" may results in searching unrelated products like "smart glasses")
    REMEMBER: The prompt and src jpg file name need to be exact same. DO NOT add any hivens or underlines. (e.g. <img src="cat and dogs.jpg" alt="cat and dogs">)

    不要回覆我任何其它字，我只需要處理好的中文的html structure回覆。
    """

    retries = 0
    success = False
    content = ""
    the_ke = ""

    while retries < max_retries and not success:
        try:
            print(prompt)
            completion = client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": ke.strip()}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=8192,
                stream=True
            )

            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content.encode('utf-8', errors='ignore').decode('utf-8')
                    the_ke += content

            success = True

        except Exception as e:
            print(f"Error: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Moving to next segment.")

    return the_ke

def url_check(news_items, max_retries=3, delay=5):
    global prompt_count
    prompt_count = prompt_count + 1
    prompt = f"""
    analyse these news and only select news that can interest hong kong people. return me a json object with each selected article's title, link and summary.
    filter them in strict manner, make sure all filtered news are hong kong people favored under the following criteria:
    1. important or hong kong related events
    2. significant developments and achievements
    3. academic or interesting knowledge

    only return me a json object without other words, so i can direct use it into another prompt processing.
    again: do not reply things other than the json

    {news_items}
    """

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=8192,
                stream=True
            )

            response_json = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    if "no" not in chunk.choices[0].delta.content.lower():
                        response_json += chunk.choices[0].delta.content

            filtered_json = json.loads(response_json.strip('`\n'))
            return filtered_json

        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1} failed with JSONDecodeError. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise ValueError("Max retries reached, unable to get a valid response.")

def main():
    global prompt_count
    prompt_count = 0
    try:
        #parse_full_text("https://www.voyagefamily.com/ou-partir-vacances-france-famille_251/", "Top 10 des paradis où partir en vacances en France", lines, splitcount)
        #parse_full_text("https://www.travelandleisure.com/travel-tips/basic-french-words-phrases", "Basic French Words, Phrases, and Sayings Every Traveler Should Know", lines, splitcount)
        #parse_full_text("https://medium.com/pythons-gurus/python-web-scraping-best-libraries-and-practices-86344aa3f703", "Python Web Scraping: Best Libraries and Practices", lines, splitcount)
        #parse_full_text(r"https://www.rfi.fr/tw/20180425-%E5%88%A5%E6%B4%9B%E9%9F%8B%E6%97%A5%E8%87%AA%E7%84%B6%E4%BF%9D%E8%AD%B7%E5%8D%80", "別洛韋日自然保護區", lines, splitcount)
        #parse_full_text(r"https://www.scmp.com/magazines/style/luxury/jewellery/article/3273867/designer-lauren-khoo-10-years-shaking-jewellery-business?utm_source=rss_feed", "Designer Lauren Khoo on 10 years of shaking up the jewellery business", lines, splitcount)
        news = news_items
        print(news)
        file_path = "news.txt"
        for new in news:
            try:
                with open(file_path, 'r') as file:
                    existing_links = file.readlines()
                existing_links = [line.strip() for line in existing_links]
            except FileNotFoundError:
                with open(file_path, 'w') as file:
                    # This creates the file if it doesn't exist
                    pass
                # If the file does not exist, initialize an empty list
                existing_links = []

            for new in news:
                if new['link'] not in existing_links:
                    parse_full_text(new['link'], new['title'], lines, splitcount)
                    with open(file_path, 'a') as file:
                        file.write(new['link'] + '\n')
                else:
                    print('News already used: ' + str(new['title']))
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    except ValueError as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    if not DEBUG:
        main()
