import os
import sys

from semantic_router import RouteLayer
sys.path.append('../')
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from tiktoken import get_encoding
from weaviate.classes.query import Filter
from litellm import completion_cost
from loguru import logger 
import streamlit as st

from src.database.weaviate_interface_v4 import WeaviateWCS
from src.llm.llm_interface import LLM
from src.reranker import ReRanker
from src.txt2sql import Text2SQL
from src.llm.prompt_templates import generate_prompt_series, huberman_system_message
from app_functions import (convert_seconds, search_result, validate_token_threshold,
                           stream_chat, load_data)

from src.database.weaviate_interface_v4 import WeaviateWCS
from src.database.weaviate_interface_v4 import WeaviateIndexer
from src.preprocessor.preprocessing import FileIO
from weaviate.classes.config import Property, DataType
from sentence_transformers import SentenceTransformer

 
## PAGE CONFIGURATION
st.set_page_config(page_title="Huberman Labs", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

###################################
#### SET UP APP CONFIGURATION #####
###################################

# Example models
# turbo = 'gpt-3.5-turbo-0125'
# claude = 'claude-3-haiku-20240307'

ICON_DIR = './app_assets/'
reader_model_name = 'gpt-4o-mini'
# collection_name = None # set to None to use the sidebar selectbox
data_path = '../data/huberman_labs.json'
embedding_model_path = '../models/allminilm-finetuned-256/'
###################################

## RETRIEVER
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
model_path = 'sentence-transformers/all-MiniLM-L6-v2'

retriever = None
guest_list = None
data = None
router = None
reranker = None
llm = None

# in Streamlit, this entire script is re-run on each user input
# so only instantiate retriever once per session
if not st.session_state.get('retriever'):
    st.session_state['retriever'] = WeaviateWCS(endpoint=url, api_key=api_key, model_name_or_path=model_path)
retriever = st.session_state['retriever']
if retriever._client.is_live():
    logger.info('Weaviate is ready!')

# only load and sort the data once per session
if not st.session_state.get('data'):
    # load data from json file
    data = FileIO.load_json(data_path)
    st.session_state['data'] = data
    st.session_state['guest_list'] = sorted(list(set([d['guest'] for d in data])))
else:
    data = st.session_state['data']
    guest_list = st.session_state['guest_list']


# only create a router once per session, and only if a semantic router file is available
if not st.session_state.get('router'):
     # if a semantic router file is available, load it 
    if os.path.exists('./semantic_router.json'):
        router = RouteLayer.from_json('./semantic_router.json')
        st.session_state['router'] = router
    else:
        st.session_state['router'] = None   

router = st.session_state['router']


# only instantiate a reranker once per session
if not st.session_state.get('reranker'):
    # instantiate reranker
    st.session_state['reranker'] = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
reranker = st.session_state['reranker']

## QA MODEL
# only instantiate the connection to the LLM once per session
if not st.session_state.get('llm'):
    # instantiate the LLM
    # if you want to use a different model, change the model name here
    # e.g. 'gpt-3.5-turbo-0125' or 'claude-3-haiku-20240307'
    st.session_state['llm'] = LLM(model_name=reader_model_name)
llm = st.session_state['llm']



## TOKENIZER
encoding = get_encoding("cl100k_base")

## Display properties
display_properties = None
verbosity_options = ['low', 'medium', 'high']



# best practice is to dynamically load collections from weaviate using client.show_all_collections()
# available_collections = ['Huberman_minilm_128', 'Huberman_minilm_256', 'Huberman_minilm_512']
available_collections = retriever.show_all_collections()

## COST COUNTER
if not st.session_state.get('cost_counter'):
    st.session_state['cost_counter'] = 0

def main(retriever: WeaviateWCS):
    #################
    #### SIDEBAR ####
    #################
    with st.sidebar:
        collection_name = st.selectbox( 'Collection Name:',options=available_collections, index=None,placeholder='Select Collection Name')
        guest_input = st.selectbox('Select Guest', options=guest_list,index=None, placeholder='Select Guest')
        alpha_input = st.slider('Alpha (Hybrid Search Weighting)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        retrieval_limit = st.number_input('Retrieval Limit', min_value=1, value=5)
        reranker_topk = st.number_input('Reranker Top K', min_value=1, value=5)
        temperature_input = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        verbosity = st.selectbox("Verbosity:", options=range(len(verbosity_options)), format_func=lambda i: verbosity_options[i], index=0)
    # retriever.return_properties.append('expanded_content')

    ##############################
    ##### SETUP MAIN DISPLAY #####
    ##############################
    st.image(f'{ICON_DIR}/hlabs_logo.png', width=400)
    st.subheader("Search with the Huberman Lab podcast:")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')
     #   if query:
     #       st.write('This app is not currently functioning as intended. Uncomment lines 104-172 to enable Q&A functionality.')

   
        

    ########################
    ##### SEARCH + LLM #####
    ########################
    if query and not collection_name:
        st.error('Please first select a collection name')

    if query:
         # make hybrid call to weaviate
        guest_filter = Filter.by_property(name='guest').equal(guest_input) if guest_input else None
         # if a router is available, use it to determine the route
         # otherwise, use the regular hybrid search
        route='regular'
        if router:
            route = router(query).name
        if route == 'regular':
            display_properties = [prop.name for prop in retriever.show_collection_properties(collection_name)]

            hybrid_response = retriever.hybrid_search(query, collection_name, alpha=alpha_input, limit=5, return_properties=display_properties,
                                                  filter=guest_filter)
         
            ranked_response = reranker.rerank(hybrid_response, query, apply_sigmoid=True)       
            logger.info(f'# RANKED RESULTS: {len(ranked_response)}')   

            token_threshold = 2500 # generally allows for 3-5 results of chunk_size 256
            content_field = 'content'

            # validate token count is below threshold
            valid_response = validate_token_threshold(  ranked_response, 
                                                     query=query,
                                                     system_message=huberman_system_message,
                                                     tokenizer=encoding,# variable from ENCODING,
                                                     llm_verbosity_level=verbosity,
                                                     token_threshold=token_threshold, 
                                                     content_field=content_field,
                                                     verbose=True)
            logger.info(f'# VALID RESULTS: {len(valid_response)}')
            #set to False to skip LLM call
            make_llm_call = True
            # prep for streaming response
            with st.spinner('Generating Response...'):
                st.markdown("----")
                # generate LLM prompt
                prompt = generate_prompt_series(query=query, results=valid_response, verbosity_level=verbosity)
                if make_llm_call:
                    with st.chat_message('Huberman Labs', avatar=f'{ICON_DIR}/huberman_logo.png'):
                        stream_obj = stream_chat(llm, prompt, max_tokens=250, temperature=temperature_input)
                        st.write_stream(stream_obj) # https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream
            
                # need to pull out the completion for cost calculation
                string_completion = ' '.join([c for c in stream_obj])
                call_cost = completion_cost(completion=string_completion, 
                                         model=reader_model_name, 
                                         prompt=huberman_system_message + ' ' + prompt,
                                         call_type='completion')
                st.session_state['cost_counter'] += call_cost
                logger.info(f'TOTAL SESSION COST: {st.session_state["cost_counter"]}')

    # ##################
    # # SEARCH DISPLAY #
    # ##################
                st.subheader("Search Results")
                for i, hit in enumerate(valid_response):
                    col1, col2 = st.columns([7, 3], gap='large')
                    episode_url = hit['episode_url']
                    title = hit['title']
                    show_length = hit['length_seconds']
                    time_string = convert_seconds(show_length) # convert show_length to readable time string
                    with col1:
                        st.write( search_result(i=i, 
                                             url=episode_url,
                                             guest=hit['guest'],
                                             title=title,
                                             content=ranked_response[i]['content'], 
                                             length=time_string),
                                             unsafe_allow_html=True)
                        st.write('\n\n')

                    with col2:
                        image = hit['thumbnail_url']
                        st.image(image, caption=title.split('|')[0], width=200, use_container_width=False)
                        st.markdown(f'<p style="text-align": right;"><b>Guest: {hit["guest"]}</b>', unsafe_allow_html=True)
        else:
            # router thinks this is better answered as a sql query 
            txt2sql = Text2SQL(llm, retriever, collection_name, reranker)
            response = txt2sql(route, query)
            st.write(f'Query results: {response[0]}')
if __name__ == '__main__':
    main(retriever)
