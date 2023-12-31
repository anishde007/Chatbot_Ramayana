
import gradio as gr
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

apii=os.environ('spi')
COUNT, N = 0, 0
chat_history = []
chain = ''
# enable_box = gr.Textbox.update(value=None,
#                           placeholder='Upload your OpenAI API key', interactive=True)
# disable_box = gr.Textbox.update(value='OpenAI API key is Set', interactive=False)
def database():
    with open('database.txt', 'r', encoding='utf-8') as file:
    # Read the content of the file
        document = file.read()
    def split_text_into_batches(text, batch_size):

        batches = []
        for i in range(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            batches.append(batch)
        return batches
    documents=split_text_into_batches(str(document),400)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.from_texts(documents, embeddings)

    return db

def set_apikey(api_key):
    os.environ["HUGGINFACEHUB_API_TOKEN"] = apii
    return disable_box
def enable_api_box():
    return enable_box
def add_text(history, text):
    if not text:
        raise gr.Error('Enter text')
    history = history + [(text, '')]
    return history
def generate_response(history, query):
    global COUNT, N, chat_history, chain
    db=database()
    llm=HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.5, "max_length":90},huggingfacehub_api_token=apii)
    chain = load_qa_chain(llm, chain_type="refine")
    doc = (db.similarity_search_with_score(query))
    score=doc[0][-1]
    doc = doc[0][:-1]
    threshold =   1
    if score > threshold:
        # No relevant information found or information is below the specified threshold
        result="Sorry, but I can't answer that at the moment."
        print("Sorry, but I can't answer that at the moment.")
    else:
        # Relevant information found, proceed with the chain
        result=chain.run(input_documents=doc, question=query)
        print(chain.run(input_documents=doc, question=query))

    chat_history += [(query, result)]

    for char in result:
        history[-1][-1] += char
        yield history, ''

with gr.Blocks() as demo:
    # Create a Gradio block

    with gr.Column():


        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id='chatbot')
            # chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=570)

    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter"
            )
            # ).style(container=False)

        with gr.Column(scale=1):
            submit_btn = gr.Button('Submit')



    # Event handler for submitting text and generating response
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=generate_response,
        inputs=[chatbot, txt],
        outputs=[chatbot, txt]
    )

if __name__ == "__main__":    
    demo.queue()
    demo.launch(debug=True)

