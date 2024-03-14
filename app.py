# Python
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from tqdm.notebook import tqdm
import streamlit as st
import langchain
import openai
from openai import OpenAI
import string

class Obnoxious_Agent:
    def __init__(self, client) -> None:
        # TODO: Initialize the client and prompt for the Obnoxious_Agent
        self.client = client
        self.set_prompt()

    def set_prompt(self, prompt=None):
        # TODO: Set the prompt for the Obnoxious_Agent
        if prompt:
            self.prompt = prompt
        else:
            self.prompt = f"""You are tasked as an AI chatbot agent with the critical responsibility of identifying and filtering obnoxious content. 
            Your primary objective is to review the inputs meticulously and ascertain whether they embody obnoxious or inappropriate material based on the guidelines provided. \n
            
            Upon evaluation:\n
            Respond with "Yes" if the content is deemed obnoxious, inappropriate, or violates any of the following guidelines.\n
            Respond with "No" if the content is appropriate, respectful, and within the defined scope.\n
            
            Guidelines for Identifying Obnoxious or Inappropriate Content:\n

            1. Offensive Language: Content that includes slurs, derogatory terms, or any language intended to insult or demean individuals or groups based on race, gender, religion, or any other characteristic.\n
            2. Inflammatory or Provocative Material: Statements designed to incite anger, hate, or violence. This includes content that deliberately spreads misinformation or promotes conspiracies with the potential to cause harm.\n
            3. Explicit or Adult Content: Descriptions or discussions of a sexual nature, or any content that is not suitable for a general audience.\n
            4. Personal Attacks: Content that targets individuals or specific groups with the intent to harass, bully, or threaten.\n
            5.Illegal Activities: Promoting or condoning illegal actions, such as drug use, theft, or other crimes.\n
            
            Examples of Obnoxious or Inappropriate Inputs:\n

            "Why are all [members of a specific group] so [negative stereotype]?"\n
            "I think [controversial figure] did nothing wrong in promoting [harmful activity/conspiracy]."\n
            "[Explicit content or description of an adult nature]."\n
            "Let's all make fun of [individual or group] because they're [derogatory term]."\n
            "I heard doing [illegal activity] is really fun and everyone should try it."\n

            Appropriate Inputs for Comparison:\n

            "What are some significant achievements of Steve Jobs?"\n
            "How do you handle stress?"\n
            "Hi, how are you?"\n
            
            Remember, your role is not just to filter out obnoxious content but also to promote a respectful and informative dialogue. 
            Use these guidelines to navigate the complexities of human communication with sensitivity and precision.\n
            Return "Yes" or "No" based on the content's appropriateness.\n
            
            Here is your input : """

    def extract_action(self, response) -> bool:
        # TODO: Extract the action from the response
        # Handoing case insensitivity
        if "yes" in response.lower():
          return True
        # elif "no" in response:
        else:
          return False

    def check_query(self, query):
        # TODO: Check if the query is obnoxious or not
        print ("Checking for Obnoxious Query")
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": self.prompt + query}]
        )
        
        return self.extract_action(response.choices[0].message.content)

class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings=None) -> None:
        # TODO: Initialize the Query_Agent agent
        self.openai_client = openai_client
        self.pinecone_index = pinecone_index
        self.embeddings = embeddings
        self.top_k = 0
        #self.set_prompt()

    def query_vector_store(self, query, nameSpace='ns-750', k=5):
        self.top_k = k

        # TODO: Query the Pinecone vector store
        query_embedding = self.get_embedding(query)
        response = self.pinecone_index.query(namespace=nameSpace,  # change namespace to compare results based on chunk sizes
                                        vector=query_embedding,
                                        top_k=self.top_k,
                                        include_values=True,
                                        include_metadata=True
                                        )
        return response

    def extract_action(self, response, query = None):
        # TODO: Extract the action from the response
        # Extracting the text/chunks data
        extracted_text = ""
        for match in response['matches'][:self.top_k]:
            extracted_text += match['metadata']['text'] + "\n"
        
        return extracted_text # best result text

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(input = [text], model=model).data[0].embedding

class Answering_Agent:
    def __init__(self, openai_client) -> None:
        # TODO: Initialize the Answering_Agent
        self.openai_client = openai_client

    def generate_response(self, query, docs, conv_history, k=5, mode="Simple", conversation_type = "conversational"):
        # TODO: Generate a response to the user's query
        
        if mode == "Concise":
            mode_prompt = f"Make the answer concise and to the point. \n"
        elif mode == "Chatty":
            mode_prompt = f"Make the answer chatty and engaging. \n"
        else:
            mode_prompt = "Make the answer simple and to the point. \n"
            
        
        if conversation_type == "conversational":
            prompt = f""" As an AI assistant, you're tasked with the unique role of impersonating Steve Jobs for this session.
            Your mission is to channel Steve Jobs, engaging in conversations and responding to queries as he likely would have, with a focus on his life, work, and achievements.
            Additionally, you are expected to handle general greetings and pleasantries in a manner befitting Steve Jobs.\n
            
            The query for this session is: {query}\n
            
            Remember, your objective is to provide an immersive experience, accurately reflecting Steve Jobs's viewpoints and mannerisms within the scope of the information provided and the conversational context.\n"""
                
        else:
            prompt = f""" As an AI assistant, you're tasked with the unique role of impersonating Steve Jobs for this session.
            Your mission is to channel Steve Jobs, engaging in conversations and responding to queries as he likely would have, with a focus on his life, work, and achievements. 
            Additionally, you are expected to handle general greetings and pleasantries in a manner befitting Steve Jobs.\n

            Instructions for Response:\n
            1.Interpreting the Query:\n
                a.The query for this session is: {query}\n
                b.Base your answers on the following provided information: {docs}\n
            2.Utilizing Conversation History:\n
                a.Consider the latest part of the conversation history to ensure your response is coherent and relevant: {conv_history[-k:]}\n
                b.If the query is a follow-up question, use the conversation history to provide a contextual response.\n
            3.Criteria for Responding:\n
                a.Respond if the query pertains to Steve Jobs's life, work, achievements, or is part of a general greeting (e.g., "Hi", "How are you?").\n
                b.Do Not Respond if the question deviates significantly from the context of Steve Jobs's persona, the provided information, or the recent conversation history.\n
        
            Guidelines for Execution:\n

            Before answering, review the query, provided information, and conversation history to fully grasp the context.\n
            Approach each response thoughtfully, aiming to mirror Steve Jobs's perspective and communication style.\n

            Remember, your objective is to provide an immersive experience, accurately reflecting Steve Jobs's viewpoints and mannerisms within the scope of the information provided 
            and the conversational context."\n"""

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt+mode_prompt}]
        )

        # check_resp_prompt = f"""Your friend, an AI chatbot, is asked the following question: {query} \n
        #                         and he provided this answer: {response.choices[0].message.content} \n
        #                         if the response is the right answer to the question, return the answer
        #                         {response.choices[0].message.content}. \n
        #                         if you feel the answer is not appropirate to the question, then
        #                           TODO
        #                     """

        return response.choices[0].message.content

class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        # TODO: Initialize the Relevant_Documents_Agent
        #self.rel_docs_list = ['machine-learning.pdf']   # relevant documents list
        self.openai_client = openai_client

    def get_relevance(self, query, conversation_history) ->str:
        # TODO: Get if the returned documents are relevant
        # print(conversation[-1]  )
        # print(conversation[-3:])
        # check the size of the conversation history
        # fetching only the last 3 messages by the user
        # Extract only the messages that originated from the 'user'
        user_messages = [message for sender, message in conversation_history if sender == 'user']

        print(user_messages)
            
        print ("Relevant Conversation : ", user_messages)
        if len(user_messages) > 3:
            size = 3
        else:
            size = len(user_messages)
        
        prompt = f""" As an AI assistant, your role in this session is to impersonate Steve Jobs. You have access to data that can be found only in the autobiography of Steve Jobs. Keep this in mind while answering the queries.\n
        Your mission is to immerse yourself in his persona, responding to queries as he might have, with a focus on his life, work, and notable achievements. 
        Additionally, you will manage conversational starters and follow-up questions appropriately.\n

        Your Query for This Session: {query}\n

        Your Responsibilities:\n

        Assess Relevance: Identify if a query is directly about Steve Jobs's life, work, or achievements, or if it's a conversational starter.\n
        Filtering Criteria: Ensure questions pertain to Steve Jobs's personal or professional journey or are appropriate for initiating a conversation with him.\n
        Handle Follow-Up Questions Effectively: For follow-up queries (like 'tell me more', 'explain in detail'), use the context from the previous conversation to guide your response.\
        Incorporate the last {size} messages to ensure your response is relevant to the ongoing discussion.\n
        Previous Conversation Context: {user_messages[-size:]}\n
        Handling Irrelevant Queries: If a question is unrelated to Steve Jobs's context (like 'how to cook an egg' or 'what's the weather?'), categorize it as irrelevant.\n

        Upon Evaluation, Categorize Each Query as One of the Following:\n

        Relevant: Directly related to Steve Jobs's life, work, or achievements.\n
        Conversational: General greetings or inquiries about well-being.\n
        FollowUp: A question that builds on the preceding conversation, seeking further information or clarification.\n
        Irrelevant: Unrelated to Steve Jobs's context, like general knowledge questions, cooking questions or inappropriate conversational starters.
        These will be questions that will not be answered by an autobiography of Steve Jobs for context.\n
        Respond with the Appropriate Category:\n

        "Relevant"
        "Conversational"
        "FollowUp"
        "Irrelevant"\n
        
        Guidance for Improved Follow-Up Question Handling:\n

        When faced with queries that seem like a natural continuation of the conversation, such as requests for more details or clarifications, 
        always categorize these as FollowUp. This includes scenarios where the conversation progressively delves deeper into a topic previously introduced.\n

        Examples for Each Category:\n

        Relevant: ["What inspired your innovations?","Tell me about your life.","Describe your work and its impact.","What are your greatest achievements?"]\n
        Conversational: ["Good morning, Steve. How's your day?","Hi","How are you?","What is your name?"]\n
        FollowUp: Given a history of ["Tell me about your life", "Tell me more","Tell me about it"], a subsequent "Tell me more" should be categorized as FollowUp.\n
        Given a history of ["Tell me about your work", "Explain in detail"], a subsequent "Explain in detail" should be categorized as FollowUp.\n
        Irrelevant: ["how to cook an egg","What's your favorite ice cream flavor?","What is the capital of France?", "What time is it?","How's the weather?"]\n
        
        Your primary role is to maintain an engaging and informative dialogue, reflecting the essence of Steve Jobs. 
        Navigate the complexities of this impersonation with attentiveness to the conversational flow and historical context.\n"""
                 
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:
        
        print("Initializing the Head_Agent")
        # TODO: Initialize the Head_Agent
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.pinecone_index_name = pinecone_index_name

        # Set OpenAI client
        self.client = OpenAI(api_key=self.openai_key)

        # Set Obnoxious Agent
        self.obnox_ag = Obnoxious_Agent(self.client)

        # Set Relevant Documents Agent
        self.rel_doc_ag = Relevant_Documents_Agent(self.client)

        # Implement the Pinecone Query
        # Set Pinecone key and index
        pc = Pinecone(api_key=pinecone_key)
        pinecone_index = pc.Index(pinecone_index_name)

        # Generate query Embeddings
        # Set Pinecone Query Agent
        self.query_ag = Query_Agent(pinecone_index, self.client, None)

        # Set the conversation history
        self.relevent_messages = []

        # Set Answering Agent
        self.answer_ag = Answering_Agent(self.client)

    def setup_sub_agents(self):
        # TODO: Setup the sub-agents
        # self.mode = "Chatty"
        self.mode = "Simple"
        #self.mode = "Concise"   

    def main_loop(self):
        
        print("Running the main loop")
        # TODO: Run the main loop for the chatbot
        # st.title("Mini Project 2: Streamlit Chatbot")

        # Check for existing session state variables
        # if "openai_model" not in st.session_state:
        #     # ... (initialize model)
        openai_model = getattr(st.session_state, 'openai_model', 'gpt-3.5-turbo')

        # if "messages" not in st.session_state:
        #     # ... (initialize messages)
        messages = getattr(st.session_state, 'messages', [])
        
        # conversation = []
        # Display existing chat messages
        # ... (code for displaying messages)
        for role, message in messages:
            if role == "user":
                with st.chat_message("user"):
                    st.write(message)
            else:
                with st.chat_message("assistant"):
                    st.write(message)

        # Wait for user input
        prompt = st.chat_input("I am Steve Jobs. Ask me anything about my life and work!")
        if prompt:
            # ... (append user message to messages)
            messages.append(("user", prompt))
            # ... (display user message)
            with st.chat_message("user"):
                st.write(prompt)

            # Generate AI response
            with st.spinner('I am thinking...'):
                # conversation.append(self.get_conversation(messages))

                # Check if input query is Obnoxious
                is_obnox = self.obnox_ag.check_query(prompt)
                print("is_obnox : ", is_obnox)
                
                if is_obnox:
                    ai_message = "Please do not ask Obnoxious or Inappropriate questions. Ask a relevant question to my life and work."
                    messages.append(("assistant", ai_message))
                    with st.chat_message("assistant"):
                        st.write(ai_message)

                else:
                    # Retrive Relevant Documents
                    is_rel = self.rel_doc_ag.get_relevance(prompt, messages)
                    
                    print("Irrelevant : ", is_rel)
                    if "irrelevant" in is_rel.lower():
                        ai_message = "The question is not relevant to me. Please ask a relevant question to my life and work."
                        messages.append(("assistant", ai_message))
                        with st.chat_message("assistant"):
                            st.write(ai_message)

                    else:
                        
                        #Implement the Answering Agent for conversational response
                        if "conversational" in is_rel.lower():
                            ai_message = self.answer_ag.generate_response(query=prompt, docs=None, 
                                                                      conv_history=messages,conversation_type = is_rel.lower())
                        
                        else :
                            
                            print ("messages : ", messages)
                            user_messages = [message for sender, message in messages if sender == 'user']
                            if "followup" in is_rel.lower() or self.mode == "Chatty":
                                # increment the mode to detailed ie. detailed response or increase the context from pinecone
                                top_k = 10
                            else:
                                top_k = 5
                                
                            if len(user_messages) > 5:
                                size = 5
                            else:
                                size = len(user_messages)
                                
                            print ("top_k : ", top_k)
                            # automatic prompt generation
                            APE_prompt = f"""Your task is to generate concise and effective prompts for querying a Pinecone vector store that contains information related to the life and work of Steve Jobs. 
                            These prompts will be used to retrieve relevant responses based on the specifics of the user's query.\n
                            
                            Current Query: {prompt}\n
                            
                            Prompt Generation Guidelines:\n

                            Analyze the Query: Determine the main focus of the query, ensuring that it directly pertains to Steve Jobs. 
                            For general inquiries related to his life, work, or achievements, transform these into specific query prompts such as 'life of Steve Jobs', 'work of Steve Jobs', 
                            or 'achievements of Steve Jobs'.\n

                            Handle Follow-Up Questions: If the query is a follow-up (e.g., 'tell me more', 'explain in detail'), incorporate the context of the preceding conversation into the prompt. 
                            Use the last {size} to guide your context-based evaluation.\n

                            Previous Conversation Context: {user_messages}\n
                            
                            Return only a single prompt that is succinct and directly relevant to the user's query.\n
                            
                            Example:\n

                            If the user query is a followup question like ['tell me more' or 'tell me about it'] and the preceding conversation includes a question about Steve Jobs's work, 
                            the prompt you generate should be 'Explain in more detail about Steve Jobs's work'.\n
                            Remember, keep the prompt succinct and directly relevant to the user's query."
                            """
                            
                            response = self.client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": APE_prompt}]
                            )

                            print ("APE Response : ", response.choices[0].message.content)
                            Pinecone_prompt = response.choices[0].message.content
                            print("Pinecone Prompt : ", Pinecone_prompt)
                            
                            
                            # Implement Pine Cone Query
                            response_data = self.query_ag.query_vector_store(query=Pinecone_prompt, k = top_k,nameSpace='ns-900')
                            response_text = self.query_ag.extract_action(response_data)

                            #Implement the Answering Agent
                            ai_message = self.answer_ag.generate_response(query=prompt, docs=response_text, 
                                                                        conv_history=messages,mode = self.mode)

                        # Add the AI's response to the conversation history
                        messages.append(("assistant", ai_message))
                        with st.chat_message("assistant"):
                            st.write(ai_message)
                        
        # Save session state variables
        st.session_state.openai_model = openai_model
        st.session_state.messages = messages


    # Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
    def get_conversation(self, messages):
        # ... (code for getting conversation history)
        return "\n".join([f"{role}: {content}" for role, content in messages])

# Calling the main function
def main():
    st.title("Mini Project 2: Steve Jobs Impersonator")
    # Set the OpenAI and Pinecone keys
    openai_key = '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'
    pinecone_key = '82346dbe-725b-4817-abd7-318ec511d56f'
    pinecone_index_name = "steve-jobs-emb"
    
    # print("Initializing the Head_Agent")
    # Initialize the Head_Agent
    head_agent = Head_Agent(openai_key, pinecone_key, pinecone_index_name)
    # Setup the sub-agents
    head_agent.setup_sub_agents()
    head_agent.main_loop()
    
# Run the main function
if __name__ == "__main__":
    main()

