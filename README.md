# Steps to Replicate
1.Clone the repository to your local machine:
  https://github.com/Phillament/CHATBOT.git
2.Run the following command in the terminal to install necessary python packages:
  pip install -r requirements.txt
3.Run the following command in your terminal to start the chat UI:
  chainlit run chatbot.py -w

# Code Structure
  **Imports:**
   Imports necessary libraries and modules.
   Dependencies include OpenAI, Google Generative AI, PyPDF2, and others.

 **Environment Setup:**
   Loads environment variables, including API keys for services like OpenAI.

 **Text Processing:**
   Defines a text splitter to divide the extracted text from a PDF into manageable chunks for further processing.

 **Chat Initialization:**
   Defines a function 'on_chat_start()' that initiates the chat process.
   Waits for the user to upload a PDF file.
   Processes the uploaded file by extracting text from it using PyPDF2.
   Splits the extracted text into chunks.
   Creates metadata for each chunk.
   Initializes a vector store using OpenAI embeddings.
   Sets up a conversation buffer memory and a conversational retrieval chain using the created vector store.
   Notifies the user when the system is ready for interaction.

 **Message Handling:**
   Defines a function 'main()' to handle incoming messages during the conversation.
   Retrieves the conversational chain from the user session.
   Processes the message content using the chain and sends back a response to the user.

