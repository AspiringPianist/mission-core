# mission-core
Curated Learning Resources for GenAI. Get started with the basics - build AI software, agents and chatbots!

# Setup
## Python Setup
First create a new python virtual environment and activate it. During the creation of this repository and code, python 3.11.5 was used, but anything above 3.11+ should work fine.  

```bash
python -m venv env

# On Windows
./env/Scripts/activate
```

Now install all the requirements with  

```bash
pip install -r requirements.txt
```

Start jupyter notebook in the root folder of this repository using  
   
```bash
jupyter notebook
```

## Groq and Ollama (optional) Setup
We will be accessing LLMs via the Groq console or run them locally on your own machine using Ollama. We prefer using Groq because it requires minimal setup and won't waste your time learning here.

Note for Ollama users: We recommend you have a NVIDIA GPU of atleast 4GB VRAM or a macbook with Apple Silicon.
If you don't have a GPU, a much better option is to use the free Groq API.
Get the Groq API Key from here 
```https://console.groq.com/keys```

Please rename the .env.example file to .env and paste in your Groq API Key here (within "") (optional if you are using ollama)  

```cp .env.example .env```

Langchain provides clear and concise ways to access many LLM Providers with similar syntax so we will accessing the Groq LLM using the `langchain-groq` library and LLMs via Ollama using the `langchain-ollama` library.

