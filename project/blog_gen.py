# import statements
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import requests
import os
from pydantic import BaseModel, Field
from typing import List
import base64
from bs4 import BeautifulSoup


class Titles(BaseModel):
    titles: List[str] = Field(description="List of titles for the blog post", min_items=4, max_items=4)

class SectionContent(BaseModel):
    content: str = Field(description="Detailed content of the section")

class Section(BaseModel):
    title: str = Field(description="Title of the blog post section")
    content: str = Field(description="Detailed content for the title")

class BlogGenerator():
    def __init__(self, GROQ_API_KEY, HF_API_KEY, TEMPLATE, PDF):
        print('Initializing Blog Generator...')
        self.groq_api_key = GROQ_API_KEY
        self.hf_api_key = HF_API_KEY
        self.template = None
        self.pdf_path = PDF
        self.pdf_reader = PyPDF2.PdfReader(self.pdf_path)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatGroq(api_key=self.groq_api_key, model_name="llama-3.3-70b-specdec", temperature=0.7)
        self.hf_endpoint = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TEMPLATE)
        with open(template_path, 'r', encoding='utf-8') as file:
            self.template = file.read()
        self.soup = BeautifulSoup(self.template, "html.parser")
        print('Initialized Blog Generator !')
    def process_pdf(self):
        print('Processing PDF...')
        text = ""
        for page in self.pdf_reader.pages:
            text += page.extract_text()
        
        chunks = self.text_splitter.split_text(text)
        print('PDF processed !')
        return Chroma.from_texts(
            texts = chunks,
            embedding = self.embeddings,
            persist_directory="./chroma_db"
        )
    
    def generate_content(self, vectorstore):
        print('Generating Titles..')
        title_llm = self.llm.with_structured_output(Titles)
        rag_query = " ".join([doc.page_content for doc in vectorstore.similarity_search(
            "Extract main topics for blog sections",
            k=3
        )])
        titles = title_llm.invoke(f"""Based on this context: {rag_query}
            Generate exactly 4 section titles for a technical blog post.
            Return only the titles in a list format.
            Make them clear and professional.""")
        print('Titles Generated!', titles)
        sections = []
        for title in titles.titles:
            print('Generating Content for', title)
            section_context = " ".join([doc.page_content for doc in vectorstore.similarity_search(
                f"Find information about {title}",
                k=3
            )])
            query = f"""Context: {section_context}
                Title: {title}
                Task: Write detailed content for this section.
                Requirements:
                - Professional tone
                - Clear explanation
                - Relevant to the title
                - 1-2 paragraphs"""
            content_llm = self.llm.with_structured_output(SectionContent)
            content = content_llm.invoke(query)
            print('Content Generated!', content.content)
            sections.append(Section(title=title, content=content.content))
        return sections
    
    def generate_image(self, prompt):
        print('Generating Image...')
        response = requests.post(
            self.hf_endpoint,
            headers={"Authorization": f"Bearer {self.hf_api_key}"},
            json = {
                "inputs": prompt,
                "parameters": 
                    {"width": 768, "height": 512}
    
            }
        )
        print(response)
        print('Generated Image!')
        return base64.b64encode(response.content).decode()
    
    def generate_blog(self, title, tags, description):
        print('Generating Blog...')
        vectorstore = self.process_pdf()
        sections = self.generate_content(vectorstore)
        header = self.soup.find('header')
        header.find('h1').string = title
        header.find('p').string = tags
        
        # Add introduction
        main = self.soup.find('main')
        intro_section = self.soup.new_tag('section', id='introduction')
        intro_h2 = self.soup.new_tag('h2')
        intro_h2.string = "Introduction"
        intro_p = self.soup.new_tag('p')
        intro_p.string = description
        intro_section.extend([intro_h2, intro_p])
        main.clear()
        main.append(intro_section)

        #Generate and add further sections
        for i, section in enumerate(sections, 1):
            section_elem = self.soup.new_tag('section', id=f'section-{i}')
            section_h2 = self.soup.new_tag('h2')
            section_h2 = section.title
            section_p = self.soup.new_tag('p')
            section_p.string = section.content
            section_elem.extend([section_h2, section_p])
            section_img = self.soup.new_tag('img', src=f"data:image/png;base64,{self.generate_image(section.title)}")
            section_elem.extend([section_img])
            main.append(section_elem)
        print(str(self.soup))
        print('Blog Generated!')
        with open("output.html", "w") as f:
            f.write(str(self.soup))
        return str(self.soup)
    
    
generator = BlogGenerator("GROQ_KEY_HERE", "HF_API_KEY", "./index.html", "./data.pdf")
generator.generate_blog("AI Agents", "AI, Agents, RAG", "This blog post explores the use of AI agents for various tasks.")
