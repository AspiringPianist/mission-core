################################################# IMPORTS #################################################################################
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64, requests, os, time
from pydantic import BaseModel, Field
from typing import List
################################################# Pydantic Models for Structured Output ####################################################

class BlogTitles(BaseModel):
    titles: List[str] = Field(description="List of 4 section titles for the blog", max_items=4, min_items=4)

class SectionContent(BaseModel):
    content: str = Field(description="Detailed content for the section")

class Section(BaseModel):
    title: str = Field(description="Section title")
    content: str = Field(description="Section content")

################################################# Blog Generator Class #################################################################################

class BlogGenerator:
    def __init__(self, template_path="index.html"):
        """
        Initialize ChatGroq as we have seen earlier via langchain_groq,
        Use Embeddings for HuggingFace just like before,
        Use Beautiful Soup to parse the HTML file so we can access it by HTML syntax
        Initialize the HuggingFace API Endpoint for calling Stable Diffusion for Image Generation
        """
        self.llm = ChatGroq(
            api_key="GROQ_API_KEY",
            model="llama-3.3-70b-specdec"
        )
        print('Initialized ChatGroq...')
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print('Initialized HuggingFaceEmbeddings')
        
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), template_path)
        with open(template_path, 'r', encoding='utf-8') as file:
            self.template = file.read()
        self.soup = BeautifulSoup(self.template, 'html.parser')
        
        self.hf_api_key = "HUGGING_FACE_TOKEN"
        self.hf_api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

    def process_pdf(self, content):
        """
        As discussed in RAG Theory, first split the document into chunks. Then use the embedding function
        from langchain_huggingface (allMiniLML6v2) to convert each chunk into embeddings (list of numbers)
        where each number in the list has some semantic or relation with the language/meaning.
        """
        print('Processing PDF...')
        texts = RecursiveCharacterTextSplitter(
            chunk_size=1000,                    # each chunk is 1000 characters in length
            chunk_overlap=200                   # mix in 200 chars from previous chunk in the current chunk to keep relevance
        ).split_text(content)
        
        return Chroma.from_texts(               # returns a vectorstore     
            texts=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

    def generate_sections(self, vectorstore):
        """
        First use RAG to find out what the entire PDF is about and the different topics present in it.
        Parse the titles/topics with the structured output we made using Pydantic
        Then for each title:
            Use RAG with the title as query to get relevant content and rephrase/analyze using the LLMs
            to generate `content` for each title
        Finally combine them into a Section object with both title and content.

        This method reduces the token usage from LLMs by splitting the work into multiple (5) prompts instead of a single prompt.
        """
        context = " ".join([doc.page_content for doc in vectorstore.similarity_search(
            "Extract main topics for blog sections",
            k=3
        )])
        
        # First prompt: Generate titles
        titles_llm = self.llm.with_structured_output(BlogTitles)
        titles_llm = self.llm.with_structured_output(BlogTitles)
        titles_response = titles_llm.invoke(
            f"""Based on this context: {context}
            Generate exactly 4 section titles for a technical blog post.
            Return only the titles in a list format.
            Make them clear and professional."""
        )
        
        print('Generated titles:', titles_response.titles)
        
        sections = []
        # Second-fifth prompts: Generate content for each title
        for title in titles_response.titles:
            section_context = " ".join([doc.page_content for doc in vectorstore.similarity_search(
                f"Find information relevant to: {title}",
                k=3
            )])
            
            content_llm = self.llm.with_structured_output(SectionContent)
            content_response = content_llm.invoke(
                f"""Context: {section_context}
                Title: {title}
                Task: Write detailed content for this section.
                Requirements:
                - Professional tone
                - Clear explanation
                - Relevant to the title
                - 1-2 paragraphs"""
            )
            print(f'Generated content for: {title}')
            
            sections.append(Section(title=title, content=content_response.content))
        
        return sections

    def generate_image(self, prompt):
        """
        Generating images via the HuggingFace endpoint for Flux1.Schnell
        In the return statement we use base64 to encode it
        """
        print('Generating image...')
        response = requests.post(
            self.hf_api_url,
            headers={"Authorization": f"Bearer {self.hf_api_key}"},
            json={
                "inputs": prompt,
                "parameters": {"width": 768, "height": 512}
            }
        )
        print(response)
        return base64.b64encode(response.content).decode()

    def generate_blog(self, title, tags, description, supporting_content):
        """
        We are using beautiful soup to inject the generated content into the respective HTML Positions
        """
        print('Generating blog...')
        
        # Update header
        header = self.soup.find('header')
        h1_tag = self.soup.new_tag('h1')
        h1_tag.string = title
        p_tag = self.soup.new_tag('p')
        p_tag.string = f"Tags: {tags}"
        header.clear()
        header.extend([h1_tag, p_tag])

        # Process content
        vectorstore = self.process_pdf(supporting_content)
        sections = self.generate_sections(vectorstore)
        
        # Update main content
        main = self.soup.find('main')
        main.clear()

        # Add introduction
        intro_section = self.soup.new_tag('section', id='introduction')
        intro_h2 = self.soup.new_tag('h2')
        intro_h2.string = "Introduction"
        intro_p = self.soup.new_tag('p')
        intro_p.string = description
        intro_section.extend([intro_h2, intro_p])
        main.append(intro_section)

        # Generate and add sections
        for i, section in enumerate(sections, 1):
            section_elem = self.soup.new_tag('section', id=f'section{i}')
            print('Adding Section: ', section)
            
            h2_tag = self.soup.new_tag('h2')
            h2_tag.string = section.title
            section_elem.append(h2_tag)
            
            p_tag = self.soup.new_tag('p')
            p_tag.string = section.content
            section_elem.append(p_tag)
            
            img_tag = self.soup.new_tag('img', src=f"data:image/png;base64,{self.generate_image(section.title)}")
            section_elem.append(img_tag)
            
            main.append(section_elem)

        return str(self.soup.prettify())