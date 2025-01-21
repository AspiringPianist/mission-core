from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64, requests, os, time
from pydantic import BaseModel, Field
from typing import List

class BlogTitles(BaseModel):
    titles: List[str] = Field(description="List of 4 section titles for the blog", max_items=4, min_items=4)

class SectionContent(BaseModel):
    content: str = Field(description="Detailed content for the section")

class Section(BaseModel):
    title: str = Field(description="Section title")
    content: str = Field(description="Section content")

class BlogGenerator:
    def __init__(self, template_path="index.html"):
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
        self.hf_api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"

    def process_pdf(self, content):
        print('Processing PDF...')
        texts = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_text(content)
        
        return Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

    def generate_sections(self, vectorstore):
        # Get context for titles
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
        print('Generating image...')
        response = requests.post(
            self.hf_api_url,
            headers={"Authorization": f"Bearer {self.hf_api_key}"},
            json={
                "inputs": prompt,
                "parameters": {"width": 768, "height": 512}
            }
        )
        return base64.b64encode(response.content).decode()

    def generate_blog(self, title, tags, description, supporting_content):
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
