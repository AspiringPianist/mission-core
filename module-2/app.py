import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import tempfile
from pathlib import Path
from typing import List, Dict
import os
import re

class HTMLEnhancer:
    @staticmethod
    def enhance_text(html_content: str) -> str:
        enhancements = [
            (r'"([^"]*)"', r'<span class="quote">\1</span>'),
            (r'!\s*([^.!?]*[.!?])', r'<div class="highlight">\1</div>'),
            (r'\b(Note|Important|Key Point):\s*([^.!?]*[.!?])', r'<div class="note"><strong>\1:</strong> \2</div>'),
            (r'\bExample:\s*([^.!?]*[.!?])', r'<div class="example">\1</div>'),
            (r'\b([A-Z][A-Za-z\s]{2,}:)\s', r'<strong class="topic">\1</strong> '),
        ]
        
        for pattern, replacement in enhancements:
            html_content = re.sub(pattern, replacement, html_content)
        return html_content

class BlogGenerator:
    def __init__(self):
        self.model = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=7000
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.html_enhancer = HTMLEnhancer()

    CONTENT_GUIDELINES = """
    Format your response using proper HTML only:
    - Wrap each paragraph in <p> tags
    - Use <div class="highlight"> for key insights
    - Use <div class="example"> for examples
    - Use <div class="note"> for important notes
    - Use <ul> or <ol> for lists with <li> items
    - Use <blockquote> for quotes
    - Use <strong> for emphasis
    - Use <code> for technical terms
    
    Do not use any markdown, asterisks, or other formatting.
    Ensure all content is properly wrapped in HTML tags.
    """

    SECTION_PROMPT = """
    Write a detailed section about: {title}
    Using this context: {context}
    Style: {style}

    Requirements:
    1. Start with a clear introduction
    2. Include at least one example in a dedicated div
    3. Add one key insight in a highlight div
    4. Include relevant quotes if applicable
    5. End with a clear conclusion
    6. Use proper HTML formatting for all content

    {guidelines}

    Return only properly formatted HTML content.
    """

    STYLES = {
    "Formal": {
        "font-family": "'Merriweather', serif",
        "color-scheme": {
            "primary": "#2b6cb0",
            "secondary": "#2c5282",
            "highlight": "#ebf8ff",
            "highlight-border": "#4299e1",
            "example": "#f0fff4",
            "example-border": "#9ae6b4",
            "note": "#fff5f5",
            "note-border": "#feb2b2"
        },
        "heading-style": "classic",
        "spacing": "comfortable",
        "border-radius": "8px"
    },
    "Casual": {
        "font-family": "'Inter', sans-serif",
        "color-scheme": {
            "primary": "#38a169",
            "secondary": "#2f855a",
            "highlight": "#f0fff4",
            "highlight-border": "#68d391",
            "example": "#ebf8ff",
            "example-border": "#4299e1",
            "note": "#faf5ff",
            "note-border": "#b794f4"
        },
        "heading-style": "modern",
        "spacing": "relaxed",
        "border-radius": "12px"
    },
    "Storytelling": {
        "font-family": "'Lora', serif",
        "color-scheme": {
            "primary": "#805ad5",
            "secondary": "#6b46c1",
            "highlight": "#faf5ff",
            "highlight-border": "#9f7aea",
            "example": "#fff5f7",
            "example-border": "#f687b3",
            "note": "#fffff0",
            "note-border": "#f6e05e"
        },
        "heading-style": "elegant",
        "spacing": "airy",
        "border-radius": "16px"
    }
}


    def _get_vectorstore(self, chunks: List[str]) -> Chroma:
        temp_dir = tempfile.mkdtemp()
        return Chroma.from_texts(
            chunks,
            self.embeddings,
            persist_directory=str(Path(temp_dir) / "chroma_db")
        )

    def extract_structure(self, text: str) -> Dict:
        prompt = PromptTemplate(
            template="""Analyze this content and provide:
            1. An engaging title
            2. 4-5 logical sections
            
            Content preview: {text}
            
            Return strictly in format:
            TITLE: [title]
            SECTIONS:
            - [section1]
            - [section2]
            - [section3]
            - [section4]""",
            input_variables=["text"]
        )
        
        response = self.model.invoke(prompt.format(text=text[:3000]))
        lines = response.content.strip().split('\n')
        
        return {
            "title": lines[0].replace('TITLE:', '').strip(),
            "sections": [l.replace('-', '').strip() for l in lines[2:] if l.strip().startswith('-')]
        }

    def generate_section(self, title: str, context: str, style: str) -> str:
        prompt = PromptTemplate(
            template=self.SECTION_PROMPT,
            input_variables=["title", "context", "style", "guidelines"]
        )
        
        response = self.model.invoke(
            prompt.format(
                title=title,
                context=context,
                style=style,
                guidelines=self.CONTENT_GUIDELINES
            )
        )
        
        # Enhance the HTML content
        return self.html_enhancer.enhance_text(response.content)

    def create_blog(self, pdf_text: str, style: str) -> str:
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_text(pdf_text)
        
        retriever = self._get_vectorstore(chunks).as_retriever()
        structure = self.extract_structure(pdf_text)
        
        # Generate enhanced content
        intro = self.html_enhancer.enhance_text(
            self.model.invoke(
                f"Write an engaging HTML-formatted introduction about {structure['title']}"
            ).content
        )

        sections = ""
        for section_title in structure["sections"]:
            context = "\n".join(
                doc.page_content 
                for doc in retriever.get_relevant_documents(section_title)
            )
            content = self.generate_section(section_title, context, style)
            sections += f'<section><h2>{section_title}</h2>{content}</section>'

        conclusion = self.html_enhancer.enhance_text(
            self.model.invoke(
                f"Write a HTML-formatted conclusion summarizing {structure['title']}"
            ).content
        )

        # Calculate reading time
        word_count = len(f"{intro} {sections} {conclusion}".split())
        reading_time = max(1, round(word_count / 200))

        return self.get_styled_html(
            structure["title"],
            reading_time,
            intro,
            sections,
            conclusion,
            style
        )

    def get_styled_html(self, title, reading_time, intro, sections, conclusion, style) -> str:
        style_vars = self.STYLES[style]
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Merriweather:wght@400;700&family=Lora:wght@400;600&display=swap" rel="stylesheet">
            <style>
                {self.get_css(style_vars)}
            </style>
        </head>
        <body>
            <article class="blog-post">
                <header class="blog-header">
                    <h1>{title}</h1>
                    <div class="meta">{reading_time} min read</div>
                </header>
                <div class="intro">{intro}</div>
                {sections}
                <div class="conclusion">{conclusion}</div>
            </article>
        </body>
        </html>
        """

    @staticmethod
    def get_css(style_vars: Dict) -> str:
      return f"""
      :root {{
          --font-family: {style_vars['font-family']};
          --heading-style: {style_vars['heading-style']};
      }}
      
      body {{
          font-family: var(--font-family);
          line-height: 1.8;
          color: #2d3748;
          background: #f7fafc;
          max-width: 900px;
          margin: 0 auto;
          padding: 2rem;
      }}
      
      .blog-post {{
          background: white;
          padding: 3rem;
          border-radius: 12px;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      }}
      
      .blog-header {{
          text-align: center;
          margin-bottom: 3rem;
      }}
      
      h1 {{
          font-size: 2.5rem;
          margin-bottom: 1rem;
          line-height: 1.3;
          color: #2b6cb0;
      }}
      
      h2 {{
          font-size: 1.8rem;
          margin: 2rem 0 1.5rem;
          padding-bottom: 0.5rem;
          border-bottom: 2px solid #e2e8f0;
          color: #2c5282;
      }}
      
      .meta {{
          color: #666;
          font-size: 0.9rem;
          display: flex;
          justify-content: center;
          gap: 1rem;
      }}
      
      p {{
          margin-bottom: 1.2rem;
          font-size: 1.1rem;
      }}
      
      .highlight {{
          background: #ebf8ff;
          border-left: 4px solid #4299e1;
          padding: 1.5rem;
          margin: 1.5rem 0;
          border-radius: 0 8px 8px 0;
      }}
      
      .example {{
          background: #f0fff4;
          border: 1px solid #9ae6b4;
          padding: 1.5rem;
          margin: 1.5rem 0;
          border-radius: 8px;
      }}
      
      .note {{
          background: #fff5f5;
          border: 1px solid #feb2b2;
          padding: 1.5rem;
          margin: 1.5rem 0;
          border-radius: 8px;
      }}
      
      blockquote {{
          font-style: italic;
          border-left: 4px solid #cbd5e0;
          padding-left: 1rem;
          margin: 1.5rem 0;
          color: #4a5568;
      }}
      
      .quote {{
          font-style: italic;
          color: #4a5568;
          position: relative;
          padding: 0 10px;
      }}
      
      .quote::before, .quote::after {{
          content: '"';
          color: #718096;
      }}
      
      .topic {{
          color: #2b6cb0;
          font-weight: 600;
      }}
      
      code {{
          background: #edf2f7;
          padding: 0.2rem 0.4rem;
          border-radius: 4px;
          font-family: 'Fira Code', monospace;
          font-size: 0.9em;
      }}
      
      ul, ol {{
          margin: 1.5rem 0;
          padding-left: 1.5rem;
      }}
      
      li {{
          margin-bottom: 0.5rem;
      }}
      
      .conclusion {{
          margin-top: 3rem;
          padding-top: 2rem;
          border-top: 2px solid #e2e8f0;
      }}
      
      @media (max-width: 768px) {{
          body {{
              padding: 1rem;
          }}
          
          .blog-post {{
              padding: 1.5rem;
          }}
          
          h1 {{
              font-size: 2rem;
          }}
      }}
      """


def main():
    st.title("Enhanced Blog Generator")
    
    if uploaded_file := st.file_uploader("Upload PDF", type=["pdf"]):
        style = st.radio("Blog Style", ["Formal", "Casual", "Storytelling"])
        
        if st.button("Generate Blog"):
            with st.spinner("Creating your blog..."):
                pdf_text = "".join(
                    page.extract_text() 
                    for page in PdfReader(uploaded_file).pages
                )
                
                blog_html = BlogGenerator().create_blog(pdf_text, style)
                
                st.components.v1.html(blog_html, height=800, scrolling=True)
                st.code(blog_html, language="html")

if __name__ == "__main__":
    main()
