import streamlit as st
from html_generator import BlogGenerator
import base64
import PyPDF2
from io import BytesIO

def display_html_preview(html_content):
    st.components.v1.html(html_content, height=800, scrolling=True)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}">Download {file_label}</a>'
    return href

def extract_pdf_text(pdf_bytes):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    st.set_page_config(page_title="Blog Generator", layout="wide")

    st.title("🚀 Smart Blog Generator")
    st.write("Transform your ideas into engaging blog posts with AI")

    generator = BlogGenerator()

    with st.sidebar:
        st.header("📝 Blog Settings")
        blog_title = st.text_input("Blog Title", "My Blog Post")
        tags = st.text_input("Tags (comma-separated)", "AI, Blog, Technology")
        description = st.text_area("Short Description", "Write a brief description of your blog here.")

        uploaded_file = st.file_uploader("Upload Supporting Document (Required)", type=['pdf'])
        generate_button = st.button("🎨 Generate Blog")

    if generate_button and uploaded_file:
        with st.spinner("Creating your blog..."):
            # Extract text from PDF
            pdf_content = extract_pdf_text(uploaded_file.read())
            
            # Generate blog content
            html_content = generator.generate_blog(
                title=blog_title,
                tags=tags,
                description=description,
                supporting_content=pdf_content
            )

            # Save generated HTML
            output_path = "generated_blog.html"
            with open(output_path, "w", encoding='utf-8') as f:
                f.write(html_content)

            # Display preview
            st.subheader("📄 Preview")
            display_html_preview(html_content)

            # Download options
            st.markdown("### 📥 Download Options")
            st.markdown(get_binary_file_downloader_html(output_path, 'blog.html'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
