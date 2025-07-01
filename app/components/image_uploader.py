import streamlit as st
from pathlib import Path

class ImageUploader:
    def __init__(self, config):
        self.config = config
        self.supported_formats = config['data']['supported_formats']
        self.max_file_size_mb = config['data']['max_file_size_mb']

    def upload_image(self):
        st.subheader("Upload Pathology Slide")
        uploaded_file = st.file_uploader(
            "Choose a pathology slide (WSI)",
            type=[fmt.replace('.', '') for fmt in self.supported_formats],
            help=f"Supported formats: {', '.join(self.supported_formats)}. Max size: {self.max_file_size_mb} MB."
        )
        if uploaded_file is not None:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                st.error(f"File size {file_size_mb:.2f} MB exceeds the limit of {self.max_file_size_mb} MB.")
                return None
            else:
                st.success(f"Uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
                return uploaded_file
        return None 