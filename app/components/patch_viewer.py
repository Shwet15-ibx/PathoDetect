import streamlit as st
from PIL import Image
import numpy as np

class PatchViewer:
    def __init__(self, config):
        self.config = config

    def display_patches(self, patches, predictions):
        st.write("### Patch Viewer")
        if not patches or not predictions:
            st.info("No patches or predictions to display.")
            return
        grid_size = self.config['visualization']['patches']['grid_size']
        cols = st.columns(grid_size[0])
        for idx, patch_info in enumerate(patches):
            col = cols[idx % grid_size[0]]
            with col:
                if isinstance(patch_info, dict):
                    patch = patch_info['patch']
                else:
                    patch = patch_info
                pred = predictions[idx] if predictions and idx < len(predictions) else {'cancer_prob': 0.0}
                st.image(patch, caption=f"Cancer: {pred['cancer_prob']:.2f}", use_container_width=True)
                if (idx + 1) % (grid_size[0] * grid_size[1]) == 0:
                    cols = st.columns(grid_size[0]) 