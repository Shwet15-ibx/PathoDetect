import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

class HeatmapOverlay:
    def __init__(self, config):
        self.config = config

    def create_heatmap(self, image, predictions, patch_size):
        # Assume predictions is a list of dicts with 'x', 'y', 'cancer_prob'
        heatmap = np.zeros((image.height, image.width))
        for pred in predictions:
            x = pred.get('x', None)
            y = pred.get('y', None)
            prob = pred['cancer_prob']
            if x is not None and y is not None:
                heatmap[y:y+patch_size, x:x+patch_size] += prob
            # else: skip or could place at (0,0) if you want
        # Normalize
        heatmap = np.clip(heatmap, 0, 1)
        # Overlay
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(heatmap, cmap=self.config['visualization']['heatmap']['colormap'], alpha=self.config['visualization']['heatmap']['alpha'])
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf) 