import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import exr_util

class EXRViewer:
    def __init__(self, dir_a, dir_b, max_idx=31):
        self.dir_a = dir_a
        self.dir_b = dir_b
        self.idx = 0
        self.max_idx = max_idx
        self.scale_factor = 1.0
        
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.canvas.manager.set_window_title('EXR Pixel Comparator')

        plt.subplots_adjust(bottom=0.25)
        ax_scale = plt.axes([0.20, 0.12, 0.60, 0.03])
        self.slider_scale = Slider(
            ax=ax_scale,
            label='Exposure (Scale) ',
            valmin=0.01,
            valmax=5.0,
            valinit=1.0,
            valstep=0.01
        )
        self.slider_scale.on_changed(self.on_scale_change)

        ax_slider = plt.axes([0.20, 0.05, 0.60, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label='Image Index ',
            valmin=0,
            valmax=self.max_idx,
            valinit=self.idx,
            valstep=1
        )
        self.slider.on_changed(self.on_slider_change)
        
        self.update_plot()
        plt.show()

    def load_image(self, path):
        if not os.path.exists(path):
            return None
        
        img = exr_util.load_image(path)
        if img is None:
            return None
        
        img_display = np.clip(img * self.scale_factor, 0, 1)
        return img_display

    def update_plot(self):
        filename_a = f"ambient_slice_{self.idx}.exr"
        filename_b = f"pred_ambient_slice_{self.idx}.exr"
        
        path_a = os.path.join(self.dir_a, filename_a)
        path_b = os.path.join(self.dir_b, filename_b)
        
        img_a = self.load_image(path_a)
        img_b = self.load_image(path_b)
        
        self.axes[0].clear()
        self.axes[1].clear()
        
        kwargs = {'interpolation': 'nearest', 'aspect': 'equal'}
        
        # Draw A
        if img_a is not None:
            self.axes[0].imshow(img_a, **kwargs)
            self.axes[0].set_title(f"A: {filename_a}\n({img_a.shape[0]}x{img_a.shape[1]} Ground Truth)", fontsize=10)
        else:
            self.axes[0].text(0.5, 0.5, "File Not Found", ha='center', va='center')
            self.axes[0].set_title(f"A: Missing")

        # Draw B
        if img_b is not None:
            self.axes[1].imshow(img_b, **kwargs)
            self.axes[1].set_title(f"B: {filename_b}\n({img_b.shape[0]}x{img_b.shape[1]} Prediction)", fontsize=10)
        else:
            self.axes[1].text(0.5, 0.5, "File Not Found", ha='center', va='center')
            self.axes[1].set_title(f"B: Missing")

        for ax in self.axes:
            ax.axis('off')

        self.fig.suptitle(f"Index: {self.idx} / {self.max_idx}", fontsize=14, fontweight='bold')
        self.fig.canvas.draw()

    def on_scale_change(self, val):
        self.scale_factor = val
        self.update_plot()
    
    def on_slider_change(self, val):
        self.idx = int(val)
        self.update_plot()

if __name__ == "__main__":
    level_name = 'ThirdPersonExampleMap'
    current_dir = Path(__file__).resolve().parent
    folder_A = os.path.join(current_dir, f'data/{level_name}/textures')
    folder_B = os.path.join(current_dir, f'data/{level_name}/model_checkpoints/eval')
    viewer = EXRViewer(folder_A, folder_B, max_idx=39)