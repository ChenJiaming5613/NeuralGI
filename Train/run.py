import os
import shutil
from pathlib import Path
from train_dataset_maker import make_train_dataset
from config import Config
import train
import eval
import exr_compare
import save_model

def run_level(level_name: str):
    level_name = 'Room'
    current_dir = Path(__file__).resolve().parent
    json_path = os.path.join(current_dir, f'data/VLM_{level_name}.json')
    if not os.path.exists(json_path):
        raise f"Error: Config file not found at {json_path}"
    
    copied_json_path = os.path.join(current_dir, f'data/{level_name}/VLM_{level_name}.json')
    os.makedirs(os.path.join(current_dir, f'data/{level_name}'), exist_ok=True)
    shutil.copy(json_path, copied_json_path)
    output_dir = os.path.join(current_dir, 'data', level_name)
    make_train_dataset(copied_json_path, output_dir, scale_factor=1.0, save_exr=True)

    config = Config(level_name, copied_json_path)
    train.main(config)
    eval.main(config)

    output_file = os.path.join(config.save_dir, f"{level_name}_VLM-MLP.bin")
    save_model.save_model(config, output_file)

    # folder_A = os.path.join(current_dir, f'data/{level_name}/textures')
    # folder_B = os.path.join(current_dir, f'data/{level_name}/model_checkpoints/eval')
    # viewer = exr_compare.EXRViewer(folder_A, folder_B, max_idx=31)

if __name__ == '__main__':
    level_names = [
        'Room',
        'Sponza',
        'ThirdPersonExampleMap'
    ]

    for level_name in level_names:
        run_level(level_name)

