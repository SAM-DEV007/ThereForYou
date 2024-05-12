from pathlib import Path

import yaml


if __name__ == '__main__':
    model_data_path = Path(__file__).resolve().parent.parent / 'Model_Data/'
    dataset_path = Path(__file__).resolve().parent / 'Processed/'

    config = {
        'path': str(dataset_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {0: 'rope'}
    }

    with open(str(model_data_path / 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)