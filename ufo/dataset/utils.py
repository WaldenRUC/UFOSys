from typing import Dict, Any, Union
import numpy as np
from ufo.dataset import Dataset

def convert_numpy(data: Any) -> Any:
    """将可能包含numpy的data全转换成list/int等形式
    """
    if isinstance(data, dict):
        return {key: convert_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy(element) for element in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer,)):
        return int(data)
    elif isinstance(data, (np.floating,)):
        return float(data)
    elif isinstance(data, (np.bool_)):
        return bool(data)
    elif isinstance(data, (np.str_)):
        return str(data)
    else:
        return data
    
def filter_dataset(dataset: Dataset, filter_func=None):
    '''根据filter_func移除Dataset中不符要求的元素
    '''
    if filter_func is None:
        return dataset
    data = dataset.data
    for item in data:
        if not filter_func(item):
            data.remove(item)
    return Dataset(config=dataset.config, data=data)

def remove_images(data: Any) -> Any:
    from PIL import Image
    from typing import Any
    if isinstance(data, dict):
        return {key: remove_images(value) 
                for key, value in data.items()
                if not isinstance(value, Image.Image)}
    elif isinstance(data, list):
        return [remove_images(element) 
                for element in data 
                if not isinstance(element, Image.Image)]
    elif isinstance(data, tuple):
        return tuple(remove_images(element) 
                     for element in data 
                     if not isinstance(element, Image.Image))
    elif isinstance(data, set):
        return {remove_images(element) 
                for element in data 
                if not isinstance(element, Image.Image)}
    else:
        return data