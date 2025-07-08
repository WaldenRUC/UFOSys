import re, os, yaml, random, datetime


class Config:
    def __init__(self, config_file_path=None, config_dict={}):
        self.yaml_loader = self._build_yaml_loader()    # init basic yaml loader
        # 加载传入的yaml配置和字典配置, 并合并作为external_config
        self.file_config = self._load_file_config(config_file_path)
        self.variable_config = config_dict
        self.external_config = self._merge_external_config()
        
        # 加载basic_config.yaml作为internal_config
        self.internal_config = self._get_internal_config()
        
        # 优先级: internal_config < file_config < variable_config
        self.final_config = self._get_final_config()
        
        self._check_final_config()
        self._set_additional_key()
        self._init_device()
        self._set_seed()
        if not self.final_config.get('disable_save', False):
            self._prepare_dir()
    
    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader
    
    def _load_file_config(self, config_file_path: str):
        file_config = dict()
        if config_file_path:
            with open(config_file_path, "r", encoding="utf-8") as f:
                file_config.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config
    
    @staticmethod
    def _update_dict(old_dict: dict, new_dict: dict):
        # Update the original update method of the dictionary:
        # If there is the same key in `old_dict` and `new_dict`, and value is of type dict, update the key in dict
        same_keys = []
        for key, value in new_dict.items():
            if key in old_dict and isinstance(value, dict):
                same_keys.append(key)
        for key in same_keys:
            old_item = old_dict[key]
            new_item = new_dict[key]
            old_item.update(new_item)
            new_dict[key] = old_item
        old_dict.update(new_dict)
        return old_dict
    
    def _merge_external_config(self):
        '''
        合并file_config与variable_config, 
        variable_config的非字典值会替换掉file_config中相同键对应的值; 
        variable_config的字典值会更新file_config中相同键对应的字典值
        '''
        external_config = dict()
        external_config = self._update_dict(external_config, self.file_config)
        external_config = self._update_dict(external_config, self.variable_config)
        return external_config
    
    def _get_internal_config(self):
        '''加载basic_config.yaml作为internal_config'''
        current_path = os.path.dirname(os.path.realpath(__file__))
        init_config_path = os.path.join(current_path, "basic_config.yaml")
        internal_config = self._load_file_config(init_config_path)
        return internal_config
    
    def _get_final_config(self):
        final_config = dict()
        final_config = self._update_dict(final_config, self.internal_config)
        final_config = self._update_dict(final_config, self.external_config)
        return final_config
    
    def _set_additional_key(self):
        '''set <dataset_path> from <dataset_name> and <data_dir>'''
        dataset_name, data_dir = self.final_config['dataset_name'], self.final_config['data_dir']
        self.final_config['dataset_path'] = os.path.join(data_dir, dataset_name)
        
    def _init_device(self):
        pass
    def _check_final_config(self):
        pass
    def _set_seed(self):
        pass
        
    def _prepare_dir(self):
        '''
        根据dataset, time和note构建输出的目录
        将final_config存储到该目录下
        '''
        save_note = self.final_config['save_note']
        save_dir = self.final_config['save_dir']
        if not save_dir.endswith('/'):
            save_dir += '/'
            
        current_time = datetime.datetime.now()
        self.final_config["save_dir"] = os.path.join(
            save_dir,
            f"{current_time.strftime('%Y_%m_%d_%H_%M')}_{self.final_config['dataset_name']}_{save_note}",
        )
        print(f'Save dir: {self.final_config["save_dir"]}')
        os.makedirs(self.final_config["save_dir"], exist_ok=True)
        config_save_path = os.path.join(self.final_config["save_dir"], "config.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(self.final_config, f, indent=4, sort_keys=False)
            
    
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config[key] = value
        
    def __getattr__(self, item):
        if "final_config" not in self.__dict__:
            raise AttributeError("'Config' object has no attribute 'final_config'")
        if item in self.final_config:
            return self.final_config[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")
    
    def __getitem__(self, item):
        return self.final_config.get(item)
    
    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config

    def __repr__(self):
        return self.final_config.__str__()