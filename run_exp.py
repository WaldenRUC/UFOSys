from ufo.config import Config
from ufo.utils import get_dataset
from ufo.pipeline import UFOPipeline
import argparse



def ufo(args):
    config_dict = {
        'data_dir': args.data_dir,
        'dataset_name': args.dataset_name,
        'openai_apikey': args.openai_apikey,            
        'openai_baseurl': args.openai_baseurl,
        'openai_model': args.openai_model,
        'test_sample_num': args.sample_num,
        'random_sample': args.random_sample,
        'batch_size': args.batch_size,
        'retriever_sources': ['human', 'web', 'knowledge'],
    }
    config = Config('config.yaml', config_dict)
    
    #* 加载已经生成完毕的数据集, 包含query和response的jsonl文件
    test_data = get_dataset(config)['test']
    pipeline = UFOPipeline(config)
    output_dataset = pipeline.run(test_data)
    output_dataset.save(f"{config['save_dir']}/{args.data_dir.split('/')[-1]}_{args.dataset_name}.json")
    print(f"saved in {config['save_dir']}/{args.data_dir.split('/')[-1]}_{args.dataset_name}.json")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running experiment')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--method_name', type=str, default='ufo')
    parser.add_argument('--openai_apikey', type=str, default='EMPTY')
    parser.add_argument('--openai_baseurl', type=str, default='http://0.0.0.0:2233/v1/')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--openai_model', type=str, default='Qwen2.5-14B-Instruct')
    #* ===== 是否用小批量数据测试 =====
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--random_sample', action='store_true')
    args = parser.parse_args()
    ufo(args)