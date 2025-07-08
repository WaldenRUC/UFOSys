import gradio as gr
import random, logging, time, json, rich
from typing import List, Dict
from ufo.config import Config
from ufo.utils import get_dataset
from ufo.pipeline import UFOPipeline
from ufo.evaluate import get_multi_source_majority, get_multi_source_seq_majority
import numpy as np
config_dict = {
    'data_dir': './',
    'dataset_name': 'test',
    'disable_save': True,
    'openai_apikey': 'EMPTY',
    'openai_baseurl': 'http://0.0.0.0:2233/v1/',
    'openai_model': 'Qwen2.5-14B-Instruct',
    'retriever_sources': ['human', 'web', 'knowledge'],
    'batch_size': 1,
}
config = Config('config.yaml', config_dict)
pipeline = UFOPipeline(config, online=True)


def display_items(items: List[Dict]) -> str:
    """将项目列表格式化为HTML显示"""
    html = "<div style='margin-top: 20px;'>"
    for _id, item in enumerate(items):
        # item_score = display_score(item['factuality'])
        if item['factuality'] == 0:
            item_score = 'False'
            item_icon = '❌'
        else:
            item_score = 'True'
            item_icon = '✅'
        
        html += f"""
        <div style='background: #f5f5f5; padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
            <h3 style='margin-top: 0;'>{item_icon} Claim {_id+1}: {item['claim']}</h3>
            <p><strong>factuality: {item_score}</strong></p>
            <p><strong>query:</strong> {item['query']}</p>
            <p><strong>source:</strong> {item['source']}</p>
            <p><strong>evidence:</strong> {item['evidence']}</p>
            <p><strong>answer:</strong> {item['answer']}</p>
            <p><strong>reasoning:</strong> {item['reasoning']}</p>
        </div>
        """
    html += "</div>"
    return html

def display_items_from_file(preds, samples_items: List[List[Dict]]) -> str:
    '''传入列表的元素为sample'''
    html = "<div style='margin-top: 20px;'>"
    for sample_id, sample in enumerate(samples_items):
        # 组标题
        html += f"""
        <div style='background: #e0e0e0; padding: 10px; margin-bottom: 15px; border-radius: 5px;'>
            <h2 style='margin-top: 0;'>🔍 Sample #{sample_id + 1} Factuality Score: {preds[sample_id]}</h2>
        """
        # 组内项目
        for item_idx, item in enumerate(sample):
            item_score = 'True' if item['factuality'] else 'False'
            item_icon = '✅' if item['factuality'] else '❌'
            
            html += f"""
            <div style='background: #f5f5f5; padding: 10px; margin: 10px 0 15px 20px; border-left: 3px solid #ccc; border-radius: 3px;'>
                <h3 style='margin-top: 0;'>{item_icon} Claim {item_idx+1}: {item['claim']}</h3>
                <p><strong>Factuality:</strong> {item_score}</p>
                <p><strong>Query:</strong> {item['query']}</p>
                <p><strong>Source:</strong> {item['source']}</p>
                <p><strong>Evidence:</strong> {item['evidence']}</p>
                <p><strong>Answer:</strong> {item['answer']}</p>
                <p><strong>Reasoning:</strong> {item['reasoning']}</p>
            </div>
            """
        
        html += "</div>"  # 关闭组容器
    return html

def factuality_evaluation_text(input_text: str, input_reference_answers: str, ordered_source_names:list=['human', 'web', 'knowledge'], is_sequential=True):
    rich.print(f'ordered_source_names: {ordered_source_names}')
    rich.print(f'is_sequential: {is_sequential}')
    system_input = [
        {
            'response': input_text,
            'reference_answers': input_reference_answers.split('\n')
        }
    ]
    print(f'system_input: {system_input}')
    output_dataset = pipeline.run(system_input)
    with open('./temp.json', 'w', encoding='utf-8') as fw:
        json.dump(output_dataset, fw, ensure_ascii=False, indent=2)
    if is_sequential:
        preds, from_sources = get_multi_source_seq_majority(
            output_dataset, ordered_source_names=ordered_source_names
        )
    else:
        preds, from_sources = get_multi_source_majority(
            output_dataset, unordered_source_names=ordered_source_names
        )
    # score_display = display_score(preds[0])
    pred = preds[0]
    from_sources = from_sources[0]
    items_display = display_items(from_sources)
    source2count = {item: 0 for item in ordered_source_names}
    total = 0
    for item in from_sources:
        source2count[item['source']] += 1
        total += 1
    for key in source2count:
        source2count[key] = source2count[key] / total
    return {'factuality': pred}, items_display, source2count

def factuality_evaluation_file(file, ordered_source_names: list=['human', 'web', 'knowledge'], is_sequential=True):
    '''each line should contain response and reference_answers'''
    system_input = []
    if file is None:
        return {'factuality': 0.0}, "Please upload file first.", {}
    with open(file.name, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    responses, reference_answers = [], []
    for i, item in enumerate(data):
        responses.append(item['response'])
        if isinstance(item['reference_answers'], str):
            item['reference_answers'] = [item['reference_answers']]
        reference_answers.append(item['reference_answers'])
    for _response, _reference in zip(responses, reference_answers):
        system_input.append({
            'response': _response,
            'reference_answers': _reference
        })
    output_dataset = pipeline.run(system_input)
    with open('./temp.json', 'w', encoding='utf-8') as fw:
        json.dump(output_dataset, fw, ensure_ascii=False, indent=2)
    
    if is_sequential:
        preds, from_sources = get_multi_source_seq_majority(
            output_dataset, ordered_source_names=ordered_source_names
        )
    else:
        preds, from_sources = get_multi_source_majority(
            output_dataset, unordered_source_names=ordered_source_names
        )
    items_display = display_items_from_file(preds, from_sources)
    source2count = {item: 0 for item in ordered_source_names}
    total = 0
    for sample in from_sources:
        for item in sample:
            source2count[item['source']] += 1
            total += 1
    for key in source2count:
        source2count[key] = source2count[key] / total
    return {'factuality': np.mean(preds)}, items_display, source2count


# 创建Gradio界面
with gr.Blocks(title="UFO Dashboard", theme=gr.themes.Soft(font=['Georgia'])) as demo:
    gr.Markdown("<div style='text-align: center;'><h1>🛸UFO Dashboard</h1></div>")
    gr.Markdown("## The UFO system categorizes and integrates three fact sources for factuality evaluation: ")
    gr.Markdown("### ✍🏻 Human-written evidence")
    gr.Markdown("### 🌐 Web search results")
    gr.Markdown("### 🤖 LLM knowledge")
    # gr.Markdown("Add some texts, and the system will output factuality score with configured multiple fact sources!")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            input_text = gr.Textbox(
                label="Evaluated Text", 
                placeholder="Add some texts for evaluation...",
                lines=3)
            # submit_text_btn = gr.Button("Evaluate the text!", variant="primary")
        with gr.Column(scale=1, min_width=300):
            reference_answers = gr.Textbox(
                label="Reference Answers (Optional)", 
                placeholder="Add some texts for evaluation...",
                lines=3)
        with gr.Column(scale=2, min_width=300):
            file_input = gr.File(
                label="Upload File",
                file_count="single",
                file_types=[".jsonl"],  # 允许的文件类型
                height=170
            )
    with gr.Row():
        is_sequential = gr.Checkbox(label='Sequential Evaluation', value=True)
    with gr.Row():
        ordered_source_names = gr.Dropdown(choices=['human', 'web', 'knowledge'], multiselect=True, label='Select Fact Sources', value=['human', 'web', 'knowledge'])
    
    with gr.Row():
        with gr.Column(scale=2):
            submit_text_btn = gr.Button("Evaluate the Text!", variant="primary")
        with gr.Column(scale=2):
            submit_file_btn = gr.Button("Evaluate the File!", variant="primary")

    
    with gr.Row():
        with gr.Column():
            score_output = gr.Label(label="Factuality Score")
                
        with gr.Column():
            source_usage = gr.Label(
                show_label=True, label='Sources for Fact Verification')
            
    with gr.Column():
        items_output = gr.HTML(label="Fact Units")
    
    submit_text_btn.click(
        fn=factuality_evaluation_text,
        inputs=[input_text, reference_answers, ordered_source_names, is_sequential],
        outputs=[score_output, items_output, source_usage]
    )
    
    submit_file_btn.click(
        fn=factuality_evaluation_file,
        inputs=[file_input, ordered_source_names, is_sequential],
        outputs=[score_output, items_output, source_usage]
    )
    
    gr.Examples(
        examples=[
            [
                "Suzhou, a renowned historical and cultural city in China, is located in the southeastern part of Jiangsu Province, near Beijing.", 
                "Suzhou is a major prefecture-level city in southeastern part of Jiangsu province, China, bordered by Zhejiang province to the south, Shanghai municipality to the east, and Anhui province to the northwest. As part of the Yangtze Delta megalopolis, it is a major economic center and focal point of trade and commerce."],
        ],
        inputs=[input_text, reference_answers],
        run_on_click=False,
        label="Have a try!"
    )

if __name__ == "__main__":
    demo.launch()