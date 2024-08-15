from DetectBERT import DetectBERT 
from builder import CFGBuilder 
from cfg import CFG
from utils import *
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn


vul_dict = { 
    'CWE-22': 'path_disclosure',
    'CWE-77': 'command_injection',
    'CWE-79': 'xss',
    'CWE-89': 'sql',
    'CWE-352': 'xsrf',
    'CWE-601': 'open_redirect',
    'CWE-94': 'remote_code_execution',
}

class DetectBERTPythonPipeline():
    def __init__(self, model:DetectBERT, source ) -> None:
        builder = CFGBuilder(src=source, separate=True, max_statement_len=1)
        self.model = model
        self.cfg = builder.build_from_src(src=source, name='global')
        self.name_dict = get_user_defined_names(source)
        self.funcs = self._get_funcs()
        

    def _get_funcs(self)-> None: 
        funcs = [] 
        queue = [self.cfg]
        while(len(queue) > 0):
            current_graph = queue.pop()
            for graph in current_graph.functioncfgs:
                queue.append(current_graph.functioncfgs[graph])

            prop = self._get_funcs_statements(
                current_graph, 
                name_dict=self.name_dict,
            )
            
            if prop is not None:
                funcs.append(prop)
                
        return pd.DataFrame(funcs)

    def _get_funcs_statements(self, graph, name_dict):
        contents=[]
        normalized_contents = [] 
        node_map = {}
        edges = [] 
        lines = []

        for node in graph:
            if node.id not in node_map:
                contents.append(node.get_source())
                normalized_contents.append(node.get_normalized_source(name_dict))
                current_lines = node.get_lineno()
                lines.append(current_lines)
                node_map[node.id] = len(node_map)
        
                
        for node in graph:
            source_id = node_map[node.id]
            for link in node.exits:
                next_id = node_map[link.target.id]
                edges.append([source_id, next_id])

        return {
            'content': contents, 
            'normalized_content': normalized_contents,
            'lines': lines,
            'data': Data(
                edge_index=torch.LongTensor(edges).t().contiguous(), 
            ),
        }

    def predict(self):
        self.model.model.eval()
        self.model.embs_model.eval()
        out = {} 
            
        
        
        with torch.no_grad():
            for i in range(len(self.funcs)):
                vul_lines = []
                confident = []
                label = []
                func = self.funcs.iloc[i]
                output = self.model(func['normalized_content'])
                logits = output.logits[0]
                softmax = nn.Softmax(dim=1) 
                normalized_logits = softmax(logits)
                preds = normalized_logits.argmax(dim=1)
                attentions =  output.attentions
                for i in range(len(preds)):
                    if preds[i] != 0:
                         
                        print('label:', preds[i].item())
                        print('confident:', normalized_logits[i][preds[i]].item())
                        print(func['lines'][i], func['content'][i])
                        for line in func['lines'][i]:
                            out[line] = {
                                'confident': normalized_logits[i][preds[i]].item(),
                                'label': preds[i].item(), 
                                'attentions': attentions,
                                'content': func['content']
                            }

                        vul_lines.append(func['lines'][i])
                        confident.append(normalized_logits[i][preds[i]].item())
                        label.append(preds[i].item())
                        div()
                

        return out

            
def read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return ''.join(lines)


if __name__ == '__main__':
    import os
    model_path = f'{os.getcwd()}/data/models/mpnet_cvefixes_w_masked'
    embs_model_arch = f'sentence-transformers/all-MiniLM-L12-v2'
    model = DetectBERT(embs_model_arch, num_labels=len(vul_dict) + 1) 
    model.from_pretrained(model_path)
    source_path = f'/home/jupyter-iec_chau/DetectBERT/examples/sql-2'
    source = read_file(source_path)
    print_with_lines(source)
    pipeline = DetectBERTPythonPipeline(model, source)
    # print(pipeline.funcs[['content','lines']])
    out = pipeline.predict()
    print(out)
        

    
    
  


