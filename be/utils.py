import keyword
import torch 
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, PreTrainedModel, BertModel
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import AutoTokenizer
import ast
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import ast
from builder import CFGBuilder 
from typing import Any 
from ast import *
import astor


def findComments(sourcecode):
  commentareas = []
  
  inacomment = False
  commentstart = -1
  commentend = -1
  
  
  for pos in range(len(sourcecode)):
    if sourcecode[pos] == "#":
      if not inacomment:
        commentstart = pos 
        inacomment = True
    
    if sourcecode[pos] == "\n":
      if inacomment:
        commentend = pos
        inacomment = False
    
    if commentstart >= 0 and commentend >= 0:
      t = [commentstart, commentend]
      commentareas.append(t)
      commentstart = -1
      commentend = -1


  return commentareas


def get_dataset_stat(dataset):
    labels = {'0':0,'1':0}
    print(f'dataset length: {len(dataset)}')
    for X, y in dataset:
        labels[str(y)] +=1

    for key in labels:
        print(f'number of sample with label {key}: {labels[key]}')
        print(f'percent: {labels[key]/len(dataset)}')

  
def div(title=''):
  print('='* 100, title, '='*100)

def small_div(title=''):
  print('-'* 100, title, '-'*100)


def removeDoubleSeperators(tokenlist):
    last = ""
    newtokens = []
    for token in tokenlist:
      if token == "\n":
        token = " "
      if len(token) > 0:
        if ((last == " ") and (token == " ")):
          o = 1 #noop
          #print("too many \\n.")
        else:
          newtokens.append(token)
          
        last = token
        
    return(newtokens)


def stripComments(code):
    
  withoutComments = ""
  lines = code.split("\n")
  withoutComments = ""
  therewasacomment = False
  for c in lines:
    if "#" in c:
      therewasacomment = True
      position = c.find("#")
      c = c[:position]
    withoutComments = withoutComments + c + "\n"
  
  
  change = withoutComments
  
  withoutComments = change

  return withoutComments


class VulDataset(Dataset):
  def __init__(self, allblocks: pd.DataFrame, data_split='train'):
    self.allblocks = allblocks 
    self.data_split = data_split 
    labels = allblocks['label'].tolist()

  def __len__(self):
    if self.data_split == 'train': 
      return len(self.keystrain)
    elif self.data_split == 'valid':
      return len(self.keysvalid)
    else:
      print('Invalid split')

  def __getitem__(self, idx):
    if self.data_split == 'train': 
      key_idx = self.keystrain[idx]
    elif self.data_split == 'valid':
      key_idx = self.keysvalid[idx]
    return self.allblocks.iloc[key_idx]


def get_dataset(vul:str,is_full: bool=False):
  data_path = '/root/data/thesis/data/cvefixes/raw_dataset'
  print('is full:', is_full)
  train_dataset = torch.load(f'{data_path}/train/{vul}{"_full" if is_full else ""}.pth')
  valid_dataset = torch.load(f'{data_path}/valid/{vul}{"_full" if is_full else ""}.pth')
  return train_dataset, valid_dataset
  
def get_vudenc_dataset(vul:str,is_full: bool=False):
  data_path = '/root/data/thesis/data/vudenc/raw_dataset'
  print('is full:', is_full)
  train_dataset = torch.load(f'{data_path}/train/{vul}{"_full" if is_full else ""}.pth')
  valid_dataset = torch.load(f'{data_path}/valid/{vul}{"_full" if is_full else ""}.pth')
  return train_dataset, valid_dataset

def get_suspicious_dataset(vul):
  data_path = '/root/data/thesis/data/cvefixes/raw_dataset'
  train_dataset = torch.load(f'{data_path}/train/{vul}_suspicious.pth')
  valid_dataset = torch.load(f'{data_path}/valid/{vul}_suspicious.pth')
  return train_dataset, valid_dataset

  
def to_diff_dict(str1):
  lines = str1.splitlines()
  diff_dict = {}
  added_lines = []
  deleted_lines = []
  for line in lines:
      if(line[0]=='+'):
          added_lines.append(line[1:].strip())
      if(line[0]=='-'):
          deleted_lines.append(line[1:].strip())

  diff_dict = {'added':added_lines, 'deleted':deleted_lines}
  return diff_dict


def add_token(tokenizer: PreTrainedTokenizerFast, model:PreTrainedModel=None):
    tokenizer.add_tokens([f'VAR_{i}' for i in range(300)])
    tokenizer.add_tokens([f'FUNC_{i}' for i in range(100)])
    tokenizer.add_tokens([f'CLASS_{i}' for i in range(100)])

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer 


def add_special_token(tokenizer: PreTrainedTokenizerFast, model:PreTrainedModel=None):
    tokenizer.add_special_tokens([f'VAR_{i}' for i in range(300)])
    tokenizer.add_special_tokens([f'FUNC_{i}' for i in range(100)])
    tokenizer.add_special_tokens([f'CLASS_{i}' for i in range(100)])

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer 



def get_model_and_tokenizer(ckt:str, use_special_token=False):
  from transformers import AutoTokenizer, AutoModel
  model = AutoModel.from_pretrained(ckt, output_attentions=True)
  tokenizer = AutoTokenizer.from_pretrained(ckt)
  if not use_special_token:
    model, tokenizer = add_token(tokenizer, model)
  else:
    model, tokenizer = add_special_token(tokenizer, model)
  return model, tokenizer 

def get_line(line_no, code):
   return code.split('\n')[line_no - 1]

def get_lines(line_nos, code):
   lines = code.split('\n')
   prev_line = 0
   for line_no in line_nos:
      if prev_line + 1 != line_no:
         small_div('')
      
      print(lines[line_no - 1])
      prev_line = line_no 

      

class GraphDS(Dataset):
    def __init__(self, graphs:pd.DataFrame):
        self.graphs = graphs
        self.class_weight = self._get_class_weights() 

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
            return self.graphs.iloc[index]['data']

    def mean_pooling(self, model_output, attention_mask):
      token_embeddings = model_output[0] #First element of model_output contains all token embeddings
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
      
    def cls_pooling(self, model_output):
      return model_output.last_hidden_state[:, 0]
    

    def getitem_dynamic(self, index, model, tokenizer, device='cpu', undirected=False, masked=False) -> torch.FloatTensor:
        graph = self.graphs.iloc[index]
        inputs = tokenizer(
        #   graph['features'], 
          graph['content'], 
          padding=True, 
          return_tensors='pt',
          truncation=True,
          max_length=200
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        if masked:
          attention_mask = torch.where(input_ids < tokenizer.vocab_size, attention_mask, 0).to(device)
        model = model.to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask) 
        sentence_embeddings = self.cls_pooling(out)
        if undirected: 
          edges = graph['undirected_data'].edge_index
        else:
          edges = graph['data'].edge_index

        label = graph['data'].y

        if edges.shape[0] == 0:
          edges = torch.tensor([[0],[0]]).to(device)
        
        return Data(
            x=sentence_embeddings, 
            edge_index=edges, 
            y=label
        )
    

    def build_visual(self, index):
        self.graphs.iloc[index]['graph'].build_visual(self.graphs.iloc[index]['name'], 'pdf')

    def get_contents(self, index):
        return self.graphs.iloc[index]['features']

    def _get_class_weights(self):
        from sklearn.utils import compute_class_weight
        ys = []
        for i in range(len(self.graphs)):
            ys.append(self.graphs.iloc[i]['data'].y)
        ys = torch.concatenate(ys).tolist()
        return compute_class_weight(class_weight='balanced', classes=np.unique(ys), y=ys) 
    

def parse_src(src):
    return ast.parse(src)
    
def get_doc_string_pos(src):
    doc_string_lines= [] 
    try:
        lines = src.split('\n')
        # for i, line in enumerate(lines, start=1):
            # if ('#' in line and line.strip()[0] == '#') or line.strip() == '':
                # doc_string_lines.append(i)
        root = parse_src(src)
        for node in ast.walk(root):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef ,ast.Module)):
                if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str)):
                    start = 1 if isinstance(node, ast.Module) else node.lineno
                    end =  node.body[0].value.lineno 
                    if '"""' in lines[end - 1]:
                        while '"""' not in lines[start - 1]:
                            start = start + 1
                    elif "'''" in lines[end - 1]:
                        while "'''" not in lines[start - 1]:
                            start = start + 1
                    for i in range(start, end + 1):
                        doc_string_lines.append(i)
                    # print(start, end)
                    # print('\n'.join(lines[start - 1: end ]))
    except Exception as e: 
        print(e)
    return doc_string_lines

def is_user_define_name(name:str, name_dict:dict, name_type:str):
    return name.count('__') != 2 and name != 'self'

def set_new_name(name_type:str, name_dict:str, node_name:str):
    if node_name not in name_dict[name_type]:
        name_dict[name_type][node_name] = f'{name_type}_{len(name_dict[name_type])}'  
    
def get_user_defined_names(source):
    root = parse_src(source)
    out = dict(
        VAR=dict(),
        CLASS=dict(),
        FUNC=dict()
    ) 
    for node in ast.walk(root):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_user_define_name(name=node.name, name_dict=out, name_type='FUNC'):
                set_new_name(name_type='FUNC', name_dict=out, node_name=node.name)
            for sub_node in ast.walk(node.args): 
                if isinstance(sub_node, ast.arg) and is_user_define_name(name=sub_node.arg, name_dict=out, name_type='VAR'):
                    set_new_name(name_type='VAR', name_dict=out, node_name=sub_node.arg)
        elif isinstance(node, ast.ClassDef):
            set_new_name(name_type='CLASS', name_dict=out, node_name=node.name)
        elif isinstance(node, ast.Assign):
            try:
                target = node.targets[0]
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if is_user_define_name(name=target.id, name_dict=out, name_type='VAR'):
                            set_new_name(name_type='VAR', name_dict=out, node_name=target.id)
                    elif isinstance(target, ast.Tuple):
                        for sub_node in ast.walk(target): 
                            if isinstance(sub_node, ast.Name) and is_user_define_name(name=sub_node.id, name_dict=out, name_type='VAR'):
                                set_new_name(name_type='VAR', name_dict=out, node_name=sub_node.id)
                    elif isinstance(target, ast.Subscript):
                        for sub_node in ast.walk(target): 
                            if isinstance(sub_node, ast.Name) and is_user_define_name(name=sub_node.id, name_dict=out, name_type='VAR'):
                                set_new_name(name_type='VAR', name_dict=out, node_name=sub_node.id)
                    elif isinstance(target, ast.List):
                        for sub_node in ast.walk(target): 
                            if isinstance(sub_node, ast.Name) and is_user_define_name(name=sub_node.id, name_dict=out, name_type='VAR'):
                                set_new_name(name_type='VAR', name_dict=out, node_name=sub_node.id)
            except Exception as e: 
                print(ast.dump(node))
    return out

def remove_doc_strings(source):
    root = parse_src(source)
    for node in ast.walk(root):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if not(node.body and isinstance(node.body[0], ast.Expr)):
                continue
            node.body = node.body[1:]
            if len(node.body)<1:
                node.body.append(ast.Pass())

    return astor.to_source(root) 

class normalize_user_defined_VAR(ast.NodeTransformer):
    def __init__(self,  name_dict) -> None:
        super().__init__()
        self.name_dict = name_dict 

    def visit_arg(self, node):
        if node.arg in self.name_dict['VAR']:
            return ast.arg(**{**node.__dict__, 'arg':self.name_dict['VAR'][node.arg]})
        return node 

    def visit_keyword(self, node: keyword) -> Any:
        if node.arg in self.name_dict['VAR']:
            return ast.keyword(**{**node.__dict__, 'arg':self.name_dict['VAR'][node.arg]})
        return node 
    
    def visit_Str(self, node: Str) -> Any:
        if len(node.s) >= 100:
            node.s = 'string'
        return node
            

    def visit_Name(self, node):
        if node.id in self.name_dict['VAR']:
            return ast.Name(**{**node.__dict__, 'id':self.name_dict['VAR'][node.id]})
        if node.id in self.name_dict['FUNC']:
            return ast.Name(**{**node.__dict__, 'id':self.name_dict['FUNC'][node.id]})
        if node.id in self.name_dict['CLASS']:
            return ast.Name(**{**node.__dict__, 'id':self.name_dict['CLASS'][node.id]})
        return node 

class normalize_user_defined_FUNC(ast.NodeTransformer):
    def __init__(self,  name_dict:dict) -> None:
        super().__init__()
        self.name_dict = name_dict 
         
    def visit_ClassDef(self, node: ClassDef) -> Any:
        if node.name in self.name_dict['CLASS']:
            node.name = self.name_dict['CLASS'][node.name]
        return node 

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        if node.name in self.name_dict['FUNC']:
            node.name = self.name_dict['FUNC'][node.name]
        return node
    
    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> Any:
        if node.name in self.name_dict['FUNC']:
            node.name = self.name_dict['FUNC'][node.name]
        return node 

def normalize_code_line(code, name_dict):
    import re
    if 'import ' in code or ' import ' in code:
        return code

    for key in name_dict:

        current_type = name_dict[key]
        for name in current_type:
            new_name = current_type[name]
            apperances = re.findall(r"\s*[^a-zA-Z\d_\'\".](+" + name + r")[^a-zA-Z\d_\'\"]+", code)
            new_appearances = [] 

            for appearance in apperances:
                new_appearances.append(appearance.replace(name, new_name))
            
            for i in range(len(apperances)):
                code = code.replace(apperances[i], new_appearances[i])

    return code

def normalize_code(code: str) -> str:
    tree = ast.parse(code)
    name_dict = get_user_defined_names(code)
    transformed = normalize_user_defined_VAR(name_dict=name_dict).visit(tree) 
    # transformed = normalize_user_defined_FUNC(code).visit(transformed) 
    code = astor.to_source(transformed)
    return code 

def cast_type(sample):
    statement_type = {
        "[PAD]'": 0,
        "For": 1,
        "Expr'": 2, 
        "Assign'": 3,
        "Return'": 4,
        "Assert'": 5, 
        "Import'": 6, 
        "AugAssign'":7,
        "Condition": 8,
        "Docstring": 9,
        "AnnAssign'": 10,
        "ImportFrom'": 11,
        "FunctionDef'": 12,
        "AsyncFunctionDef'": 13 
    } 
    type_ids = [] 
    for type in sample['type']:
        type_ids.append(statement_type[type])

    sample['type_ids'] = torch.LongTensor(type_ids)
    return sample
  
def remove_comment_diff(diffs:list, comment_lines):
   for diff in diffs:
      cur_line = diff[0]
      code = diff[1].strip()
      if cur_line in comment_lines or code == '' or 'import ' in code:
         diffs.remove(diff)
   return diffs

def check_valid_diff(before, after):
   before_code = before['code']
   after_code = after['code']
   if len(before_code) == len(after_code):  
      for i in range(len(before_code)):
         if before_code[i].strip() != after_code[i].strip():
            return True
      return False
   return True 
   
def normalize_comments(src:list, comment_lines):
   lines = src.split('\n')
   for i, line in enumerate(lines, 1):
      if i in comment_lines:
         lines[i - 1] = ''
   return '\n'.join(lines) 
      
def parse_diff(sample):
      diff_parsed = ast.literal_eval(sample['diff_parsed'])
      before_comment_lines = get_doc_string_pos(sample['code_before']) 
      after_comment_lines = get_doc_string_pos(sample['code_after'])
      deleted_diff = list(filter(lambda item: item[1] != '' and '#' not in item[1],  diff_parsed['deleted'])) 
      added_diff = list(filter(lambda item: item[1] != '' and '#' not in item[1],  diff_parsed['added']))
      deleted_diff = remove_comment_diff(deleted_diff, before_comment_lines)
      added_diff = remove_comment_diff(added_diff, before_comment_lines) 

      diff_parsed ={
         'deleted': deleted_diff,
         'added': added_diff
      }

      sample['deleted'] = {
         'code': [item[1] for item in diff_parsed['deleted']],
         'line_no': [item[0] for item in diff_parsed['deleted']],
      }
      sample['added'] = {
         'code': [item[1] for item in diff_parsed['added']],
         'line_no': [item[0] for item in diff_parsed['added']],
      }
      is_valid_diff = check_valid_diff(sample['deleted'] , sample['added'])

      normalized_code_before = sample['code_before']
      normalized_code_after = sample['code_after']
      
      normalized_code_before = normalize_code(normalized_code_before)
      normalized_code_after = normalize_code(normalized_code_after)
      normalized_code_before = remove_doc_strings(normalized_code_before)
      normalized_code_after = remove_doc_strings(normalized_code_after)
         
      sample['normalized_code_before'] = normalized_code_before
      sample['normalized_code_after'] = normalized_code_after
      sample['before_doc_string_pos'] = before_comment_lines 
      sample['after_doc_string_pos'] = after_comment_lines 
      sample['is_valid_diff'] = is_valid_diff
      return sample

def print_with_lines(src):
    for item in enumerate(src.split('\n'), 1):
        print(item[0], item[1])

def filename_and_lineno_to_func_def(src, lineno):
    candidate = None
    for item in ast.walk(ast.parse(src)):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if item.lineno > lineno:
                # Ignore whatever is after our line
                continue
            if candidate:
                distance = lineno - item.lineno
                if distance < (lineno - candidate.lineno):
                    candidate = item
            else:
                candidate = item

    if candidate:
        return candidate.name
    else:
        return 'global'

def filename_and_lineno_to_class_def(src, lineno):
    candidate = None
    for item in ast.walk(ast.parse(src)):
        if isinstance(item, (ast.ClassDef)):
            if item.lineno > lineno:
                # Ignore whatever is after our line
                continue
            if candidate:
                distance = lineno - item.lineno
                if distance < (lineno - candidate.lineno):
                    candidate = item
            else:
                candidate = item

    if candidate:
        return candidate.name
    else:
        return ''

def get_vul_method(sample):
    vul_methods = set() 

    for i, lineno in enumerate(sample['deleted']['line_no']):
            func_name = filename_and_lineno_to_func_def(sample['code_before'], lineno) 
            class_name = filename_and_lineno_to_class_def(sample['code_before'], lineno)
            if class_name == '':
                vul_methods.add(func_name)
            else: 
                vul_methods.add(class_name + '.' +func_name)

    for i, lineno in enumerate(sample['added']['line_no']):
            func_name = filename_and_lineno_to_func_def(sample['code_after'], lineno) 
            class_name = filename_and_lineno_to_class_def(sample['code_after'], lineno)
            if class_name == '':
                vul_methods.add(func_name)
            else: 
                vul_methods.add(class_name + '.' +func_name)
    
    sample['vul_methods'] = vul_methods
    return sample 


def get_graph_prop(graph, name_dict ,vul_lines=set() ,label=0, is_code_before=True):
    contents=[]
    normalized_contents = [] 
    ys=[]
    node_map = {}
    edges = [] 
    undirected_edges= []
    lines = []
    contain_changes = False 

    for node in graph:
        if node.id not in node_map:

            contents.append(node.get_source())
            normalized_contents.append(node.get_normalized_source(name_dict))
            current_lines = node.get_lineno()
            lines.append(current_lines)
            is_vulnerable = False 

            for line in current_lines: 
                source = node.get_source()
                if (
                    line in vul_lines and 
                    'docstring' not in source and 
                    'def ' not in source and
                    'class ' not in source 
                ):
                    is_vulnerable = True
                    break
            
            if is_vulnerable:   
                contain_changes = True

            if is_vulnerable and is_code_before:
                ys.append(label)
            else:
                ys.append(0)
            node_map[node.id] = len(node_map)
    
    if contain_changes != True and not is_code_before:
        return None 
        
            
    for node in graph:
        source_id = node_map[node.id]
        for link in node.exits:
            next_id = node_map[link.target.id]
            edges.append([source_id, next_id])

    return {
        'graph': graph,
        'name': graph.name, 
        'label': label if label in ys else 0,
        'ys': ys,
        'edges': torch.LongTensor(edges),
        'content': contents, 
        'features': normalized_contents,
        'lines': lines,
        'data': Data(
            edge_index=torch.LongTensor(edges).t().contiguous(), 
            y=torch.tensor(ys)
        ),
    }

def get_graphs_from_src(
    src: str, 
    label:int,
    vul_lines=set(),
    is_code_before=True
) -> pd.DataFrame: 
    graphs = [] 
    # print(src)
    name_dict = get_user_defined_names(src)
    cfg = (
        CFGBuilder(src=src, separate=True, max_statement_len=1)
        .build_from_src(src=src, name='global')
    )
    queue = [cfg]
    while(len(queue) > 0):
        current_graph = queue.pop()
        for graph in current_graph.functioncfgs:
            queue.append(current_graph.functioncfgs[graph])
        prop = get_graph_prop(
            current_graph, 
            vul_lines=vul_lines, 
            name_dict=name_dict,
            label=label+1,
            is_code_before=is_code_before
        )
        if prop is not None:
            graphs.append(prop)
    return pd.DataFrame(graphs)



            
