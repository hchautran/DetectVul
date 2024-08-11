from transformers import (
    BertConfig, 
    BertPreTrainedModel, 
    BertModel
)
from torchmetrics.classification.precision_recall import Precision, Recall, BinaryPrecision, BinaryRecall 
from torchmetrics.classification.accuracy import BinaryAccuracy, Accuracy
from torchmetrics.classification.f_beta import F1Score, BinaryF1Score 
from torchmetrics.classification.matthews_corrcoef import MulticlassMatthewsCorrCoef, BinaryMatthewsCorrCoef
from torchmetrics.classification.auroc import MulticlassAUROC, BinaryAUROC
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, RobertaModel , BertForSequenceClassification
from sklearn.utils import compute_class_weight
from typing import Optional, Tuple, Union, List
from utils import get_model_and_tokenizer
from datasets import Dataset, ClassLabel
# from normalizer.python import Normalizer
from torch_geometric.data import Data
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch.nn as nn
import pandas as pd
from utils import *
import numpy as np
import torch
import wandb
import json

accelerator = Accelerator(gradient_accumulation_steps=4)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
accelerator.state.device = device

class BertForLineClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
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

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.type_embeddings = nn.Embedding(13, config.hidden_size, padding_idx=0)
        
        

        self.post_init()

    def forward(
        self,
        inputs_embeds:torch.Tensor,
        type_ids:torch.Tensor=None,
        labels:torch.Tensor=None,
        class_weight=None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        return_embs: Optional[bool] = None
    ): 

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if type_ids is not None:
            inputs_embeds = inputs_embeds + self.type_embeddings(type_ids)
            

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None and class_weight is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight, reduce='mean')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        if not return_embs:
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return dict(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            input_embebs=inputs_embeds
        )



class BertForFuncClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds:torch.Tensor,
        labels:torch.Tensor=None,
        class_weight=None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ): 

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.last_hidden_state[:, -1]
        # print(outputs)
        pooled_output = self.dropout(pooled_output)
        # print(pooled_output.shape)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None and class_weight is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight, reduce='mean')
            loss = loss_fct(logits.view(-2, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DetectBERT(nn.Module):

    def __init__(self, embs_model_ckt:str, num_labels:int, num_hidden_layers=6, max_lines=1024, heads=12 ,func_cls=False):
        super(DetectBERT, self).__init__()
        self.embs_model, self.tokenizer = get_model_and_tokenizer(embs_model_ckt)
        self.embs_model = self.embs_model.to(device)
        config = BertConfig(
            vocab_size=1,
            num_attention_heads=heads,
            hidden_size=self.embs_model.config.hidden_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_lines,
            num_labels=num_labels,
            
        )
        self.func_cls = func_cls
        self.model = BertForLineClassification(config=config).to(device)


    def cls_pooling(self, model_output):
      return model_output.last_hidden_state[:, 0]

    def forward(
        self, 
        lines, 
        labels:torch.LongTensor=None, 
        class_weight:torch.FloatTensor=None, 
        masked=True, 
        return_embs=None
        ):
        inputs = self.tokenizer(
            lines, 
            padding=True, 
            return_tensors='pt',
            truncation=True,
            max_length=128
        )
        if class_weight is not None and labels is not None:
            class_weight = class_weight.to(device)
            labels = labels.to(device)

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        if masked:
          attention_mask = torch.where(input_ids < self.tokenizer.vocab_size, attention_mask, 0).to(device)
        

        embs_out = self.embs_model(input_ids=input_ids, attention_mask=attention_mask) 
        line_embeddings = self.cls_pooling(embs_out).unsqueeze(0)

        return self.model( 
            inputs_embeds=line_embeddings.to(device),
            labels=labels,
            class_weight=class_weight,
            return_embs=return_embs
        )

    def save_pretrained(self, name):
        self.tokenzier.save_pretrained(f'{name}/')
        self.embs_model.save_pretrained(f'{name}/embs')
        self.model.save_pretrained(f'{name}/classifier')
        
    def from_pretrained(self, name):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f'{name}/embs')
        self.embs_model = AutoModel.from_pretrained(f'{name}/embs')
        self.model = BertForLineClassification.from_pretrained(f'{name}/classifier')
        
class DetectBERTTrainerConfig():
    def __init__(self, ds, epoch_num, model_ckt, architecture ,learning_rate=1e-5, masked=True ,device='cpu', func_cls=False, save_each=10, num_layers=6, heads=12) -> None:
        self.learning_rate = learning_rate
        self.architecture=architecture 
        self.device = device
        self.ds = ds
        self.epoch_num = epoch_num
        y = np.concatenate(self.ds['train'][:]['label']) 
        self.class_weight = torch.FloatTensor(compute_class_weight('balanced', y=y, classes=np.unique(y)))
        self.num_classes = len(self.class_weight) 
        print(self.class_weight)
        self.model = DetectBERT(model_ckt, num_labels=self.num_classes, func_cls=func_cls, num_hidden_layers=num_layers, heads=heads)
        self.func_cls = func_cls
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.masked = masked
        self.save_each = save_each 
        self.num_layers= num_layers
        self.heads=heads


class DetectBERTTrainer():
    def __init__(self, 
                config, 
                log=True
        ):
        self.config = config  
        self.train_ds = self.config.ds['train']
        self.test_ds = self.config.ds['test']
        self.valid_ds= self.config.ds['test']
        self.device = self.config.device
        self.model = self.config.model
        self.optimizer = self.config.optimizer 
        self.scheduler = self.config.scheduler 
        self.class_weight = self.config.class_weight
        self.num_classes = self.config.num_classes 
        self.masked = self.config.masked
        self.func_cls = self.config.func_cls
        self.log = log

    def run_train(self):
        self.model.train()
        total_step = len(self.train_ds)
        total_loss = 0
        for i in tqdm(range(total_step)):  
            try:
                self.optimizer.zero_grad()
                sample = self.train_ds[i]
                labels = torch.LongTensor(sample['label']) 
                # print(sample['lines'])
                # print(sample['label'])
                with accelerator.accumulate(self.model):
                    out = self.model(sample['lines'], labels=labels, class_weight=self.class_weight, masked=self.masked)
                    if self.log:
                       wandb.log({'current loss': out.loss.item()})
                    total_loss += out.loss.item() 
                    accelerator.backward(out.loss)
                    self.optimizer.step()
            except:
                print(len(sample['lines']))
        return total_loss/total_step
 
    def run_test(self, threshold=0.5, average='macro', eval_valid=False):
        print(device)
        self.model = self.model.to(device)
        self.model.eval()
        if self.num_classes > 2:
            f1 = F1Score(task='multiclass', num_classes=self.num_classes, average=average, threshold=threshold).to(self.device)
            precision = Precision('multiclass', num_classes=self.num_classes, average=average, threshold=threshold).to(self.device)
            recall = Recall('multiclass' ,num_classes=self.num_classes, average=average, threshold=threshold).to(self.device)
            acc = Accuracy('multiclass' ,num_classes=self.num_classes, average=average, threshold=threshold).to(self.device)
            aucroc = MulticlassAUROC(num_classes=self.num_classes, average=average).to(device)
            mcc = MulticlassMatthewsCorrCoef(num_classes=self.num_classes).to(self.device)
        else:
            f1 = BinaryF1Score(threshold=threshold).to(self.device)
            precision = BinaryPrecision(threshold=threshold).to(self.device)
            recall = BinaryRecall(threshold=threshold).to(self.device)
            acc = BinaryAccuracy(threshold=threshold).to(self.device)
            aucroc = BinaryAUROC().to(self.device)
            mcc = BinaryMatthewsCorrCoef().to(self.device)

        preds = []
        preds_logits = []
        targets = []
        statement_types = []
        statement_type = {
            "Assign'": [[],[],[]],
            "AugAssign'": [[],[],[]],
            "Condition": [[],[],[]], 
            "Return'": [[],[],[]],
            "Docstring": [[],[],[]],
            "ImportFrom'": [[],[],[]],
            "FunctionDef'": [[],[],[]],
            "For": [[],[],[]],
            "Expr'": [[],[],[]],
            "Assert'": [[],[],[]],
            "Import'": [[],[],[]],
            "AnnAssign'": [[],[],[]],
            "AsyncFunctionDef'": [[],[],[]],
        } 
        total_loss = 0
        ds = self.test_ds if eval_valid else self.valid_ds

        with torch.no_grad():
            for i in tqdm(range(len(ds))):  
                    sample = ds[i]
                    labels = torch.LongTensor(sample['label']).to(device)
                    out = self.model(sample['lines'], labels ,class_weight=self.class_weight.to(device))
                    preds.append(out.logits[0].argmax(dim=1))
                    statement_types.extend(sample['type'])
                    preds_logits.append(out.logits[0])
                    targets.append(labels)

    
        preds = torch.cat(preds).to(self.device) if not self.config.func_cls else torch.tensor(preds).to(self.device)
        targets = torch.cat(targets).to(self.device) if not self.config.func_cls else torch.tensor(targets).to(self.device)
        softmax = torch.nn.Softmax(dim = 1)
        preds_logits = softmax(torch.cat(preds_logits))

        print(len(statement_types))
        print(preds.shape)
        print(targets.shape)
        
        for i, statement in enumerate(statement_types): 
            statement_type[statement[:len(statement)]][0].append(preds[i])
            statement_type[statement[:len(statement)]][1].append(targets[i])
            statement_type[statement[:len(statement)]][2].append(preds_logits[i].unsqueeze(dim=0))



        for key in statement_type:

            try:
                p = torch.tensor(statement_type[key][0]).to(device)
                t = torch.tensor(statement_type[key][1]).to(device)
                logits = torch.cat(statement_type[key][2],0).to(device)
                print(f'{key} & {len(statement_type[key][0])} & {f1(p, t)*100:2f} & {mcc(p, t):4b} & {aucroc(logits, t)*100:2f}\\\\')
            except:
                print(key)


        if average == 'none':
            print(f1(preds, targets))
            print(precision(preds, targets)) 
            print(recall(preds, targets))
            print(aucroc(preds_logits.to(device), targets))
            print(mcc(preds, targets))
            return preds.tolist(), targets.tolist(), statement_type   

        return  (
            total_loss/len(ds),
            f1(preds, targets).item(), 
            precision(preds, targets).item(), 
            recall(preds, targets).item(),
            acc(preds, targets).item(),
        )

    
    def fit(self,data_path, name ):
        self.model, self.scheduler, self.optimizer = accelerator.prepare(
            self.model, self.scheduler, self.optimizer 
        )

        if self.log:
            wandb.init(
                name=name,
                project=f'DetectBERT',
                entity='eddiechen372',
                reinit=True,
                config={
                    'embs_architecture': self.config.architecture,
                    'hidden_dim': self.model.embs_model.config.hidden_size,
                    'num_layers': self.config.num_layers,
                    'heads': self.config.heads,
                    'architecture': 'bert'
                }
            )
        emb_num_params = sum(p.numel() for p in self.model.embs_model.parameters() if p.requires_grad)
        cls_num_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        if self.log: 
            wandb.log({'embs num params': emb_num_params, 'cls num params':cls_num_params}) 
        for epoch in range(1, self.config.epoch_num + 1):
            accelerator.free_memory()
            train_loss = self.run_train()
            eval_loss, test_f1, test_p, test_r, test_acc = self.run_test()
            self.scheduler.step(eval_loss)
            print(f'train loss: {train_loss}', f'eval loss: {eval_loss}')
            print(f'Epoch: {epoch:03d}, test precision: {test_p:.4f}, test recall: {test_r:.4f}, Test f1: {test_f1:.4f} , Test acc: {test_acc:.4f}')
            if epoch % self.config.save_each == 0:
                print('eval')
                self.save(f'{data_path}/{name}_epoch_{epoch}')
                self.push_to_hub(data_path,f'{name}_epoch_{epoch}')
                self.evaluate()

            if self.log: 
                wandb.log({
                    'epoch': epoch, 
                    'test f1': test_f1, 
                    'test precision': test_p, 
                    'test recall': test_r, 
                    'test acc': test_acc, 
                    'train loss': train_loss,
                    'eval loss': eval_loss
                })
        
    def evaluate(self, average='none'):
        self.model, self.scheduler, self.optimizer = accelerator.prepare(
            self.model, self.scheduler, self.optimizer
        )
        emb_num_params = sum(p.numel() for p in self.model.embs_model.parameters() if p.requires_grad)
        cls_num_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        print('embs num params:', emb_num_params) 
        print('cls num params:', cls_num_params) 
        self.run_test(average=average, eval_valid=True)

    def push_to_hub(self, path, name):
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=path+'/'+name,
            path_in_repo=f"models/{name}",
            repo_id="EddieChen372/DetectBERT",
        )
         
    def save(self, name):
        self.model.embs_model.save_pretrained(f'{name}/embs')
        self.model.tokenizer.save_pretrained(f'{name}/embs')
        self.model.model.save_pretrained(f'{name}/classifier')

    def load(self, name):
        from transformers import AutoModel, AutoTokenizer
        self.model.embs_model = AutoModel.from_pretrained(f'{name}/embs')
        self.model.tokenizer = AutoTokenizer.from_pretrained(f'{name}/embs')
        self.model.model = BertForLineClassification.from_pretrained(f'{name}/classifier')
        
        
def get_vudenc_data(mode):
    import os
    data_path = f'{os.getcwd()}/data'
    with open(f"{data_path}/plain_{mode}", 'r') as infile:
        data = json.load(infile)
    files = {} 

    for r in data:
        # print(r)
        for c in data[r]:
            if "files" in data[r][c]:                      
                for f in data[r][c]["files"]:
                    if r+f not in files:
                        files[r+f] = []
                    if not "source" in data[r][c]["files"][f]:
                        continue
                    if "source" in data[r][c]["files"][f]:
                        current_file = data[r][c]["files"][f]
                        files[r+f].append(current_file)
    return files

def get_code_after(code_before, diffs):
    code_after = code_before

    for diff in diffs:
        lines = diff['diff'].split('\n')
        after_lines = []
        before_lines = []
        for line in lines:
            if line == '' or 'No newline at end of file' in line: continue
            if line[0] == '-':
                before_lines.append(line[1:])
            elif line[0] == '+':
                after_lines.append(line[1:])
            else:  
                before_lines.append(line[1:])
                after_lines.append(line[1:])
        
        after_diff = '\n'.join(after_lines)
        before_diff = '\n'.join(before_lines)
        code_after = code_after.replace(before_diff, after_diff)
        
    return code_after

def get_diff_line(src_before:str, src_after:str ,diffs: list):
    src_lines_before = src_before.split('\n')
    src_lines_after = src_after.split('\n')
    deleted = dict(
        code=[],
        line_no=[]
    )
    added = dict(
        code=[],
        line_no=[]
    )
    
    for diff in diffs:
        for bad_part in diff['badparts']:
            deleted['code'].append(bad_part)
            deleted['lineno'].append(src_lines_before.index(bad_part))
            
        for good_part in diff['goodparts']:
            added['code'].append(good_part)
            added['lineno'].append(src_lines_after.index(good_part))
        
    return deleted, added

def check_similar(str1, str2): 
    str1 = str1.replace(' ','')
    str2 = str2.replace(' ','')
    return str1 == str2

def get_vul_lines(diffs, src):
    lines = list(enumerate(src.split('\n'), 1))
    diff_line_no = [] 
    diff_code = []

    for changes in diffs:
        bad_codes = []
        for bad_code in changes['badparts']:
            is_vulnerable = True 
            for good_code in changes['goodparts']:
                if 'def' in bad_code or bad_code.strip() == '' or check_similar(bad_code, good_code): 
                    is_vulnerable = False 
                    break
            if is_vulnerable:
                bad_codes.append(bad_code)

        for bad_code in bad_codes:
            for lineno, code in lines: 
                if check_similar(bad_code, code): 
                    diff_line_no.append(lineno)
                    diff_code.append(code)
                    break
    return diff_line_no, diff_code

def get_vul_df(all_files):
    df = pd.DataFrame(columns=['url','code_before', 'vulnerable_lines'])
    for file in all_files:
        commits = all_files[file]
        if len(commits) == 1:
            diffs = commits[0]['changes']
            code_before = commits[0]['sourceWithComments']
            # code_after = get_code_after(code_before=commits[0]['sourceWithComments'], diffs=diffs)
            # code_before = remove_invalid_syntax(commits[0]['sourceWithComments'])
            # code_after = remove_invalid_syntax(code_after)
            line_no, code = get_vul_lines(diffs, code_before) 
            vul_lines = {
                'line_no': line_no,
                'code': code
            }
            new_record = pd.DataFrame([{
                'url': file,
                'code_before': code_before,
                # 'code_after': code_after,
                'vulnerable_lines': vul_lines,
            }])
            df = pd.concat([df , new_record]) 
    return df


if __name__ == '__main__': 
    import torch
    import os 
    import pandas as pd
    from utils import *
    from datasets import load_dataset


    ds = load_dataset('EddieChen372/CVEFixes_Python_with_norm_vul_lines')
    ds = ds.map(cast_type)
    print(ds)
    # l
    print(ds['train'][2])
    
    # # print(ds['train']['type'])
    model_ckt = 'sentence-transformers/all-mpnet-base-v2'

    config = DetectBERTTrainerConfig(
        ds=ds,
        model_ckt=model_ckt,
        num_layers=3,
        epoch_num=100,
        learning_rate=1e-5,
        device=device,
        masked=True,
        save_each=25,
        architecture='mpnet',
        heads=12,
        func_cls=False
    ) 
    trainer = DetectBERTTrainer(config=config, log=True)
    trainer.fit(f"{os.getcwd()}/data", 'test')
    # trainer.load('/data/thesis/data/models/mpnet_cvefixes_w_masked')
    # path = '/data/thesis/data/models'
    
   
    # print(trainer.push_to_hub(path, 'mpnet_cvefixes_w_masked'))
    

    trainer.evaluate()
    