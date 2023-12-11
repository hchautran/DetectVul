import torch
from utils import GraphDS, get_model_and_tokenizer
from transformers import AutoModel, AutoTokenizer
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import  GATConv, GraphConv, GATv2Conv, TransformerConv, GCN2Conv , GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torchmetrics.classification.f_beta import F1Score, BinaryF1Score
from torchmetrics.classification.precision_recall import Precision, Recall, BinaryPrecision, BinaryRecall 
from torchmetrics.classification.accuracy import BinaryAccuracy, Accuracy
from torchmetrics.classification.matthews_corrcoef import MulticlassMatthewsCorrCoef, BinaryMatthewsCorrCoef
from tqdm.auto import tqdm
from accelerate import Accelerator
import torch.nn as nn
import numpy as np
import pandas as pd
import wandb
from collections import OrderedDict
import sys
args = sys.argv[1:]

accelerator = Accelerator(gradient_accumulation_steps=2)

device = torch.device(f'cuda:{int(args[0])}' if torch.cuda.is_available() else 'cpu')

class GATV2Model(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, heads=12 , num_classes=2 ):
        super(GATV2Model, self).__init__()
        torch.manual_seed(12345)
        self.acts = nn.ModuleList([])
        self.gnns = nn.ModuleList([])
        self.num_layers = num_layers
        for i in range(num_layers):
            if i == 0: 
                self.gnn.append(GATv2Conv(in_dim, hidden_dim, heads=heads, concat=False ,dropout=0.6))
            else:
                self.gnn.append(GATv2Conv(hidden_dim, hidden_dim, concat=False, heads=heads, dropout=0.6))
            
            self.acts.append(nn.ELU())
            
        
        # self.layers = Sequential(modules=modules) 
        self.out= GATv2Conv(hidden_dim, num_classes)
            


    def forward(self, x, edge_index):
        for i, layer in enumerate(self.gnns):
            x = layer(x, edge_index)
            x = self.acts[i](x)
    
        x = self.out(x, edge_index)
        return x 

class GCNModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers,  num_classes=2 ):
        super(GCNModel, self).__init__()
        torch.manual_seed(12345)
        self.acts = nn.ModuleList([])
        self.gnns = nn.ModuleList([])
        self.num_layers = num_layers
        for i in range(num_layers):
            if i == 0: 
                self.gnn.append( GCNConv(in_dim, hidden_dim, dropout=0.6))
            else: 
                self.gnn.append(GCNConv(hidden_dim, hidden_dim, dropout=0.6))
            
            self.acts.append(nn.ELU())
        
        # self.layers = Sequential(modules=modules) 
        self.out= GATv2Conv(hidden_dim, num_classes)
            


    def forward(self, x, edge_index):
        for i, layer in enumerate(self.gnns):
            x = layer(x, edge_index)
            x = self.acts[i](x)
    
        x = self.out(x, edge_index)
        return x 


class GraphTrainerConfig():
    def __init__(self, 
                 train_ds, 
                 test_ds, 
                 class_weight, 
                 epoch_num, 
                 learning_rate=1e-3, 
                 embs_learning_rate=1e-5, 
                 architecture='MiniLM',
                 num_layers=3,
                 device='cpu'
        ) -> None:
        self.learning_rate = learning_rate
        self.embs_learning_rate = embs_learning_rate 
        self.device = device
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.epoch_num = epoch_num
        self.class_weight = class_weight
        self.architecture = architecture 
        self.num_layers = num_layers


class GraphTrainer():
    def __init__(self, 
                config, 
                model, 
                embs_model, 
                tokenizer,
                masked
        ):
        self.config = config  
        self.tokenizer = tokenizer 
        self.train_ds = self.config.train_ds
        self.test_ds = self.config.test_ds
        self.device = self.config.device
        self.embs_model = embs_model
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.embs_optimizer = torch.optim.AdamW(self.embs_model.parameters(), lr=self.config.embs_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.embs_optimizer)

        class_weight = torch.FloatTensor(self.config.class_weight).to(self.device)
        self.num_classes = len(self.config.class_weight)
        print(num_classes)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
        self.masked = masked

    def run_train(self):
        self.model.train()
        self.embs_model.train()
        total_step = len(self.train_ds)
        total_loss = 0
        progress = tqdm(range(total_step))
        for i in range(total_step):  
            try:
                data = self.train_ds[i]
                if len(data.edge_index)==0:
                    # print(data)
                    progress.update(1)
                    continue
                with accelerator.accumulate(self.embs_model):
                    data =  self.train_ds.getitem_dynamic(
                        i, 
                        self.embs_model, 
                        self.tokenizer, 
                        self.device,
                        masked=self.masked
                    )
                    x = data.x.to(self.device)
                    y = data.y.to(self.device)
                    # print(x.shape)
                    edge_index  = data.edge_index.to(self.device)
                    out = self.model(x, edge_index) 
                    loss = self.criterion(out, y) 
                    total_loss +=  loss.item() 
                    accelerator.backward(loss)
                    self.optimizer.step()
                    self.embs_optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.embs_optimizer.zero_grad()
                    progress.update(1)
            except:
                progress.update(1)
                print(data)
        return total_loss/total_step
 
    def run_test(self, threshold=0.5, average='macro'):
        # self.model = self.model.to(self.device)
        self.model.eval()
        self.embs_model.eval()
        
        if self.num_classes > 2:
            f1 = F1Score(task='multiclass', num_classes=self.num_classes, average=average, threshold=threshold).to(self.device)
            precision = Precision('multiclass', num_classes=self.num_classes, average=average, threshold=threshold).to(self.device)
            recall = Recall('multiclass' ,num_classes=self.num_classes, average=average, threshold=threshold).to(self.device)
            acc = Accuracy('multiclass' ,num_classes=self.num_classes, average=average, threshold=threshold).to(self.device)
            mcc = MulticlassMatthewsCorrCoef(num_classes=self.num_classes).to(self.device)
        else:
            f1 = BinaryF1Score(threshold=threshold).to(self.device)
            precision = BinaryPrecision(threshold=threshold).to(self.device)
            recall = BinaryRecall(threshold=threshold).to(self.device)
            acc = BinaryAccuracy(threshold=threshold).to(self.device)
            mcc = BinaryMatthewsCorrCoef().to(self.device)
        
        preds = []
        targets = []
        total_loss = 0
        total_steps = len(self.test_ds)

        progress = tqdm(range(len(self.test_ds)))
        for i in range(len(self.test_ds)):  
            # try:
                data = self.test_ds[i]
                if len(data.edge_index)==0:
                    # print(data)
                    progress.update(1)
                    continue
                data = self.test_ds.getitem_dynamic(
                    i, 
                    self.embs_model, 
                    self.tokenizer, 
                    self.device,
                    # undirected=True,
                    masked=self.masked
                )
                x = data.x.to(self.device)
                y = torch.tensor(data.y).to(self.device)
                edge_index  = data.edge_index.to(self.device)
                out = self.model(x, edge_index) 
                loss = self.criterion(out, y) 
                total_loss += loss.item() 
                preds.append(out.argmax(dim=1))
                targets.append(y)
                progress.update(1)
            # except:
                # print(data)
                # progress.update(1)
        

        preds = torch.concatenate(preds).to(self.device)
        targets = torch.concatenate(targets).to(self.device)
        if average == 'none':
            print(f1(preds, targets))
            print(precision(preds, targets)) 
            print(recall(preds, targets))
            print(acc(preds, targets))
            print(mcc(preds, targets))
            return 

        return  (
            total_loss/total_steps,
            f1(preds, targets).item(), 
            precision(preds, targets).item(), 
            recall(preds, targets).item(),
            acc(preds, targets).item()
        )
    
    def fit(self, name,  log=True):

        self.model, self.embs_model, self.scheduler, self.optimizer, self.embs_optimizer = accelerator.prepare(
            self.model, self.embs_model, self.scheduler, self.optimizer, self.embs_optimizer 
        )

        if log:
            wandb.init(
                name=name,
                project=f'Hyperparam',
                entity='eddiechen372',
                reinit=True,
                config={
                    'embs_architecture': self.config.architecture,
                    'hidden_dim': self.embs_model.config.hidden_size,
                    'num_layers': self.config.num_layers,
                    'heads': 12,
                    'architecture': 'GCN'
                }
            )
        emb_num_params = sum(p.numel() for p in self.embs_model.parameters() if p.requires_grad)
        graph_num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if log: 
            wandb.log({'embs num params': emb_num_params, 'cls num params':graph_num_params}) 
        for epoch in range(1, self.config.epoch_num + 1):
            loss = self.run_train()
            eval_loss, test_f1, test_p, test_r, test_acc = self.run_test()
            print(f'loss: {loss}')
            print(f'Epoch: {epoch:03d}, Test f1: {test_f1:.4f}, Test precision: {test_p:.4f}, Test recall: {test_r:.4f}, Test acc: {test_acc:.4f}, ')

            if log: 
                wandb.log({
                    'epoch':epoch, 
                    'test f1':test_f1, 
                    'test precision':test_p, 
                    'test recall':test_r, 
                    'test acc':test_acc, 
                    'train loss':loss,
                    'eval loss': eval_loss
                })
        
    def evaluate(self, average):
        self.model, self.embs_model, self.scheduler, self.optimizer, self.embs_optimizer = accelerator.prepare(
            self.model, self.embs_model, self.scheduler, self.optimizer, self.embs_optimizer 
        )
        emb_num_params = sum(p.numel() for p in self.embs_model.parameters() if p.requires_grad)
        graph_num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('embs num params:', emb_num_params) 
        print('graph num params:', graph_num_params) 
        self.run_test(average=average)
        # print(f'Test precision: {test_p:.4f}, Test recall: {test_r:.4f}, Test f1: {test_f1:.4f}, Test acc: {test_acc:.4f}')

        
    def save(self, name):
        self.embs_model.save_pretrained(name)
        self.tokenizer.save_pretrained(name)
        torch.save(self.model, name +'/graph_model.pt')

    def load(self, name):
        from transformers import AutoModel, AutoTokenizer
        self.embs_model = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = torch.load( name +'/graph_model.pt')
        

if __name__ == '__main__':


    accelerator.state.device = device
    device = accelerator.device

    data_path = '/data/thesis/data'
    train_ds = GraphDS(torch.load(f'{data_path}/train_full_cve.pth'))
    test_ds = GraphDS(torch.load(f'{data_path}/test_full_cve.pth'))
    print(train_ds.graphs.iloc[0])
    print(train_ds[0])
    print(train_ds.class_weight)
    num_classes = len(train_ds.class_weight)
    


    # model_path = f'{name}'
    # embs_model = AutoModel.from_pretrained(f'{model_path}')
    # model = torch.load(f'{model_path}/graph_model.pt')
    for architecture in ['MiniLM', 'mpnet' ]:
        model_ckt = 'sentence-transformers/all-mpnet-base-v2' if architecture == 'mpnet'  else 'sentence-transformers/all-MiniLM-L6-v2'
        embs_model, tokenizer = get_model_and_tokenizer(model_ckt)
        for num_layers in [3,6,9,12]:
        # for num_layers in [3]:
            name = f'{model_ckt.split("/")[-1]}-GCN-L{num_layers}-{"w" if args[1]=="masked" else "wo"}-masked'
            print(name)
            model = GCNModel(
                in_dim=embs_model.config.hidden_size,
                hidden_dim=embs_model.config.hidden_size*2, 
                num_classes=num_classes,
                num_layers=num_layers,
            )
            config = GraphTrainerConfig(
                train_ds=train_ds,
                test_ds=test_ds,
                epoch_num=50,
                learning_rate=1e-5,
                embs_learning_rate=1e-5,
                class_weight=train_ds.class_weight,
                device=device,
                num_layers=num_layers,
                architecture=architecture
            )
            trainer = GraphTrainer(
                config=config,
                model=model,
                embs_model=embs_model,
                tokenizer=tokenizer,
                masked=True
            )

            trainer.fit(name=name, log=True)
            model_path = data_path + '/models' 
            trainer.save(model_path + '/' + name)
            trainer.load(model_path + '/' + name)
            trainer.evaluate(average='none')
