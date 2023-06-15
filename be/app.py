from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from pipeline import DetectBERTPythonPipeline 
from typing import List
import utils
from cwe_dict import vul_dict
from DetectBERT import DetectBERT
from utils import print_with_lines
import json

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
model_path = 'DetectBERT/models/mpnet_vudenc_w_masked'
embs_model_arch = 'sentence-transformers/all-MiniLM-L12-v2'
model_name = 'mpnet_vudenc_w_masked'
model = DetectBERT(embs_model_arch, num_labels=len(vul_dict) + 1) 
model.from_pretrained(model_path)

def preprocessing(src) -> List[utils.GraphDS]:
    graphs = utils.get_graphs_from_src(src=src)
    
    return graphs 

def get_prediction(src):
    pass
    

@app.route('/CWE',  methods=['GET'])
@cross_origin()
def get_vuls():
    res = jsonify(vul_dict)
    res.headers.add("Access-Control-Allow-Origin", "*")
    return res
    

@app.route('/predict', methods=['POST'])
# @cross_origin()
def predict():
    if request.method == 'POST':
 
        data = json.loads(request.data.decode())
        source = data['inputCode']
        print_with_lines(source)
        pipeline = DetectBERTPythonPipeline(model, source)
        print(pipeline.funcs[['content','lines']])
        out = pipeline.predict()
        res = []
        print(out)
        for line in out:
            confident = out[line]['confident']
            label = out[line]['label']
            
            cwe = vul_dict[label]
            cwe_name = cwe['name']
            cwe_url = cwe['url']
            cwe_id = cwe['id']
            
            res.append({
                'url': cwe_url,
                'name': cwe_name,
                'line': line,
                'confident': confident,
                'id': cwe_id
            })
        
        res = jsonify(res)
        res.headers.add("Access-Control-Allow-Origin", "*")
        return res


if __name__ == '__main__':
    app.run()