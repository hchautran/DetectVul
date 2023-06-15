

if __name__ == '__main__':
    import os
    from datasets import load_dataset
    data_path = os.path.abspath(os.getcwd())
    from huggingface_hub import Repository
    ds_names = []
    load_dataset('EddieChen372/CVEFixes_Python_with_norm_vul_lines')
    load_dataset('EddieChen372/Vudenc_with_norm_vul_lines')
    load_dataset('EddieChen372/devign_with_norm_vul_lines')
    repo = Repository(local_dir=f"{data_path}/DetectBERT", clone_from="EddieChen372/DetectBERT")