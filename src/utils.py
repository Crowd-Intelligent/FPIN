import yaml
import os

def parse_yaml(file_path):
    with open(file_path, 'r') as f:
        # judge the '_base_' key
        res = yaml.safe_load(f)
        if '_base_' in res:
            for sub_file_path in res['_base_']:
                # get the absolute path
                # sub_file_path = file_path.replace(file_path.split('/')[-1], sub_file_path)

                sub_file_path = os.path.join(os.path.dirname(file_path), sub_file_path)

                res = {**parse_yaml(sub_file_path), **res}
            # delete the '_base_' key
            del res['_base_']
        return res
