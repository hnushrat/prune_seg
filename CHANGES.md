Added missing “OUTPUT_DIR” to config file for inference.


Important changes to “vis_SemanticKITTI.py” file:
Added the path to plot map of predicted labels(numpy).
Inversion of the predicted labels was a must, as the predictions were inverse.
#### CHANGE
    raw_label = np.vectorize(learning_map_inv.__getitem__)(raw_label)
####								
This line was added right before mapping the objects from the dictionary.


### PRUNER

Add imports:
from pcseg.data import build_dataloader
import yaml
from easydict import EasyDict

Copy “spvcnn.py” from UniSeg to tools_prune/models/fusion
In the model_and_opt_loader add the following:
import yaml
from easydict import EasyDict
from .models.fusion.spvcnn.spvcnn import SPVCNN #, MinkUNet


def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))


    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)


    return config




def cfg_from_yaml_file(cfg_file, config = EasyDict()):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)
       
        # print(new_config)####


        merge_new_config(config=config, new_config=new_config)


    return config
