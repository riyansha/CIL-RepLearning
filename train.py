import argparse
import importlib
from utils.utils import *
import yaml
import logging
MODEL_DIR=None
DATA_DIR = '/data1/LibriSpeech_fscil/spk_segments'
#DATA_DIR = '/data1/nsynth/'
PROJECT='meta_sc'

def dict2namespace(dicts):
    for i in dicts:
        if isinstance(dicts[i], dict):
            dicts[i] = dict2namespace(dicts[i]) 
    ns = argparse.Namespace(**dicts)
    return ns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='nsynth-100',
                        choices=['nsynth-100', 'nsynth-200', 'nsynth-300', 'nsynth-400', 'librispeech', 'esc', 'pqrs'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-save_path', type=str, default='')
    parser.add_argument('-config', type=str, default="configs/default.yaml") 
    parser.add_argument('-debug', action='store_true')

    parser.add_argument('-lamda_proto', type=float, default=1.0)
    parser.add_argument('-way', type=int, default=5)
    parser.add_argument('-shot', type=int, default=5)
    parser.add_argument('-num_session', type=int, default=10)
    parser.add_argument('-batch_size_base', type=int, default=100)
    # about training
    parser.add_argument('-gpu', default='0')
    # print(parser.parse_args())
    args, unknown = parser.parse_known_args()
    args = parser.parse_args()
    with open(args.config, 'r') as config:
        cfg = yaml.safe_load(config) 
    cfg = cfg['train']
    cfg.update(vars(args))
    # args = argparse.Namespace(**cfg)
    args = dict2namespace(cfg)
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    with open("per_cls_"+args.dataset+".txt", "a") as result_file:
                
                    result_file.write("Class\tAccuracy\n")
                    result_file.write("\t".join([str(cls_idx) for cls_idx in range(100)]) + "\n")
    with open("per_cls_"+args.dataset+".txt", "a") as result_file:
                
                    result_file.write("Class\tAccuracy\n")
                    result_file.write("\t".join([str(cls_idx) for cls_idx in range(100)]) + "\n")
    with open("per_cls_"+args.dataset+".txt", "a") as result_file:
                
                    result_file.write("Class\tAccuracy\n")
                    result_file.write("\t".join([str(cls_idx) for cls_idx in range(50)]) + "\n")
    with open("per_cls_"+args.dataset+".txt", "a") as result_file:
                
                    result_file.write("Class\tAccuracy\n")
                    result_file.write("\t".join([str(cls_idx) for cls_idx in range(10)]) + "\n")


    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()