import argparse
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
from datetime import datetime


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',type=str, default='binary',choices=['binary','none'])
    parser.add_argument('--prompt',type=str, default='hard',choices=['hard'])
    parser.add_argument('--opt',type=str, default='pareto',choices=['pareto','none'])
    parser.add_argument('--HFmodel',type=str,default='bart-base',choices=['bart-large','bart-base'])
    parser.add_argument('--dataset',type=str,default='tbd',choices=['tbd','matres'])
    parser.add_argument('--version', type=str, default='tim-tr',choices=['tim-tr','tjm-tr','tim-te','tjm-te','ftest'])
    parser.add_argument('--model1',type=str,default='1') #Temporal Judgement Module (TJM)
    parser.add_argument('--model2',type=str,default='1') #Temporal Inference Module (TIM)
    parser.add_argument('--n_tokens',type=int,default=16) #tbd--16,matres--18
    parser.add_argument('--max_len', type=int, default=200) #200
    #classifer
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_file",type=str,default="Checkpoint/")
    parser.add_argument('--prefix_text',type=str,default="Sentence:")
    parser.add_argument('--prompt_1',type=str,default="Event1:")
    parser.add_argument('--prompt_2',type=str,default="Event2:")
    parser.add_argument('--gen_batch_size', type=int, default=32)
    parser.add_argument("--gen_learning_rate", type=float, default=3e-5)
    parser.add_argument("--gen_lr_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gen_train_epochs", type=int, default=2)
    parser.add_argument("--gen_weight_decay", type=float, default=0.01)
    parser.add_argument('--use_Adafactor', type=bool, default=True)
    parser.add_argument('--use_AdafactorSchedule', type=bool, default=False)


    args = parser.parse_args()
    args.log_name = '{}_{}_{}_{}.log'.format(args.dataset, args.mode,args.version,
                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    return args,logger
