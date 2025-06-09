from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartConfig

from Classifier.bart import bart_model
from Classifier.dataset import BartDataset
from Classifier.utils import read_json, test_forBART


def bart_test(args,logger,tepath,tagpath):
    device = args.device
    model_path = args.HFmodel
    logger.info("----test----")
    test_data=read_json(tepath)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    classes_map = read_json(tagpath)
    test_set = BartDataset(args, True,test_data, tokenizer, classes_map)
    test_loader = DataLoader(test_set,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=0,
                              collate_fn=test_set.collate_fn,
                              drop_last=False)
    # model, load weights
    myconfig = BartConfig.from_pretrained(model_path)
    myconfig.num_labels=len(classes_map)
    model_file=str(f"SaveModels/{args.dataset}/vbart/won_model{args.model1}.bin")
    model = bart_model.BartForSequenceClassification.from_pretrained(model_file, config=myconfig)
    model.to(device)
    test_result = test_forBART(model=model, args=args, device=args.device, data_loader=test_loader,
                               classes_map=classes_map)
    results = {
        'test_accuracy': test_result['accuracy'],
        'test_macro_f1': test_result['macro_f1'],
        'test_micro_f1': test_result['micro_f1'],
        'test_weighted_f1': test_result['weighted_f1']
    }
    # 记录训练中各个指标的信息
    for key, value in results.items():
        logger.info(f"{key}: {value}")

    return results