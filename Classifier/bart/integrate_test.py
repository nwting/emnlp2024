from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartConfig

from Classifier.bart import bart_model, small_bart_model
from Classifier.dataset import SmallBartDataset, BartDataset
from Classifier.utils import read_json, test_for_integrate


def integrate_test(args,logger,tepath,tagpath,save_path,model1path,model2path):
    device = args.device
    model_path = args.HFmodel
    logger.info("----test----")
    test_data=read_json(tepath)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    classes_map = read_json(tagpath)
    test_set1 = BartDataset(args, False,test_data, tokenizer, classes_map)
    test_set2 = SmallBartDataset(args, True, test_data, tokenizer, classes_map)
    test_loader1 = DataLoader(test_set1,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=0,
                              collate_fn=test_set1.collate_fn,
                              drop_last=False)
    test_loader2 = DataLoader(test_set2,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=0,
                              collate_fn=test_set2.collate_fn,
                              drop_last=False)
    # model, load weights
    myconfig = BartConfig.from_pretrained(model_path)
    myconfig.num_labels=len(classes_map)

    model_file1 = model1path
    model1 = bart_model.BartForSequenceClassification.from_pretrained(model_file1, config=myconfig)
    model1.to(device)
    mlist=[]

    model_file2 = model2path
    model2 = small_bart_model.BartForSequenceClassification.from_pretrained(model_file2, config=myconfig)
    model2.to(device)


    test_result = test_for_integrate(model1=model1,model2=model2,args=args, device=device, data_loader1=test_loader1,data_loader2=test_loader2,classes_map=classes_map,save_path=save_path)

    # print(test_result)
    results = {
        'test_macro_f1': test_result['macro_f1'],
        'test_micro_f1': test_result['micro_f1'],
        'test_weighted_f1': test_result['weighted_f1']
    }
    # 记录训练中各个指标的信息
    for key, value in results.items():
        logger.info(f"{key}: {value}")