import torch
from torch.utils.data import DataLoader
from transformers import Adafactor, AdamW, BartTokenizer
from transformers.optimization import AdafactorSchedule, get_linear_schedule_with_warmup

from Classifier.bart import small_bart_model
from Classifier.dataset import SmallBartDataset
from Classifier.utils import read_json,train_one_epoch_for_smallBART, validate_for_smallBART


def small_bart_train(args,logger,trpath,depath,tagpath,save_path):
    device=args.device
    model_path=args.HFmodel
    # 日志及模型保存
    won_model_file= save_path
    logger.info("----"+str(model_path)+"----")
    # data
    train_data=read_json(trpath)
    valid_data=read_json(depath)
    # tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_path)
    # classes_map
    classes_map = read_json(tagpath)
    train_set = SmallBartDataset(args, True, train_data, tokenizer, classes_map)
    train_loader = DataLoader(train_set,
                              batch_size=args.gen_batch_size,
                              shuffle=True,
                              sampler=None,
                              pin_memory=True,
                              # num_workers=args.num_workers,
                              collate_fn=train_set.collate_fn,
                              drop_last=False)
    valid_set = SmallBartDataset(args, True, valid_data, tokenizer, classes_map)
    valid_loader = DataLoader(valid_set,
                              batch_size=1,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=0,
                              collate_fn=valid_set.collate_fn,
                              drop_last=False)
    # model
    model = small_bart_model.BartForSequenceClassification.from_pretrained(model_path,num_labels=len(classes_map))
    model.to(device)
    if args.use_Adafactor and args.use_AdafactorSchedule:
        # https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/optimizer_schedules#transformers.Adafactor
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)
    elif args.use_Adafactor and not args.use_AdafactorSchedule:
        optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False,
                              lr=args.gen_learning_rate)
        total_steps = len(train_loader) * args.gen_train_epochs
        gen_lr_warmup_steps = total_steps * args.gen_lr_warmup_ratio
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=gen_lr_warmup_steps,
                                                       num_training_steps=total_steps)
    else:
        optimizer = AdamW(model.parameters(), lr=args.gen_learning_rate, weight_decay=args.gen_weight_decay)
        # total_steps=(len(train_data)//args.gen_batch_size)*args.gen_train_epochs if len(train_data)%args.gen_batch_size==0 else (len(train_data)//args.gen_batch_size+1)*args.gen_train_epochs
        total_steps=len(train_loader) * args.gen_train_epochs
        gen_lr_warmup_steps=total_steps*args.gen_lr_warmup_ratio
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=gen_lr_warmup_steps,
                                                       num_training_steps=total_steps)
    best_f1=0.0
    for epoch in range(args.gen_train_epochs):
        train_result = train_one_epoch_for_smallBART(args=args,
                                               model=model,
                                               classes_map=classes_map,
                                               device=device,
                                               data_loader=train_loader,
                                               epoch=epoch,
                                               optimizer=optimizer,
                                               lr_scheduler=lr_scheduler)

        dev_result = validate_for_smallBART(args=args,
                                      model=model,
                                      device=device,
                                      data_loader=valid_loader,
                                      classes_map=classes_map)

        results = {
            'learning_rate': optimizer.param_groups[0]["lr"],
            'train_loss': train_result['loss'],
            'dev_loss': dev_result['loss'],
            'dev_f1': dev_result['won_micro_f1'],

        }
        if epoch % 10 == 9:
            logger.info('Training/training loss: {:.4f}'.format(train_result['loss'] / 10, epoch * len(train_loader) + epoch))
            print('Training/training loss', train_result['loss'] / 10, epoch * len(train_loader) + epoch)
            print('Dev/dev loss', dev_result['loss'] / 10, epoch * len(train_loader) + epoch)

        logger.info("=" * 100)
        logger.info(f"epoch: {epoch}")
        # 记录训练中各个指标的信息
        for key, value in results.items():
            logger.info(f"{key}: {value}")
        if results['dev_f1'] > best_f1:
            torch.save(model.state_dict(), won_model_file)
            best_f1 = results['dev_f1']
            print(f'best:{best_f1}')
            logger.info(f"best-mi-f1:{best_f1}  epoch:{epoch}")

