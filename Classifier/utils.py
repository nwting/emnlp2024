import json
import sys
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, BartForSequenceClassification, T5Tokenizer, BartTokenizer, \
    BartForConditionalGeneration, BartModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, hamming_loss
from torch.autograd import Variable
from Classifier.bart.min_norm_solvers import MinNormSolver, gradient_normalizers
import warnings
warnings.filterwarnings("ignore")

def read_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def remove_zero(numlist):
    l=[]
    for i in numlist:
        if i!=0:
            l.append(i)
    return l
def logit2label(logits):
    labels=[]
    logits=torch.sigmoid(logits)
    for bs in range(logits.shape[0]):
        bslabel=[1 if logits[bs,i]>0.5 else 0 for i in range(logits.shape[1])]
        labels.append(bslabel)
    return labels
def bl2ol(args,bl):
    ol=[]
    if args.dataset=='tbd':
        labelset = [[1, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0], [1, 1, 0, 1, 0], [0, 0, 0, 0, 0]]
        v=5
    elif args.dataset=='matres':
        labelset = [[1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0]]
        v=2
    for i in bl:
        try:
            ol.append(labelset.index(i))
        except:
            ol.append(v)
    return ol
def binary2origin(args,logits,labels):
    orilogits=[]
    logits = torch.sigmoid(logits)
    if args.dataset=='tbd':
        for bs in range(logits.shape[0]):
            bslogits=[]
            bslogits.append(logits[bs,0]*(1.0-logits[bs,1])*logits[bs,2])
            bslogits.append(logits[bs, 0] * (1.0 - logits[bs, 1]) * (1.0 - logits[bs, 2]))
            bslogits.append(logits[bs, 0] * logits[bs, 1] * (1.0 - logits[bs, 3])* logits[bs, 4])
            bslogits.append(logits[bs, 0] * logits[bs, 1] * (1.0 - logits[bs, 3]) * (1.0 - logits[bs, 4]))
            bslogits.append(logits[bs, 0] * logits[bs, 1] * logits[bs, 3])
            bslogits.append((1.0 - logits[bs, 0]))
            orilogits.append(bslogits)
        labelset = [[1, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0], [1, 1, 0, 1, 0], [0, 0, 0, 0, 0]]
    elif args.dataset=='matres':
        for bs in range(logits.shape[0]):
            bslogits=[]
            bslogits.append(logits[bs,0]*logits[bs,2])
            bslogits.append(logits[bs, 0]* (1.0 - logits[bs, 2]))
            bslogits.append((1.0 - logits[bs, 0])* (1.0 - logits[bs, 1]))
            bslogits.append((1.0 - logits[bs, 0])* logits[bs, 1])
            orilogits.append(bslogits)
        labelset = [[1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0]]
    orilogits=torch.tensor(orilogits)
    
    orilabels=[labelset.index(i.cpu().tolist()) for i in labels]
    return orilogits,orilabels


def train_one_epoch_forBART(args,model:BartForSequenceClassification,label2id, device, data_loader, epoch, optimizer, lr_scheduler):
    model.train()
    tokenizer=BartTokenizer.from_pretrained(args.HFmodel)
    predicted_labels=torch.LongTensor([]).to(device)
    ground_truth_labels=torch.LongTensor([]).to(device)
    sum_loss=torch.zeros(1).to(device)
    optimizer.zero_grad()
    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        p_input_ids = data['p_input_ids'].to(device)
        p_attention_mask = data['p_attention_mask'].to(device)
        labels = data['labels'].to(device)#[6,bs]
        if args.opt=='pareto':
            output = model(input_ids=input_ids, attention_mask=attention_mask,p_input_ids=p_input_ids, p_attention_mask=p_attention_mask,labels=labels,mode=args.mode)
            logits,loss_list,model_param=output.logits,output.loss_list,output.model_param
            a = logits.view(-1, logits.size(-1))
            pred_labels = torch.max(a, dim=-1).indices
            accuracy=0.00
            grads = {}
            scale = {}
            loss_data={}
            for t in range(p_input_ids.shape[0]):
                optimizer.zero_grad()
                loss=loss_list[t]
                loss_data[t]=loss.data
                loss.backward(retain_graph=True)
                grads[t]=[]
                for param in model_param[t]:
                    if param.grad is not None:
                        grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))  # 屏蔽预训练模型的权重
            gn = gradient_normalizers(grads, loss_data, 'loss+')
            for t in range(p_input_ids.shape[0]):
                for gr_i in range(len(grads[t])):
                    grads[t][gr_i] = grads[t][gr_i] / gn[t]

            # Frank-Wolfe iteration to compute scales.
            # try:
            #     sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in range(2)])
            # except:
            #     print(gn)
            #     # print(grads)
            #     return -1
            sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in range(p_input_ids.shape[0])])
            for t in range(p_input_ids.shape[0]):
                scale[t] = float(sol[t])

            optimizer.zero_grad()
            # output = model(input_ids=input_ids, attention_mask=attention_mask, p_input_ids=p_input_ids,
            #                p_attention_mask=p_attention_mask, labels=labels, mode=args.mode)
            # loss_list= output.loss_list
            for t in range(p_input_ids.shape[0]):
                if t > 0:
                    loss = loss + scale[t]*loss_list[t]
                else:
                    loss = scale[t]*loss_list[t]
        else:
            output = model(input_ids=input_ids, attention_mask=attention_mask,p_input_ids=p_input_ids, p_attention_mask=p_attention_mask,labels=labels,mode=args.mode)
            logits,loss=output.logits,output.loss
            a = logits.view(-1, logits.size(-1))
            pred_labels = torch.max(a, dim=-1).indices
            accuracy=0.00
        loss.backward()

        sum_loss += loss.detach()
        avg_loss = sum_loss.item() / (step + 1)
        predicted_labels=torch.cat((predicted_labels,pred_labels),dim=0)
        if args.mode=='binary':
            ground_truth_labels=torch.cat((ground_truth_labels,labels[-1,:]),dim=0)
            # ground_truth_labels = torch.cat((ground_truth_labels, labels), dim=0)
        else:
            ground_truth_labels = torch.cat((ground_truth_labels, labels), dim=0)
        data_loader.desc = "[train epoch {}] lr: {:.5f}, loss: {:.3f}, acc: {:.3f}".format(
            epoch, optimizer.param_groups[0]["lr"], avg_loss, accuracy
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()
    g_label = ground_truth_labels.cpu().tolist()
    p_label = predicted_labels.cpu().tolist()
    id2label = {label2id[i]: i for i in label2id}
    label_id = [label2id[i] for i in label2id]
    label_id.remove(label2id['VAGUE'])
    t_name = [id2label[i] for i in label_id]
    report = classification_report(g_label, p_label, labels=label_id, target_names=t_name,
                                   output_dict=True)
    df = pd.DataFrame(report).transpose()
    print(df)
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }
@torch.no_grad()
def validate_forBART(args,model,label2id, device, data_loader, epoch):
    model.eval()
    tokenizer = BartTokenizer.from_pretrained(args.HFmodel)
    # predicted_labels = torch.LongTensor([]).to(device)
    # ground_truth_labels = torch.LongTensor([]).to(device)
    predicted_labels = []
    ground_truth_labels = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    sum_loss = torch.zeros(1).to(device)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        p_input_ids = data['p_input_ids'].to(device)
        p_attention_mask = data['p_attention_mask'].to(device)
        labels = data['labels'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, p_input_ids=p_input_ids,
                       p_attention_mask=p_attention_mask, labels=labels,mode=args.mode)
        logits=output.logits
        loss=output.loss
        if args.mode=='binary':
            labels = labels[0, :]
        a = logits.view(-1, logits.size(-1))
        pred_labels = torch.max(a, dim=-1).indices
        predicted_labels+=pred_labels.cpu().tolist()
        ground_truth_labels+=labels.cpu().tolist()
        sum_loss+=loss.detach()
    avg_loss = sum_loss.item() / (step + 1)

    g_label=ground_truth_labels
    p_label = predicted_labels

    id2label = {label2id[i]: i for i in label2id}
    label_id = [label2id[i] for i in label2id]
    label_id.remove(label2id['VAGUE'])
    t_name = [id2label[i] for i in label_id]
    report = classification_report(g_label, p_label, labels=label_id, target_names=t_name,
                                   output_dict=True)
    df = pd.DataFrame(report).transpose()
    print(df)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    accuracy = accuracy_score(g_label, p_label)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    macro_f1 = f1_score(g_label, p_label, average='macro')
    micro_f1 = f1_score(g_label, p_label, average='micro')
    weighted_f1 = f1_score(g_label, p_label, average='weighted')

    data_loader.desc = "[valid epoch {}] acc: {:.3f}, macro_f1: {:.3f}, micro_f1: {:.3f}, weighted_f1: {:.3f}".format(
        epoch, accuracy, macro_f1, micro_f1, weighted_f1
    )

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'loss':avg_loss,
        'won_macro_f1': report['macro avg']['f1-score'],
        'won_micro_f1': report['micro avg']['f1-score'],
        'won_weighted_f1': report['weighted avg']['f1-score']
    }
@torch.no_grad()
def test_forBART(model, args,device, data_loader, classes_map):
    model.eval()
    predicted_labels = []
    ground_truth_labels = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    reverse_classes_map = {classes_map[i]: i for i in classes_map}
    predict_logits=[]
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        p_input_ids = data['p_input_ids'].to(device)
        p_attention_mask = data['p_attention_mask'].to(device)
        labels = data['labels'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, p_input_ids=p_input_ids,
                       p_attention_mask=p_attention_mask, labels=labels,mode=args.mode)
        logits = output.logits
        if args.mode == 'binary':
            labels = labels[0, :]
        a = logits.view(-1, logits.size(-1))
        b = labels.view(-1)
        pred_labels = torch.max(a, dim=-1).indices
        predicted_labels += pred_labels.cpu().tolist()
        ground_truth_labels += labels.cpu().tolist()
        predict_logits.append(a.cpu().tolist()[0])
    g_label = ground_truth_labels
    p_label = predicted_labels
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    accuracy = accuracy_score(g_label, p_label)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    macro_f1 = f1_score(g_label, p_label, average='macro')
    micro_f1 = f1_score(g_label, p_label, average='micro')
    weighted_f1 = f1_score(g_label, p_label, average='weighted')

    data_loader.desc = "[test] acc: {:.3f}, macro_f1: {:.3f}, micro_f1: {:.3f}, weighted_f1: {:.3f}".format(
        accuracy, macro_f1, micro_f1, weighted_f1
    )
    # 将分类结果写入文件中
    # if save_path:
    label_id=[classes_map[i] for i in classes_map]
    label_id.remove(classes_map['VAGUE'])
    t_name = [reverse_classes_map[i] for i in label_id]
    report = classification_report(g_label, p_label,labels=label_id, target_names=t_name,
                                output_dict=True)
    micro_f1=report['micro avg']['f1-score']
    macro_f1=report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    df = pd.DataFrame(report).transpose()
    # df.to_csv(save_path, index=True)
    print(df)
    print(confusion_matrix(g_label, p_label))
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'prelogits':predict_logits
    }

def train_one_epoch_for_smallBART(args,model:BartForSequenceClassification,classes_map, device, data_loader, epoch, optimizer, lr_scheduler):
    model.train()
    tokenizer=BartTokenizer.from_pretrained(args.HFmodel)
    predicted_labels=[]
    ground_truth_labels= []
    sum_loss=torch.zeros(1).to(device)
    optimizer.zero_grad()
    data_loader = tqdm(data_loader)
    reverse_classes_map = {classes_map[i]: i for i in classes_map}
    y_test=[]
    pred=[]
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device) #[6,bs]
        output = model(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
        logits,loss=output.logits,output.loss
        y_true=logit2label(logits)
        y_pred=labels.cpu().tolist()
        hmloss=hamming_loss(y_true, y_pred)
        loss=loss+hmloss
        loss.backward()

        sum_loss += loss.detach()
        avg_loss = sum_loss.item() / (step + 1)
        accuracy=0
        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

        pred+=logit2label(logits)
        y_test+=labels.cpu().tolist()

        new_logits, new_labels = binary2origin(args, logits, labels)
        a = new_logits.view(-1, new_logits.size(-1))
        b = new_labels
        pred_labels = torch.max(a, dim=-1).indices
        predicted_labels += pred_labels.cpu().tolist()
        ground_truth_labels += b
    if args.dataset=='tbd':
        print(classification_report(y_test, pred, target_names=['p1','p2','p3','p4','p5']))
    elif args.dataset=='matres':
        print(classification_report(y_test, pred))
    g_label = ground_truth_labels
    p_label = predicted_labels
    accuracy = accuracy_score(g_label, p_label)
    # 将分类结果写入文件中
    label_id = [classes_map[i] for i in classes_map]
    label_id.remove(classes_map['VAGUE'])
    t_name = [reverse_classes_map[i] for i in label_id]
    report = classification_report(g_label, p_label, labels=label_id, target_names=t_name,
                                   output_dict=True)
    df = pd.DataFrame(report).transpose()
    print(df)

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

@torch.no_grad()
def validate_for_smallBART(args,model, device, data_loader,classes_map):
    model.eval()
    predicted_labels = []
    ground_truth_labels = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    reverse_classes_map = {classes_map[i]: i for i in classes_map}
    sum_loss = torch.zeros(1).to(device)
    y_test=[]
    pred=[]
    for step, data in enumerate(data_loader):
        # temp = tokenizer.decode(data['input_ids'][0])
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)  # [6,bs]
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits, loss = output.logits, output.loss
        pred+=logit2label(logits)
        y_test+=labels.cpu().tolist()

        new_logits, new_labels = binary2origin(args, logits, labels)
        a = new_logits.view(-1, new_logits.size(-1))
        b = new_labels
        pred_labels = torch.max(a, dim=-1).indices
        predicted_labels += pred_labels.cpu().tolist()
        ground_truth_labels += b

        sum_loss += loss.detach()
    avg_loss = sum_loss.item() / (step + 1)
    if args.dataset=='tbd':
        print(classification_report(y_test, pred, target_names=['p1','p2','p3','p4','p5']))
    elif args.dataset=='matres':
        print(classification_report(y_test, pred, target_names=['p1','p2','p3']))
    g_label = ground_truth_labels
    p_label = predicted_labels
    hard_label=bl2ol(args=args,bl=pred)
    # 将分类结果写入文件中
    label_id = [classes_map[i] for i in classes_map]
    label_id.remove(classes_map['VAGUE'])
    t_name = [reverse_classes_map[i] for i in label_id]
    print("reason_label:")
    report1 = classification_report(g_label, p_label, labels=label_id, target_names=t_name,
                                   output_dict=True)
    df = pd.DataFrame(report1).transpose()
    print(df)
    print("hard_label:")
    report2 = classification_report(g_label, hard_label, labels=label_id, target_names=t_name,
                                   output_dict=True)
    df = pd.DataFrame(report2).transpose()
    print(df)
    return {
        'loss':avg_loss,
        'won_micro_f1': report1['micro avg']['f1-score']
    }

@torch.no_grad()
def test_for_smallBART(args,model, device, data_loader,classes_map):
    model.eval()
    predicted_labels = []
    ground_truth_labels = []
    predict_logits=[]
    data_loader = tqdm(data_loader, file=sys.stdout)
    y_test=[]
    pred=[]
    reverse_classes_map = {classes_map[i]: i for i in classes_map}
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)  # [6,bs]
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits, loss = output.logits, output.loss
        pred+=logit2label(logits)
        y_test+=labels.cpu().tolist()

        new_logits, new_labels = binary2origin(args, logits, labels)
        a = new_logits.view(-1, new_logits.size(-1))
        b = new_labels
        pred_labels = torch.max(a, dim=-1).indices
        predicted_labels += pred_labels.cpu().tolist()
        ground_truth_labels += b
        predict_logits.append(a.cpu().tolist()[0])
    if args.dataset=='tbd':
        print(classification_report(y_test, pred, target_names=['p1','p2','p3','p4','p5']))
    elif args.dataset=='matres':
        print(classification_report(y_test, pred, target_names=['p1','p2','p3']))
    hard_label=bl2ol(args=args,bl=pred)
    g_label = ground_truth_labels
    p_label = predicted_labels
    accuracy = accuracy_score(g_label, p_label)
    macro_f1 = f1_score(g_label, p_label, average='macro')
    micro_f1 = f1_score(g_label, p_label, average='micro')
    weighted_f1 = f1_score(g_label, p_label, average='weighted')
    # 将分类结果写入文件中
    label_id = [classes_map[i] for i in classes_map]
    label_id.remove(classes_map['VAGUE'])
    t_name = [reverse_classes_map[i] for i in label_id]
    print("reason_label:")
    report = classification_report(g_label, p_label, labels=label_id, target_names=t_name,
                                   output_dict=True)
    df = pd.DataFrame(report).transpose()
    print(df)
    print("hard_label:")
    report = classification_report(g_label, hard_label, labels=label_id, target_names=t_name,
                                   output_dict=True)
    df = pd.DataFrame(report).transpose()
    print(df)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'prelogits':predict_logits,
        'truelabels':g_label
    }

@torch.no_grad()
def test_for_integrate(model1,model2,args,device, data_loader1,data_loader2, classes_map, save_path=None):
    logit1_list = test_forBART(model=model1, args=args, device=device, data_loader=data_loader1, classes_map=classes_map)['prelogits']
    model2result = test_for_smallBART(args=args, model=model2, device=device, data_loader=data_loader2,
                                      classes_map=classes_map)
    ground_truth_labels = model2result['truelabels']
    logit2_list=model2result['prelogits']
    reverse_classes_map = {classes_map[i]: i for i in classes_map}

    logit1_list=torch.tensor(logit1_list)
    logit2_list=torch.tensor(logit2_list)
    best={"x":0,"y":0,"f1":0,"g_label":[],"p_label":[]}
    k=1000
    if args.dataset=='tbd' or args.dataset=='matres':
        for x in range(1, k):  # best:x=0.19 y=0.81
            logit_list = []
            x=(x*1.0)/k
            y=1.0-x
            for num in range(len(logit1_list)):
                logits = [
                    (logit1_list[num][i] * x + logit2_list[num][i] * y).item()
                    for i in range(len(logit1_list[num]))]
                logit_list.append(logits)
            logit_list = torch.tensor(logit_list)
            a = logit_list.view(-1, logit_list.size(-1))
            pred_labels = torch.max(a, dim=-1).indices
            pred_labels = pred_labels.tolist()
            g_label = ground_truth_labels
            p_label = pred_labels
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
            accuracy = accuracy_score(g_label, p_label)
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
            # macro_f1 = f1_score(g_label, p_label, average='macro')
            # micro_f1 = f1_score(g_label, p_label, average='micro')
            # weighted_f1 = f1_score(g_label, p_label, average='weighted')
            # 将分类结果写入文件中
            label_id = [classes_map[i] for i in classes_map]
            label_id.remove(classes_map['VAGUE'])
            t_name = [reverse_classes_map[i] for i in label_id]
            report = classification_report(g_label, p_label, labels=label_id, target_names=t_name,
                                           output_dict=True)
            micro_f1 = report['micro avg']['f1-score']
            macro_f1 = report['macro avg']['f1-score']
            weighted_f1 = report['weighted avg']['f1-score']
            if micro_f1 > best['f1']:
                best['x'] = x
                best['y'] = y
                best['m'] = 0
                best['n'] = 0
                best['f1'] = micro_f1
                best['g_label'] = g_label
                best['p_label'] = p_label
                freport = report
    df = pd.DataFrame(freport).transpose()
    # df.to_csv(save_path, index=True)
    print("最终分类结果：")
    print(df)
    print(confusion_matrix(best['g_label'], best['p_label']))
    print(best['x'],best['y'],best['m'],best['n'])
    return {
        'macro_f1': freport['macro avg']['f1-score'],
        'micro_f1': freport['micro avg']['f1-score'],
        'weighted_f1': freport['weighted avg']['f1-score']
    }

