from .Network import MYNET
from utils.utils import *
from tqdm import tqdm
import torch.nn.functional as F
from .losses import SupConLoss



def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    te = Averager()
    ta = Averager()
    treg = Averager()
    tcont = Averager()
    ttotal = Averager()
    model = model.train()
    all_embeddings = []
    all_labels = []
    
    
    criterion = SupConLoss(temperature=0.07)
    
    for i, batch in enumerate(trainloader):
        data, train_label = [_.cuda() for _ in batch]
        lamda=0.2
        beta= 1
        gamma=1
        logits, embedding, _, feature_inter = model(data, stochastic = args.stochastic) # True
        # logits = logits[:, :args.base_class*4] 
        

        logits = logits[:, :args.num_base]
        loss = F.cross_entropy(logits, train_label)
        loss_ce = loss.item()
        embedding = embedding.unsqueeze(1)
      
        contrast_loss_tensor = criterion(embedding, train_label)
        contrast_loss = contrast_loss_tensor.item()
        
       
        acc = count_acc(logits, train_label)
       
        # total_loss = loss

        total_loss = lamda*loss + beta*contrast_loss

        lrc = scheduler.get_last_lr()[0]
        
        tcont.add(contrast_loss)
        te.add(loss_ce)
        ttotal.add(total_loss.item())
        
        tl.add(total_loss.item())
        ta.add(acc)
        #top.add(op_loss)
        # tcont.add(contrast_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        _, embedding, _, _ = model(data, stochastic = args.stochastic) # resulting feature space after training with contrastive loss
        
        all_embeddings.append(embedding.detach().cpu())
        all_labels.append(train_label.detach().cpu())

        
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    tl = tl.item()
    ta = ta.item()
    tcont = tcont.item()
    ttotal = ttotal.item()
    te = te.item()
    treg = 0
    
    return tl, ta, tcont, te, treg, ttotal, all_embeddings, all_labels




def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder' #used for feature extraction rather than classification
            embedding, _ = model(data, stochastic = False)

            embedding_list.append(embedding.cpu())
           
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)  #protos list is stacked along first dim creating tensor with shape [args.base_classes, embedding size]

    model.module.fc.mu.data[:args.base_class] = proto_list  #

    return model


def replace_fc(trainset, transform, model, args, session):
    present_class = (args.base_class + session * args.way)
    previous_class = (args.base_class + (session-1) * args.way)
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(previous_class, present_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.mu[previous_class:present_class] = proto_list  #wghts of fully connected layer's mean(mu) are updated

    return model



def update_sigma_protos_feature_output(trainloader, trainset, model, args, session):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    
    
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            _,embedding, _, _ = model(data, stochastic=False)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    radius = []
    if session == 0:
        
        for class_index in range(args.num_base):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            feature_class_wise = embedding_this.numpy()
            cov = np.cov(feature_class_wise.T)
            
            radius.append(np.trace(cov)/64)
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        
        args.radius = np.sqrt(np.mean(radius)) 
        args.proto_list = torch.stack(proto_list, dim=0)
    else:
        for class_index in  np.unique(trainset.targets):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            feature_class_wise = embedding_this.numpy()
            cov = np.cov(feature_class_wise.T)
            radius.append(np.trace(cov)/64)
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        args.proto_list = torch.cat((args.proto_list, torch.stack(proto_list, dim=0)), dim =0)



def update_sigma_novel_protos_feature_output(support_data, support_label, model, args, session):
    
    model = model.eval()
    
    embedding_list = []
    label_list = []

    with torch.no_grad():
        data, label = support_data, support_label
        #model.module.mode = 'encoder'
        _,embedding, _, _= model(data, stochastic=False)

        embedding_list.append(embedding.cpu())
        label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    radius = []
    assert session > 0
    for class_index in  support_label.cpu().unique():

        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        feature_class_wise = embedding_this.numpy()
        cov = np.cov(feature_class_wise.T)
        radius.append(np.trace(cov)/64)
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
    args.proto_list = torch.cat((args.proto_list, torch.stack(proto_list, dim=0)), dim =0)


def test_agg(model, testloader, epochs, args, session, print_numbers=False, save_pred=False): 
 
    test_class = args.num_base + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va_agg = Averager()
    va_agg_stochastic_agg = Averager()
    num_stoch_samples = 10

    da = DAverageMeter()
    ca = DAverageMeter()
    pred_list = []
    label_list = []
    per_class_acc_list = []
    with torch.no_grad():
        # tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(testloader):
            data, test_label = [_.cuda() for _ in batch]

            logits, features, _, _ = model(data, stochastic=False)
            logits = logits[:, :test_class]
            pred = torch.argmax(logits, dim=1)
            if session == args.num_session - 1:
                pred_list.append(pred)
                label_list.append(test_label)
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            per_cls_acc, cls_sample_count = count_per_cls_acc(logits, test_label)
            da.update(per_cls_acc)
            ca.update(cls_sample_count)
            
        vl = vl.item()
        va = va.item()
        va_agg = va
        da = da.average()
        ca = ca.average()
        
        acc_dict = acc_utils(da, args.num_base, args.num_session, args.way, session)
        

    if print_numbers:
        print(acc_dict)
    return vl, va, da, va_agg, acc_dict