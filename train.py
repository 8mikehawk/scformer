def evaluate(model):
    with torch.no_grad():
        net = model
        eval_loss = 0
        eval_acc = 0
        eval_miou = 0
        for j, sample in tqdm(enumerate(val_data)):
            valImg = Variable(sample['image'].to(cfg['device']))
            valLabel = Variable(sample['label'].long().to(cfg['device']))

            out = net(valImg)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, valLabel)
            eval_loss = loss.item() + eval_loss
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = valLabel.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrics = eval_semantic_segmentation(pre_label, true_label, cfg['n_class'])
            eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
            eval_miou = eval_metrics['miou'] + eval_miou

        if max(best_val) <= eval_miou / len(val_data):
            best_val.append(eval_miou / len(val_data))
            print('val_miou:', max(best_val))
            torch.save(net.state_dict(), cfg['val_pth_save_path'])

if __name__ == '__main__':
    from utilis.build_model import build_tri_heads
    from utilis.get_paras import get_hyper_parameters
    from utilis.load_data import load_ISIC2018, load_city_scapes
    from utilis.eval_semantic_segmentation import eval_semantic_segmentation
    import torch

    import torch.nn.functional as F

    from torch.autograd import Variable

    import torch.nn as nn
    import torch.optim as optmi
    from tqdm import tqdm

    # 定义模型
    net = build_tri_heads("tri-heads-0")

    # 定义超参
    cfg = get_hyper_parameters()

    # 读取数据
    train_data, val_data = load_ISIC2018()

    nets = net.to(cfg['device'])
    # 损失函数和优化器
    criterion = nn.NLLLoss().to(cfg['device'])
    optimizer = optmi.AdamW(net.parameters(), lr=cfg['lr'])

    best_val = [0]
    best = [0]
    best_val_dice = [0]

    for epoch in tqdm(range(cfg['epoch_number'])):
        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        net = net.train()

        for i, sample in tqdm(enumerate(train_data)):
            img_data = Variable(sample['image'].to(cfg['device']))
            img_label = Variable(sample['label'].to(cfg['device']))

            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]
            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label, cfg['n_class'])
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            train_class_acc += eval_metrix['class_accuracy']

        metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            train_class_acc / len(train_data))

        if max(best) <= train_miou / len(train_data):
            best.append(train_miou / len(train_data))
            print('train_miou:', max(best))
            torch.save(net.state_dict(), cfg['train_pth_save_path'])
        
        if epoch % 100 == 0:
            evaluate(net)