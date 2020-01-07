import os
import sys
import torch
import torch.autograd as Tautograd
import torch.nn.functional as F


# 模型进行训练
def train(train_iter, dev_iter, model, args):
    """
    训练数据
    :param train_iter:
    :param dev_iter:
    :param model:
    :param args:
    :return:
    """
    # 如果使用了cuda 则使用cuda
    if args.cuda:
        model.cuda()
    # optimizer设置优化器 使用 Adam 计算梯度下降 对参数进行迭代优化
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 数据清零
    steps = 0
    best_acc = 0
    last_step = 0
    # 模型进行训练
    # model.train() ：启用 BatchNormalization 和 Dropout
    # model.eval() ：不启用 BatchNormalization 和 Dropout
    # 把模型设置为 train（）模式
    model.train()
    # 进行轮数循环
    for epoch in range(1, args.epochs+1):
        # 从 train_iter 读取一个 batch
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature = feature.data.t()  # 原地进行转置
            target = target.data.sub(1)  # 把target中的每个值都减1
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            # 设置数据 初始化为0
            optimizer.zero_grad()
            # 输入feature 获得模型计算值
            logit = model(feature)
            # 使用交叉熵计算损失
            loss = F.cross_entropy(logit, target)
            # 反向传播 求梯度 不用显示出来，更新数据时自动调用
            loss.backward()
            # 更新数据
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))

            if steps % args.test_interval == 0:
                # 多少次迭代后 对 数据进行交叉验证集 测试
                # 交叉验证 必须加 eval
                # 对数据进行交叉验证集 测试验证
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                # 多少次迭代后 对数据进行保存
                save(model, args.save_dir, "snapshot", steps)


# 如果模型中用了dropout或bn，那么predict时必须使用eval
# 否则结果是没有参考价值的，不存在选择的余地
# 模型 交叉验证 测试
def eval(data_iter, model, args):
    # 把模型设置为 eval（）模式
    # pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature = feature.data.t()
        target = target.data.sub(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


# 模型进行预测
def predict(text, model, text_field, label_field, cuda_flag):
    # 断言 判断text是否为字符串
    assert isinstance(text, str)
    # 把模型设置为 eval（）模式
    model.eval()
    # 文本进行处理
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    # autograd根据用户对variable的操作Function构建其计算图
    x = Tautograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    # 输出结果
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_field.vocab.itos[predicted.data[0]+1]


# 保存模型
def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
