import os
import sys
import torch
import time
import math
import torch.autograd as Tautograd
import main


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins*60))
    return elapsed_mins, elapsed_secs


def train_model(train_iter, dev_iter, model, args):
    # 定义一个最大值
    best_valid_loss = float('inf')
    steps = 0
    # Vocab类在stoi属性中包含从word到id的映射
    # 即表示 忽略 将'<pad>'转换对应的id数字，即表示忽略缺少的部分，这样计算交叉熵的时候才不会计算失误
    cross_loss = args.cross_loss
    for epoch in range(1, args.epochs+1):
        start_time = time.time()
        train_loss = train(train_iter, model, cross_loss, args)
        dev_loss = eval(dev_iter, model, cross_loss, args)
        end_time = time.time()
        steps += 1
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # 当 评估的损失 低于 best_valid_loss
        if dev_loss < best_valid_loss:
            best_valid_loss = dev_loss
            # state_dict()：使用最佳验证损失的epoch参数作为模型的最终参数
            save(model, args.save_dir, 'best', steps)
        # 使用一个batch内的平均损失计算困惑度
        # PPL 困惑度
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {dev_loss:.3f} |  Val. PPL: {math.exp(dev_loss):7.3f}')


def train(train_iter, model, cross_loss, args):
    """
    训练模型
    :param train_iter:
    :param model:
    :param cross_loss:
    :param args:
    :return:
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_loss = 0
    model.train()
    steps = 0
    for batch in train_iter:
        feature = batch.src
        target = batch.trg
        optimizer.zero_grad()
        output = model(feature, target, 0.5)
        # view函数: 减少output和target的维度以便进行loss计算
        # 第一维度不参与计算，即放弃第一维度
        output = output[1:].view(-1, output.shape[-1])
        target = target[1:].view(-1)
        # 交叉熵计算 损失
        loss = cross_loss(output, target)
        loss.backward()
        # clip_grad_norm: 进行梯度裁剪，防止梯度爆炸。clip：梯度阈值
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        epoch_loss += loss.item()
        steps += 1
        # 计算正确的个数
        corrects = (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / batch.batch_size
        sys.stdout.write(
            '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                     loss.item(),
                                                                     accuracy,
                                                                     corrects,
                                                                     batch.batch_size))

    return epoch_loss / len(train_iter)


def eval(dev_iter, model, cross_loss, args):
    """
    评估测试模型
    """
    epoch_loss = 0
    model.eval()
    # 关闭autograd 引擎 不会进行反向计算，加快速度
    # 当然 也可以不写这个，只是慢一点，其实影响不大
    with torch.no_grad():
        for batch in dev_iter:
            feature = batch.src
            target = batch.trg
            output = model(feature, target, 0)
            output = output[1:].view(-1, output.shape[-1])
            target = target[1:].view(-1)
            loss = cross_loss(output, target)
            epoch_loss += loss.item()
    return epoch_loss / len(dev_iter)


def save(model, save_dir, save_prefix, steps):
    """
    保存模型
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
    f = open("best_model.txt", 'w')
    f.write(save_path)
    f.close()