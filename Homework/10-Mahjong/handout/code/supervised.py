"""
监督学习流程。划分数据集，创建网络和优化器，训练过程中每一个epoch结束会在验证集上测试。

提示：可以考虑按照下列方式修改代码：
1. 改变数据集划分、batch大小、学习率等超参数；
2. 从上一次训练的checkpoint加载模型接续训练；
3. 对训练进展进行监控和分析（如打印日志到tensorboard等）。
"""

from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from model import CNNModel
import torch.nn.functional as F
import torch
import os

if __name__ == "__main__":
    logdir = "log/"
    os.makedirs(logdir + "checkpoint", exist_ok=True)

    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    loader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
    vloader = DataLoader(dataset=validateDataset, batch_size=batchSize, shuffle=False)

    # Load model
    model = CNNModel().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Train and validate
    for e in range(16):
        print("Epoch", e)
        torch.save(model.state_dict(), logdir + "checkpoint/%d.pkl" % e)
        for i, d in enumerate(loader):
            input_dict = {
                "is_training": True,
                "obs": {"observation": d[0].cuda(), "action_mask": d[1].cuda()},
            }
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            if i % 128 == 0:
                print(
                    "Iteration %d/%d" % (i, len(trainDataset) // batchSize + 1),
                    "policy_loss",
                    loss.item(),
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Run validation:")
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {
                "is_training": False,
                "obs": {"observation": d[0].cuda(), "action_mask": d[1].cuda()},
            }
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim=1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        print("Epoch", e + 1, "Validate acc:", acc)
