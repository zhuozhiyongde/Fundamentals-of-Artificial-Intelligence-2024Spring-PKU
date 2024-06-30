#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   supervised.py
# @Time    :   2024/06/30 19:09:04
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code

"""
supervised.py: 监督学习训练模型。
划分数据集，创建网络和优化器，训练过程中每一个epoch结束会在验证集上测试。
"""

from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from feature import FeatureAgent

from model import SelfVecModel

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from tqdm import tqdm

version = "vec-fix-act-128"
data_folder = "data-vec"

writer = SummaryWriter(log_dir=f"./runs/{version}")

if __name__ == "__main__":
    logdir = f"log/{version}/"
    os.makedirs(logdir, exist_ok=True)

    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    print("[Loading Train dataset]")
    trainDataset = MahjongGBDataset(
        data_folder, 0, splitRatio, 0, augment=True, lazy=False
    )
    print("[Loading Validation dataset]")
    validateDataset = MahjongGBDataset(
        data_folder, splitRatio, 1, 0, augment=False, lazy=False
    )
    loader = DataLoader(
        dataset=trainDataset, batch_size=batchSize, shuffle=True, num_workers=24
    )
    vloader = DataLoader(
        dataset=validateDataset, batch_size=batchSize, shuffle=False, num_workers=24
    )

    # Load model
    model = SelfVecModel(
        obs_dim=FeatureAgent.OBS_SIZE, vec_dim=FeatureAgent.VEC_SIZE
    ).to("cuda")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Total number of parameters] {total_params}\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # 从 SelfAgent/log/checkpoint/8.pkl 开始加载
    # model.load_state_dict(torch.load(logdir + "checkpoint/8.pkl"))

    # Train and validate
    num_epoch = 30
    for e in range(num_epoch):
        print(f"[Epoch {e}]")
        # 存储模型
        torch.save(model.state_dict(), logdir + "%d.pkl" % e)

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(
            loader,
            desc=f"Training Epoch {e}".ljust(20),
            bar_format="{l_bar:20}{bar:40}{r_bar}",
        )

        for i, d in enumerate(pbar):
            input_dict = {
                "is_training": True,
                "obs": {
                    "observation": d[0].cuda(),
                    "action_mask": d[1].cuda(),
                    "vec": d[2].cuda(),
                },
            }
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[3].long().cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * d[0].size(0)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, d[3].cuda()).sum().item()
            total += d[0].size(0)

            # set postfix for tqdm, acc and loss
            pbar.set_postfix(
                acc=f"{correct / total:.4f}",
                loss=f"{total_loss / total:.4f}",
            )

        avg_train_loss = total_loss / total
        train_acc = correct / total
        writer.add_scalars("Loss", {"Train": avg_train_loss}, e)
        writer.add_scalars("Accuracy", {"Train": train_acc}, e)

        # Validation
        correct = 0
        total_val_loss = 0
        total = 0

        pbar = tqdm(
            vloader,
            desc=f"Validation Epoch {e}".ljust(20),
            bar_format="{l_bar}{bar:40}{r_bar}",
        )

        for i, d in enumerate(pbar):
            input_dict = {
                "is_training": False,
                "obs": {
                    "observation": d[0].cuda(),
                    "action_mask": d[1].cuda(),
                    "vec": d[2].cuda(),
                },
            }
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim=1)
                val_loss = F.cross_entropy(logits, d[3].long().cuda())
                total_val_loss += val_loss.item() * d[0].size(0)
                correct += torch.eq(pred, d[3].cuda()).sum().item()
                total += d[0].size(0)
                pbar.set_postfix(
                    acc=f"{correct / total:.4f}",
                    loss=f"{total_val_loss / total:.4f}",
                )

        avg_val_loss = total_val_loss / len(validateDataset)
        val_acc = correct / len(validateDataset)
        writer.add_scalars("Loss", {"Validate": avg_val_loss}, e)
        writer.add_scalars("Accuracy", {"Validate": val_acc}, e)
