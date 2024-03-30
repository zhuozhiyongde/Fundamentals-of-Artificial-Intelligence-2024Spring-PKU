# # 深度学习第二课课上配套代码，卷积神经网络基础
# 导入Pytorch库
import torch
from torch.nn import Conv1d, Conv2d, Conv3d, ReLU, MaxPool2d, Linear, Softmax

torch.manual_seed(99999)

# Convolution on 1D vector
x = torch.tensor([[[1., 3., 3., 0., 1., 2.]]]) # 1x1x6
w = torch.tensor([[[2., 0., 1.]]]) # 1x1x3

conv1d = Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
conv1d.weight = torch.nn.Parameter(w)

out1d = conv1d(x)

print("Input vector:\n", x)
print("Convolution kernel:\n", w)
print("Result:\n", out1d)


import torchvision.models as models
from torchvision import transforms as T
from PIL import Image
import json

# 加载ImageNet的类别索引
with open("imagenet_class_index.json", "r") as fp:
    class_idx = json.load(fp)

idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# 加载预训练的VGG16模型
vgg16_entire = models.vgg16(pretrained=True)

# 图像预处理
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图像
img = Image.open('elephant.jpg')
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0) #将单张图片转换为batch

vgg16_entire.eval() # 设置为评估模式
out = vgg16_entire(batch_t)[0] # 前向传播，预测结果

idx_pred = torch.argmax(out).item() # 获取预测结果的索引
print("Predicted class:", idx2label[idx_pred])


feature_extractor = vgg16_entire.features # 提取特征提取器

out_featue = feature_extractor(batch_t) # 前向传播，提取特征
print("Feature map shape:", out_featue.shape)

# 将特征映射展平为一维向量
out_featue = out_featue.view(out_featue.size(0), -1)
print("Flattened feature map shape:", out_featue.shape)

# 提取更浅层的特征,第9个卷积层conv4_3
feature_extractor_9 = torch.nn.Sequential(*list(vgg16_entire.features.children())[:9])
out_featue_9 = feature_extractor_9(batch_t)
print("Feature map of conv4_3 shape:", out_featue_9.shape)