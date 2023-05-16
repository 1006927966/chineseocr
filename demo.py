import torch
from torch.autograd import Variable
from PIL import Image
from chineseConvert import strLabelConverter
import dataset
import crnn as crnn
import os

# 模型的存储路径
model_path = './netCRNN_34.pth'
# 测试图像路径
imgdir = "./测试图像"
# 类别文件存储路径
txtpath = "./labeld.txt"
#alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 5990, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

#converter = utils.strLabelConverter(alphabet)
converter = strLabelConverter(txtpath)
picnames = os.listdir(imgdir)

for picname in picnames:
    print(picname)
    img_path = os.path.join(imgdir, picname)
    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    print(image.size())
    image = image.view(1, *image.size())
    print(image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    print(preds.size())
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data) # 这里面有空格和重复 没有进行处理
    #sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s =>' % (raw_pred))
