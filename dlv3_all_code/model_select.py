from models.resnet import resnet18
from models.crnn import CRNN
from models.mobilenet import mobilenet_v2
from models.lenet import LeNet
from models.alexnet import AlexNet
from models.googlenet import googlenet
from models.vgg import vgg11
from models.RNN import RNN
from models.GRU import GRU
from models.LSTM import LSTM
from models.lcznet import LCZNet
from models.liner import Linear

model = {}
model['LeNet'] = LeNet
model['vgg11'] = vgg11
model['AlexNet'] = AlexNet
model['googlenet'] = googlenet
model['mobilenet_v2'] = mobilenet_v2
model['resnet18'] = resnet18
model['RNN'] = RNN
model['GRU'] = GRU
model['LSTM'] = LSTM
model['CRNN'] = CRNN
model['LCZnet'] = LCZNet
model['Linear'] = Linear
print(model.keys())
