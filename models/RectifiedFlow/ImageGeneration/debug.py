import seaborn as sns
import torch

sns.set(font_scale=2)
sns.set(style="whitegrid")

from models import ddpm as ddpm_model

#from configs.ncsnpp import cifar10_continuous_ve as configs
from configs.ddpm import cifar10_continuous_vp as configs
config = configs.get_config()

checkpoint = torch.load('exp/ddpm_continuous_vp.pth')

#score_model = ncsnpp.NCSNpp(config)
score_model = ddpm_model.DDPM(config)
score_model.load_state_dict(checkpoint)
score_model = score_model.eval()
x = torch.ones(8, 3, 32, 32)
y = torch.tensor([1] * 8)
breakpoint()
with torch.no_grad():
  score = score_model(x, y)