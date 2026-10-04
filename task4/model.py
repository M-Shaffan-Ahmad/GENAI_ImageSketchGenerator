import torch
from torch import nn

STYLES=('Fine Pencil','Technical Ink','Tonal Charcoal')


def style_map(embedding,style,image):
    return embedding(style)[:,:,None,None].expand(-1,-1,image.shape[2],image.shape[3])


def generator_config(config):
    return {key:config[key] for key in ['base_channels','dropout','embedding_dim']}


class SketchGenerator(nn.Module):
    """Five-level U-Net, categorical style embedding and paired skip features."""
    def __init__(self,base_channels=32,dropout=.1,embedding_dim=8):
        super().__init__()
        self.style_embedding=nn.Embedding(3,embedding_dim)
        channels=[base_channels*i for i in [1,2,4,8,8]]
        self.down=nn.ModuleList();previous=3+embedding_dim
        for i,c in enumerate(channels):
            layers=[nn.Conv2d(previous,c,4,2,1)]
            if i>0:layers.append(nn.InstanceNorm2d(c,affine=True))
            layers.append(nn.LeakyReLU(.2))
            self.down.append(nn.Sequential(*layers));previous=c
        self.up=nn.ModuleList()
        for i,c in enumerate(reversed(channels[:-1])):
            layers=[nn.ConvTranspose2d(previous,c,4,2,1),nn.InstanceNorm2d(c,affine=True),nn.ReLU()]
            if i<2:layers.append(nn.Dropout(dropout))
            self.up.append(nn.Sequential(*layers));previous=2*c
        self.output=nn.Sequential(nn.ConvTranspose2d(previous,3,4,2,1),nn.Sigmoid())

    def forward(self,image,style):
        x=torch.cat([image,style_map(self.style_embedding,style,image)],dim=1)
        skips=[]
        for down in self.down:x=down(x);skips.append(x)
        for up,skip in zip(self.up,reversed(skips[:-1])):x=torch.cat([up(x),skip],dim=1)
        return self.output(x)


class PatchDiscriminator(nn.Module):
    """70-pixel PatchGAN receptive field; photo, sketch and style are conditioned."""
    def __init__(self,base_channels=32,embedding_dim=8):
        super().__init__()
        self.style_embedding=nn.Embedding(3,embedding_dim)
        layers=[];previous=6+embedding_dim
        for i,c in enumerate([base_channels,base_channels*2,base_channels*4,base_channels*8]):
            layers.append(nn.Conv2d(previous,c,4,2 if i<3 else 1,1))
            if i>0:layers.append(nn.InstanceNorm2d(c,affine=True))
            layers.append(nn.LeakyReLU(.2));previous=c
        layers.append(nn.Conv2d(previous,1,4,1,1));self.network=nn.Sequential(*layers)

    def forward(self,image,sketch,style):
        return self.network(torch.cat([image,sketch,style_map(self.style_embedding,style,image)],dim=1))


def load_checkpoint(path,device='cpu'):
    saved=torch.load(path,map_location=device,weights_only=True)
    if saved.get('component')!='sketch_cgan' or saved.get('styles')!=list(STYLES):raise ValueError('Not a compatible Task 4 checkpoint')
    generator=SketchGenerator(**generator_config(saved['config'])).to(device)
    generator.load_state_dict(saved['generator'])
    return generator,saved
