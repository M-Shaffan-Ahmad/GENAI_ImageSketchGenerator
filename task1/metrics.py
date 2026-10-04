import torch
import torch.nn.functional as F


def ssim(x, y):
    """Per-image SSIM, Gaussian 11x11 window sigma=1.5, valid spatial support."""
    coords = torch.arange(11, device=x.device, dtype=x.dtype)-5
    g = torch.exp(-coords.square()/(2*1.5**2)); g = g/g.sum()
    kernel = (g[:, None]*g[None, :]).expand(3, 1, 11, 11).contiguous()
    def avg(z):
        return F.conv2d(z, kernel, groups=3)
    mx, my = avg(x), avg(y)
    vx, vy, cov = avg(x*x)-mx*mx, avg(y*y)-my*my, avg(x*y)-mx*my
    score = ((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return score.mean((1, 2, 3))


def measurements(x, y):
    return dict(l1=(x-y).abs().mean((1, 2, 3)), ssim=ssim(x, y),
                psnr=-10*torch.log10((x-y).square().mean((1, 2, 3)).clamp_min(1e-12)))


def reconstruction_loss(x, y, alpha):
    return alpha*F.l1_loss(x, y)+(1-alpha)*(1-ssim(x, y).mean())
