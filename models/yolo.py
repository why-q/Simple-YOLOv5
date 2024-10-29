import copy
import math
import sys
import time

import yaml

sys.path.insert(0, "..")
import torch
import torch.nn as nn

from models.common import (
    C3,
    SPP,
    Bottleneck,
    BottleneckCSP,
    Concat,
    Contract,
    Conv,
    DWConv,
    Expand,
    Focus,
)

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[
                    i
                ]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(
        self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None
    ):  # cfg: yaml model file. nc && anchor relay cfg
        super(Model, self).__init__()

        with open(cfg, "r", encoding="utf-8") as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        self.model, self.save = parse_model(
            copy.deepcopy(self.yaml), ch=[ch]
        )  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # self._print_biases()
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info(verbose=True, img_size=640)

    def forward(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers

            if profile:
                o = (
                    thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2
                    if thop
                    else 0
                )  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print("%10.1f%10.0f%10.1fms %-40s" % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print("%.1fms total" % sum(dt))
        return x

    def _initialize_biases(
        self, cf=None
    ):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(
                ("%6g Conv2d.bias:" + "%10.3g" * 6)
                % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean())
            )

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):
    # print("\n------- model layer info -------")
    # print(
    #     "\n%3s%18s%3s%10s  %-40s%-30s%10s%10s"
    #     % ("", "from", "n", "params", "module", "arguments", "cin", "cout")
    # )  # print param cfg
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    np_sum = 0
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:  # noqa: E722
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = (
            nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        )  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        np_sum += np
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params
        print(
            "%3s%18s%3s%10.0f  %-40s%-30s%10s%10s" % (i, f, n, np, t, args, c1, c2)
        )  # print param cfg
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist: the from is not -1 in the yaml cofig，be Concat与Detect
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    print("sum: {}".format(np_sum))
    return nn.Sequential(*layers), sorted(save)


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print("Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def model_info(model, verbose=False, img_size=640):
    print(
        "\n------- verbose:{}, img_size:{} -------\n------- model detail info -------".format(
            verbose, img_size
        )
    )
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )  # number gradients
    if verbose:
        print(
            "%5s %40s %9s %12s %20s %10s %10s"
            % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (
                    i,
                    name,
                    p.requires_grad,
                    p.numel(),
                    list(p.shape),
                    p.mean(),
                    p.std(),
                )
            )

    try:  # FLOPS
        from thop import profile

        stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        img = torch.zeros(
            (1, model.yaml.get("ch", 3), stride, stride),
            device=next(model.parameters()).device,
        )  # input
        flops = (
            profile(copy.deepcopy(model), inputs=(img,), verbose=False)[0] / 1e9 * 2
        )  # stride GFLOPS
        img_size = (
            img_size if isinstance(img_size, list) else [img_size, img_size]
        )  # expand if int/float
        fs = ", %.1f GFLOPS" % (
            flops * img_size[0] / stride * img_size[1] / stride
        )  # 640x640 GFLOPS
    except (ImportError, Exception):
        fs = ""

    print(
        f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}"
    )


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


if __name__ == "__main__":
    import os

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = "models/yolov5s.yaml"
    assert os.path.exists(cfg), "cfg:{} file not exist.".format(cfg)

    # Create model
    model = Model(cfg).to(device)

    # DEBUG  Show model structure
    model.info()

    # Test model
    img = torch.rand(1, 3, 640, 640).to(device)  # img
    # model.train()
    # y = model(img, profile=False)
    # y = model(img, profile=True)  # err

    model.eval()
    y = model(img, profile=False)

    print("end")
