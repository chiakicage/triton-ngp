import torch
from lib.config.task.nerf import SphericalHarmonicsEncoderConfig

output_dim = [1, 4, 9, 16]
class Encoder:
    def __init__(self, cfg: SphericalHarmonicsEncoderConfig):
        self.level = cfg.level
        self.output_dim = output_dim[self.level-1]
        # self.input_dim = cfg.input_dim
        # self.output_dim = cfg.input_dim * cfg.L * 2
        # self.freqs = 2.**torch.linspace(0, cfg.L-1, steps=cfg.L)
        # self.embed_fns = []
        # for freq in self.freqs:
        #     self.embed_fns.append(lambda x, freq=freq: torch.sin(freq * x))
        #     self.embed_fns.append(lambda x, freq=freq: torch.cos(freq * x))
        # print(self.embed_fns)

    def __call__(self, x: torch.Tensor):
        # x = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        x, y, z = x[:, 0], x[:, 1], x[:, 2]
        xy = x * y
        xz = x * z
        yz = y * z
        x2 = x * x
        y2 = y * y
        z2 = z * z
        result = []
        if self.level >= 1:
            result.append(torch.ones((x.shape[0], 1)).to(x.device) * 0.28209479177387814)
            if self.level >= 2:
                result.append(
                    torch.stack(
                        [
                            0.4886025119029199 * y,
                            0.4886025119029199 * z,
                            0.4886025119029199 * x,
                        ],
                        dim=-1,
                    ).to(x.device)
                )
                if self.level >= 3:
                    result.append(
                        torch.stack(
                            [
                                1.0925484305920792 * xy,
                                -1.0925484305920792 * yz,
                                0.94617469575755997 * z2 - 0.31539156525251999,
                                -1.0925484305920792 * xz,
                                0.54627421529603959 * x2 - 0.54627421529603959 * y2,
                            ],
                            dim=-1,
                        ).to(x.device)
                    )
                    if self.level >= 4:
                        result.append(
                            torch.stack(
                                [
                                    0.59004358992664352 * y * (-3.0 * x2 + y2),
                                    2.8906114426405538 * xy * z,
                                    0.45704579946446572 * y * (1.0 - 5.0 * z2),
                                    0.3731763325901154 * z * (5.0 * z2 - 3.0),
                                    0.45704579946446572 * x * (1.0 - 5.0 * z2),
                                    1.4453057213202769 * z * (x2 - y2),
                                    0.59004358992664352 * x * (-x2 + 3.0 * y2),
                                ],
                                dim=-1,
                            ).to(x.device)
                        )
        return torch.cat(result, dim=-1)


        # data_out(0) = (T)(0.28209479177387814);                          // 1/(2*sqrt(pi))
        # if (degree <= 1) { return; }
        # data_out(1) = (T)(-0.48860251190291987f*y);                               // -sqrt(3)*y/(2*sqrt(pi))
        # data_out(2) = (T)(0.48860251190291987f*z);                                // sqrt(3)*z/(2*sqrt(pi))
        # data_out(3) = (T)(-0.48860251190291987f*x);                               // -sqrt(3)*x/(2*sqrt(pi))
        # if (degree <= 2) { return; }
        # data_out(4) = (T)(1.0925484305920792f*xy);                                // sqrt(15)*xy/(2*sqrt(pi))
        # data_out(5) = (T)(-1.0925484305920792f*yz);                               // -sqrt(15)*yz/(2*sqrt(pi))
        # data_out(6) = (T)(0.94617469575755997f*z2 - 0.31539156525251999f);                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
        # data_out(7) = (T)(-1.0925484305920792f*xz);                               // -sqrt(15)*xz/(2*sqrt(pi))
        # data_out(8) = (T)(0.54627421529603959f*x2 - 0.54627421529603959f*y2);                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
        # if (degree <= 3) { return; }
        # data_out(9) = (T)(0.59004358992664352f*y*(-3.0f*x2 + y2));                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
        # data_out(10) = (T)(2.8906114426405538f*xy*z);                             // sqrt(105)*xy*z/(2*sqrt(pi))
        # data_out(11) = (T)(0.45704579946446572f*y*(1.0f - 5.0f*z2));                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
        # data_out(12) = (T)(0.3731763325901154f*z*(5.0f*z2 - 3.0f));                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
        # data_out(13) = (T)(0.45704579946446572f*x*(1.0f - 5.0f*z2));                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
        # data_out(14) = (T)(1.4453057213202769f*z*(x2 - y2));                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
        # data_out(15) = (T)(0.59004358992664352f*x*(-x2 + 3.0f*y2));
