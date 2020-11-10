import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from mmf.modules.layers import UpBlock, ResBlock, Interpolate


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, text_dim, condition_dim):
        super(CA_NET, self).__init__()
        self.t_dim = text_dim
        self.c_dim = condition_dim
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = nn.GLU(dim=1)

    def forward(self, text_embedding):
        # Encode
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        
        # Reparametrization
        distrb = Normal(mu, logvar)
        c_code = distrb.rsample()
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, z_dim, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = z_dim + ncf 

        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            nn.GLU(dim=1))

        self.upsample1 = UpBlock(ngf, ngf // 2)
        self.upsample2 = UpBlock(ngf // 2, ngf // 4)
        self.upsample3 = UpBlock(ngf // 4, ngf // 8)
        self.upsample4 = UpBlock(ngf // 8, ngf // 16)
        

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)

        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context_key, content_value):#
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x idf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context_key.size(0), context_key.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = context_key

        # Get weight
        # (batch x queryL x idf)(batch x idf x sourceL)-->batch x queryL x sourceL
        weight = torch.bmm(targetT, sourceT)

        # --> batch*queryL x sourceL
        weight = weight.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            weight.data.masked_fill_(mask.data, -float('inf'))
        weight = torch.nn.functional.softmax(weight, dim=1)

        # --> batch x queryL x sourceL
        weight = weight.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        weight = torch.transpose(weight, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL) --> batch x idf x queryL
        weightedContext = torch.bmm(content_value, weight)  #
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        weight = weight.view(batch_size, -1, ih, iw)

        return weightedContext, weight


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf, r_num, size):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = r_num
        self.size = size
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for _ in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.avg = nn.AvgPool2d(kernel_size=self.size)
        self.A = nn.Linear(self.ef_dim, 1, bias=False)
        self.B = nn.Linear(self.gf_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.M_r = nn.Sequential(
            nn.Conv1d(ngf, ngf * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.M_w = nn.Sequential(
            nn.Conv1d(self.ef_dim, ngf * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.key = nn.Sequential(
            nn.Conv1d(ngf*2, ngf, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Conv1d(ngf*2, ngf, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.memory_operation = Memory()
        self.response_gate = nn.Sequential(
            nn.Conv2d(self.gf_dim * 2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = UpBlock(ngf * 2, ngf)

    def forward(self, h_code, c_code, word_embs, mask, cap_lens):
        """
            h_code(image features):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(word features): batch x cdf x sourceL (sourceL=seq_len)
            c_code: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        # Memory Writing
        word_embs_T = torch.transpose(word_embs, 1, 2).contiguous()
        h_code_avg = self.avg(h_code).detach()
        h_code_avg = h_code_avg.squeeze(3)
        h_code_avg_T = torch.transpose(h_code_avg, 1, 2).contiguous()
        gate1 = torch.transpose(self.A(word_embs_T), 1, 2).contiguous()
        gate2 = self.B(h_code_avg_T).repeat(1, 1, word_embs.size(2))
        writing_gate = torch.sigmoid(gate1 + gate2)
        h_code_avg = h_code_avg.repeat(1, 1, word_embs.size(2))
        memory = self.M_w(word_embs) * writing_gate + self.M_r(h_code_avg) * (1 - writing_gate)

        # Key Addressing and Value Reading
        key = self.key(memory)
        value = self.value(memory)
        self.memory_operation.applyMask(mask)
        memory_out, att = self.memory_operation(h_code, key, value)

        # Key Response
        response_gate = self.response_gate(torch.cat((h_code, memory_out), 1))
        h_code_new = h_code * (1 - response_gate) + response_gate * memory_out
        h_code_new = torch.cat((h_code_new, h_code_new), 1)

        out_code = self.residual(h_code_new)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)
        return out_code, att


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class DMGAN_G(nn.Module):
    def __init__(self, **kwargs):
        super(DMGAN_G, self).__init__()
        ngf = kwargs["gf_dim"]
        nef = kwargs["text_dim"]
        ncf = kwargs["condition_dim"]
        z_dim = kwargs["z_dim"]
        r_num = kwargs["r_num"]
        self.branch_num = kwargs["branch_num"]
        
        self.ca_net = CA_NET(nef, ncf)
        self.h_net1 = INIT_STAGE_G(z_dim, ngf * 16, ncf)
        self.img_net1 = GET_IMAGE_G(ngf)

        if self.branch_num > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf, r_num, 64)
            self.img_net2 = GET_IMAGE_G(ngf)
        if self.branch_num > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf, r_num, 128)
            self.img_net3 = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask, cap_lens):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)
        h_code1 = self.h_net1(z_code, c_code)
        fake_img1 = self.img_net1(h_code1)
        fake_imgs.append(fake_img1)

        if self.branch_num > 1:
            h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, mask, cap_lens)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if self.branch_num > 2:
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask, cap_lens)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)
        return fake_imgs, att_maps, mu, logvar
