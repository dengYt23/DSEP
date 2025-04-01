import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rgcn.model import BaseRGCN
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from SAD import SeConvTransE, SeConvTransR


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                            activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
             
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r)
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class DSEP(nn.Module):
    def __init__(self, gcn, decoder, num_ents, num_rels, num_static_rels, num_words, time_interval,
                  h_dim, opn, history_rate, num_bases=-1, num_basis=-1, num_hidden_layers=1, dropout=0, 
                  self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0, hidden_dropout=0, feat_dropout=0, 
                  static_weight=1, discount=0, angle=0, use_static=False, use_cuda=False, gpu = 0, analysis=False, 
                  hidden_size=None, ent_emb=None, rel_emb=None, static_graph=None):
        super(DSEP, self).__init__()
        
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.time_interval = time_interval
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.static_weight = static_weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.use_cuda = use_cuda
        self.device = torch.device(f'cuda:{gpu}' if use_cuda else 'cpu')
        self.static_graph = static_graph

        self.ent_emb = nn.Parameter(ent_emb.clone().detach().requires_grad_(True)).to(self.device)
        self.rel_emb = nn.Parameter(rel_emb.clone().detach().requires_grad_(True)).to(self.device)

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(num_words, h_dim)).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.static_rgcn_layer = RGCNBlockLayer(h_dim, h_dim, num_static_rels*2, num_bases, 
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)

        self.rgcn = RGCNCell(
            num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis,
            num_hidden_layers, dropout, self_loop, skip_connect, gcn,
            self.opn, None, use_cuda, analysis
            )

        self.relation_cell = nn.GRUCell(self.h_dim*2, self.h_dim)
        self.entity_cell = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder == "seconvtranse":
            self.decoder_ob1 = SeConvTransE(num_ents, h_dim, hidden_size, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2 = SeConvTransE(num_ents, h_dim, hidden_size, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = SeConvTransR(num_rels, h_dim, hidden_size, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = SeConvTransR(num_rels, h_dim, hidden_size, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 
        
        self.w1 = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size)).float()
        torch.nn.init.normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(hidden_size, h_dim)).float()
        torch.nn.init.normal_(self.w2)

        self.bias_r = nn.Parameter(torch.Tensor(hidden_size))
        torch.nn.init.zeros_(self.bias_r)

        self.seq = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, h_dim),
        )

    def forward(self, g_list):
        ent_emb = self.seq(self.ent_emb)
        ent_emb = F.normalize(ent_emb) if self.layer_norm else ent_emb

        if self.use_static:
            static_graph = self.static_graph.to(self.device)
            static_graph.ndata['h'] = torch.cat((ent_emb, self.words_emb), dim=0)
            self.static_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
        else:
            static_emb = F.normalize(ent_emb) if self.layer_norm else ent_emb
        self.h = static_emb

        # inverse relation embeddings
        r_rel_emb = torch.mm(self.rel_emb, self.w1) + self.bias_r
        # concatenate relation embeddings
        rel_emb = torch.cat((self.rel_emb, r_rel_emb), dim=0)
        rel_emb = torch.mm(rel_emb, self.w2)
        rel_emb = F.normalize(rel_emb) if self.layer_norm else rel_emb

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.device)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim, device=self.device)

            span_start, span_end = g.r_len.T
            relation_means = torch.stack([temp_e[start:end].mean(dim=0) for start, end in zip(span_start, span_end)])
            x_input[g.uniq_r] = relation_means

            x_input = torch.cat((rel_emb, x_input), dim=1)
            if i == 0:
                self.h_0 = self.relation_cell(x_input, rel_emb) 
            else:
                self.h_0 = self.relation_cell(x_input, self.h_0)
            self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h, self.h_0)  
            current_h = F.normalize(current_h) if self.layer_norm else current_h

            self.h = self.entity_cell(current_h, self.h) 
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0

    def predict(self, test_graph, all_triplets, e_e_his_emb=None, e_r_his_emb=None, e_r_his_id=None, e_e_his_id=None):
        """
        Predict entity and relation embeddings.
        
        Args:
            test_graph: m historical subgraphs.
            all_triplets: current triplets.
            e_e_his_emb: Embedding of historical facts related to the relation to be predicted at the current moment
            e_r_his_emb: Embedding of historical facts related to the entity to be predicted at the current moment
            e_r_his_id: Entity-Relation hisrotical correlation IDs.
            e_e_his_id: Subject-Obejct hisrotical correlation IDs.
            
        Returns:
            score_ent: Entity scores.
            score_rel: Relation scores.
        """

        with torch.no_grad():
            e_e_his_emb = e_e_his_emb.to(self.device)
            e_r_his_emb = e_r_his_emb.to(self.device)

            evolve_embs, _, r_emb = self.forward(test_graph)
            pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            # Time embeddings
            time_embs = self.get_init_time(all_triplets)

            # Convert IDs to matrices
            e_e_his_matrix, e_r_his_matrix = self.id2matrix(e_r_his_id, e_e_his_id)

            # raw scores
            score_r = F.softmax(self.decoder_ob1(pre_emb, r_emb, time_embs, all_triplets, e_r_his_emb), dim=1)
            score_rel_r = F.softmax(self.rdecoder_re1(pre_emb, r_emb, time_embs, all_triplets, e_e_his_emb), dim=1)

            # historical scores
            score_h = F.softmax(self.decoder_ob2(pre_emb, r_emb, time_embs, all_triplets, e_r_his_emb, e_r_his_matrix), dim=1)
            score_rel_h = F.softmax(self.rdecoder_re2(pre_emb, r_emb, time_embs, all_triplets, e_e_his_emb, e_e_his_matrix), dim=1)

            # final scores
            score_ent = self.history_rate * score_h + (1 - self.history_rate) * score_r
            score_ent = torch.log(score_ent)

            score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            score_rel = torch.log(score_rel)

            return score_ent, score_rel

    def get_loss(self, glist, all_triplets, e_e_his_emb=None, e_r_his_emb=None, e_r_his_id=None, e_e_his_id=None):
        loss_ent = torch.zeros(1, device=self.device)
        loss_rel = torch.zeros(1, device=self.device)
        loss_static = torch.zeros(1, device=self.device)

        e_r_his_emb = e_r_his_emb.to(self.device)
        e_e_his_emb = e_e_his_emb.to(self.device)

        evolve_embs, static_emb, r_emb = self.forward(glist)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        # Time embeddings
        time_embs = self.get_init_time(all_triplets)

        # Convert IDs to matrices
        e_e_his_matrix, e_r_his_matrix = self.id2matrix(e_r_his_id, e_e_his_id)

        # raw scores
        score_r = F.softmax(self.decoder_ob1(pre_emb, r_emb, time_embs, all_triplets, e_r_his_emb), dim=1)
        score_rel_r = F.softmax(self.rdecoder_re1(pre_emb, r_emb, time_embs, all_triplets, e_e_his_emb), dim=1)

        # historical scores
        score_h = F.softmax(self.decoder_ob2(pre_emb, r_emb, time_embs, all_triplets, e_r_his_emb, e_r_his_matrix), dim=1)
        score_rel_h = F.softmax(self.rdecoder_re2(pre_emb, r_emb, time_embs, all_triplets, e_e_his_emb, e_e_his_matrix), dim=1)

        # final scores
        score_ent = self.history_rate * score_h + (1 - self.history_rate) * score_r
        scores_ent = torch.log(score_ent)
        loss_ent += F.nll_loss(scores_ent, all_triplets[:, 2])
     
        score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
        scores_rel = torch.log(score_rel)
        loss_rel += F.nll_loss(scores_rel, all_triplets[:, 1])

        if self.use_static:
            for time_step, evolve_emb in enumerate(evolve_embs):
                if self.discount == 1:
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                elif self.discount == 0:
                    step = (self.angle * math.pi / 180)
                if self.layer_norm:
                    sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                else:
                    sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                    c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                    sim_matrix = sim_matrix / c
                
                mask = (math.cos(step) - sim_matrix) > 0
                loss_static += self.static_weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static
    
    def get_init_time(self, quadrupleList):
        """
        Generate time embeddings based on quadruples.
        """

        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = torch.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2
    
    def id2matrix(self, e_r_his_id, e_e_his_id, local_obj=None):
        """
        Convert IDs to matrices.
        """

        e_r_his_matrix = torch.zeros((len(e_r_his_id), self.num_ents), device=self.device).float()
        e_e_his_matrix = torch.zeros((len(e_e_his_id), self.num_rels*2), device=self.device).float()
        for i, e_id in enumerate(e_r_his_id):
            e_r_his_matrix[i, e_id] = 1
        for i, r_id in enumerate(e_e_his_id):
            e_e_his_matrix[i, r_id] = 1
        if local_obj:
            for i, obj_id in enumerate(local_obj):
                obj_id = list(obj_id)
                e_r_his_matrix[i, obj_id] = 1
        return e_e_his_matrix, e_r_his_matrix
