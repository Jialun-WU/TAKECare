import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import ModuleList
from math import sqrt

class Time2Vec(nn.Module):
    def __init__(self, dim):
        super(Time2Vec, self).__init__()
        self.dim = dim
        self.omega = nn.Parameter(torch.randn(dim))
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, t):
        t = t.unsqueeze(-1)
        cos_terms = torch.cos(t * self.omega)
        sin_terms = torch.sin(t * self.omega)
        t2v = torch.cat([cos_terms, sin_terms], dim=-1)
        return self.linear(t2v)


class LeafMP(nn.Module):
    def __init__(self, dim):
        super(LeafMP, self).__init__()
        self.linear_e = nn.Linear(dim, dim)
        self.linear_c = nn.Linear(dim, dim)
        self.linear_t = nn.Linear(dim, dim)
        self.b_e = nn.Parameter(torch.zeros(dim))
        self.b_c = nn.Parameter(torch.zeros(dim))
        self.gamma = 0.5
        self.g = nn.Linear(dim, dim)
        self.time_embed = Time2Vec(dim // 2)
        self.visit_attn = nn.Parameter(torch.randn(dim))

    def score(self, h_e, h_c, alpha_e):
        proj_e = self.linear_e(h_e) + self.b_e
        proj_c = self.linear_c(h_c) + self.b_c
        return (proj_e * proj_c).matmul(torch.sigmoid(self.linear_t(alpha_e)).unsqueeze(-1)).squeeze(-1)

    def edge_to_node(self, H_e, H_c, edge_index, timestamps):
        updated = H_c.clone()
        for c in range(H_c.shape[0]):
            edges = edge_index[c]
            if not edges:
                continue
            hcs = H_c[c].expand(len(edges), -1)
            hes = torch.stack([H_e[e] for e in edges])
            tes = torch.stack([timestamps[e] for e in edges])
            aes = self.time_embed(tes)
            scores = self.score(hes, hcs, aes)
            attn = F.softmax(scores, dim=0)
            agg = torch.sum(attn.unsqueeze(1) * self.g(hes), dim=0)
            updated[c] = self.gamma * agg + (1 - self.gamma) * H_c[c]
        return updated

    def node_to_edge(self, H_c, edge2code):
        H_e = []
        for codes in edge2code:
            agg = torch.sum(torch.stack([H_c[c] for c in codes]), dim=0)
            H_e.append(F.relu(self.linear_e(agg)))
        return torch.stack(H_e)

    def aggregate_patient(self, H_e, edge2patient):
        reps = []
        for edges in edge2patient:
            if not edges:
                reps.append(torch.zeros(H_e.size(1)).to(H_e.device))
                continue
            h_es = torch.stack([H_e[e] for e in edges])
            attn = F.softmax(torch.matmul(h_es, self.visit_attn), dim=0)
            reps.append(torch.sum(attn.unsqueeze(1) * h_es, dim=0))
        return torch.stack(reps)


class AnceMP(nn.Module):
    def __init__(self, dim, heads=4):
        super(AnceMP, self).__init__()
        self.gat_layers = ModuleList([
            GATConv(dim, dim // heads, heads=heads),
            GATConv(dim, dim // heads, heads=heads)
        ])
        self.w_co = nn.Parameter(torch.randn(dim))

    def forward(self, H, edge_index):
        for gat in self.gat_layers:
            H = gat(H, edge_index)
        return H

    def aggregate_patient(self, H, node2patient):
        reps = []
        for nodes in node2patient:
            if not nodes:
                reps.append(torch.zeros(H.size(1)).to(H.device))
                continue
            h_ns = H[nodes]
            attn = F.softmax(torch.matmul(h_ns, self.w_co), dim=0)
            reps.append(torch.sum(attn.unsqueeze(1) * h_ns, dim=0))
        return torch.stack(reps)


class PatRep(nn.Module):
    def __init__(self, dim):
        super(PatRep, self).__init__()
        self.leaf = LeafMP(dim)
        self.ance = AnceMP(dim)

    def forward(self, H_leaf, edge2code, code2edge, timestamps, edge2patient,
                H_ance, ance_edge_index, ance_node2patient):
        # Leaf message passing
        H_e = self.leaf.node_to_edge(H_leaf, edge2code)
        for _ in range(2):
            H_leaf = self.leaf.edge_to_node(H_e, H_leaf, code2edge, timestamps)
            H_e = self.leaf.node_to_edge(H_leaf, edge2code)
        leaf_rep = self.leaf.aggregate_patient(H_e, edge2patient)

        # Ancestor message passing
        H_ance = self.ance(H_ance, ance_edge_index)
        ance_rep = self.ance.aggregate_patient(H_ance, ance_node2patient)

        # Combine both levels
        rep = torch.cat([leaf_rep, ance_rep], dim=1)
        return rep