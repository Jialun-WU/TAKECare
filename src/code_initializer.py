import os
import pickle
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from CLUB_MI import InfoNCE
from tqdm import tqdm

class CodeInit(nn.Module):
    def __init__(self, code2cui_path, kg_path, vocab_path, sapbert_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", emb_dim=768, device='cuda'):
        super(CodeInitializer, self).__init__()
        self.device = device
        self.emb_dim = emb_dim

        # Load mappings and KG
        self.code2cui = self.load_pickle(code2cui_path)
        self.kg = self.load_pickle(kg_path)
        self.vocab = self.load_pickle(vocab_path)

        # SAPBERT encoder
        self.tokenizer = AutoTokenizer.from_pretrained(sapbert_model)
        self.model = AutoModel.from_pretrained(sapbert_model).to(device)

        # Placeholder: entities and relations from KG
        self.entities = sorted(list({u for rel in self.kg.values() for u, _ in rel} | set(self.kg.keys())))
        self.entity2id = {e: i for i, e in enumerate(self.entities)}
        self.n_entities = len(self.entities)
        self.n_relations = len(set([r for v in self.kg.values() for _, r in v]))

        # Embeddings
        self.entity_emb = nn.Parameter(torch.randn(self.n_entities, emb_dim))
        self.relation_emb = nn.Parameter(torch.randn(self.n_relations, emb_dim))
        nn.init.xavier_uniform_(self.entity_emb)
        nn.init.xavier_uniform_(self.relation_emb)

        self.sapbert_embeddings = {}  # CUI: torch.tensor

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def encode_cuis_with_sapbert(self):
        print("Encoding UMLS CUIs with SapBERT...")
        self.model.eval()
        with torch.no_grad():
            for cui in tqdm(self.entities):
                input_text = cui
                encoded = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
                output = self.model(**encoded)
                emb = output.last_hidden_state.mean(1).squeeze(0).detach().cpu()
                self.sapbert_embeddings[cui] = emb

    def aggregate_neighbors(self, cui):
        """
        Aggregate neighbor embeddings in the KG for a given CUI.
        """
        if cui not in self.kg:
            return self.sapbert_embeddings.get(cui, torch.zeros(self.emb_dim))

        neighbors = self.kg[cui]
        neigh_embeds = []
        for nei_cui, rel in neighbors:
            if nei_cui in self.sapbert_embeddings:
                e_idx = self.entity2id.get(nei_cui, 0)
                r_idx = hash(rel) % self.n_relations
                embed = self.entity_emb[e_idx].detach().cpu() + self.relation_emb[r_idx].detach().cpu()
                neigh_embeds.append(embed)

        if neigh_embeds:
            return torch.stack(neigh_embeds).mean(0)
        else:
            return self.sapbert_embeddings.get(cui, torch.zeros(self.emb_dim))

    def forward(self, code_batch):
        """
        Input: List of code strings (e.g., ICD, ATC, etc)
        Output: Tensor [len(code_batch), emb_dim]
        """
        embeddings = []
        for code in code_batch:
            cuis = self.code2cui.get(code, [])
            cui_embeds = [self.aggregate_neighbors(cui) for cui in cuis if cui in self.sapbert_embeddings]
            if cui_embeds:
                embeddings.append(torch.stack(cui_embeds).mean(0))
            else:
                embeddings.append(torch.zeros(self.emb_dim))
        return torch.stack(embeddings).to(self.device)

    def save_code_embeddings(self, out_path):
        print(f"Saving code embeddings to {out_path}...")
        all_codes = list(self.vocab.keys())
        with torch.no_grad():
            all_embeds = self.forward(all_codes)
        torch.save({c: all_embeds[i] for i, c in enumerate(all_codes)}, out_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--code2cui', type=str, required=True)
    parser.add_argument('--kg', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)
    parser.add_argument('--output', type=str, default="code_embed.pt")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    model = CodeInitializer(
        code2cui_path=args.code2cui,
        kg_path=args.kg,
        vocab_path=args.vocab,
        device=args.device
    )
    model.encode_cuis_with_sapbert()
    model.save_code_embeddings(args.output)