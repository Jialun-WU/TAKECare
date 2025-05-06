from code_initializer import CodeInit
from patient_encoder import PatRep
import collections
import torch
import torch.nn as nn
import argparse
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
import  torch.optim as optim
from torch.utils import data
from loss import cross_entropy_loss
import os
import torch.nn.functional as F
import random
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader
from data_loader import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace
from graph_data_loader import build_graph, example_graph, KGIN_data_loader
import sys
sys.path.append("..")
# from COGNet_ablation import COGNet_wo_copy, COGNet_wo_visit_score, COGNet_wo_graph, COGNet_wo_diag, COGNet_wo_procmy
from KG_models import IntentKG

from util import llprint, get_n_params, output_flatten, print_result
from recommend_iv import eval, test, test_addPath
from argparse_utils import str2bool, seed

"""adjust_learning_rate"""
def lr_poly(base_lr, iter, max_iter, power):
    if iter > max_iter:
        iter = iter % max_iter
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.lr, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = np.around(lr,5)
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

# Training settings
parser = argparse.ArgumentParser()
# parser.add_argument('--Test', action='store_true', default=True, help="test mode")
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default='IntentKG_final', help="model name")
parser.add_argument('--resume_path', type=str, default='', help='resume path')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--emb_dim', type=int, default=256, help='embedding dimension size')
parser.add_argument('--max_len', type=int, default=45, help='maximum prediction medication sequence')
parser.add_argument('--beam_size', type=int, default=1, help='max num of sentences in beam searching')
parser.add_argument('--threshold', type=float, default=0.3, help='the threshold of prediction')
parser.add_argument('--ln', type=int, default=1, help='layer normlization')
parser.add_argument('--num_heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--seed', type=seed, default='random')
parser.add_argument('--device', type=str, default='0', help='Choose GPU device')
parser.add_argument('--kgloss', type=float, default=0.0005, help='Choose GPU device')
parser.add_argument("--n_intents", type=int, default=4, help="number of latent factor for user favour")
parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")
parser.add_argument('--sim_regularity', type=float, default=1, help='regularization weight for latent factor')
parser.add_argument('--codeMI_regularity', type=float, default=1, help='regularization weight for MI loss between original and enhanced code embedding')
# ===== relation context ===== #
parser.add_argument('--context_hops', type=int, default=1, help='number of context hops')

parser.add_argument('--power', type=float, default=0.9)

args = parser.parse_args()

def test_model(model, resume_path, device, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, TOKENS, args):
    model.load_state_dict(torch.load(open(resume_path, 'rb'), map_location='cpu'))
    model.to(device=device)
    tic = time.time()
    smm_record, ja, prauc, precision, recall, f1, med_num = test_addPath(model, resume_path, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, 0, device, TOKENS, args)
    result = []
    for _ in range(10):
        data_num = len(ja)
        final_length = int(0.8 * data_num)
        idx_list = list(range(data_num))
        random.shuffle(idx_list)
        idx_list = idx_list[:final_length]
        avg_ja = np.mean([ja[i] for i in idx_list])
        avg_prauc = np.mean([prauc[i] for i in idx_list])
        avg_precision = np.mean([precision[i] for i in idx_list])
        avg_recall = np.mean([recall[i] for i in idx_list])
        avg_f1 = np.mean([f1[i] for i in idx_list])
        avg_med = np.mean([med_num[i] for i in idx_list])
        cur_smm_record = [smm_record[i] for i in idx_list]
        result.append([avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med])
        llprint('\nJaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
                avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med))
    result = np.array(result)
    mean = result.mean(axis=0)
    std = result.std(axis=0)

    outstring = ""
    for m, s in zip(mean, std):
        outstring += "{:.4f} + {:.4f}\t".format(m, s)

    print (outstring)
    result_saved_path = os.path.dirname(os.path.dirname(resume_path))
    with open(result_saved_path+'/seed_{}_test_result.txt'.format(args.seed), 'a') as f:
        f.write('\nmodel_path:\n{}\n'.format(resume_path))
        f.write('Jaccard: {:.4f} + {:.4f} \n'.format(mean[0], std[0]))
        f.write('AVG_F1: {:.4f} + {:.4f} \n'.format(mean[4], std[4]))
        f.write('PRAUC: {:.4f} + {:.4f} \n'.format(mean[1], std[1]))
        f.write('AVG_PRC: {:.4f} + {:.4f} \n'.format(mean[2], std[2]))
        f.write('AVG_RECALL: {:.4f} + {:.4f} \n'.format(mean[3], std[3]))
        f.write('AVG_MED: {:.4f} + {:.4f}\n'.format(mean[5], std[5]))
    print ('test time: {}'.format(time.time() - tic))
    return 

def test(model, device, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, TOKENS, args):
    model.to(device=device)
    tic = time.time()
    smm_record, ja, prauc, precision, recall, f1, med_num = test(model, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, 0, device, TOKENS, args)
    result = []
    for _ in range(10):
        data_num = len(ja)
        final_length = int(0.8 * data_num)
        idx_list = list(range(data_num))
        random.shuffle(idx_list)
        idx_list = idx_list[:final_length]
        avg_ja = np.mean([ja[i] for i in idx_list])
        avg_prauc = np.mean([prauc[i] for i in idx_list])
        avg_precision = np.mean([precision[i] for i in idx_list])
        avg_recall = np.mean([recall[i] for i in idx_list])
        avg_f1 = np.mean([f1[i] for i in idx_list])
        avg_med = np.mean([med_num[i] for i in idx_list])
        result.append([avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med])
        llprint('\nJaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
                avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med))
    result = np.array(result)
    mean = result.mean(axis=0)
    std = result.std(axis=0)

    outstring = ""
    for m, s in zip(mean, std):
        outstring += "{:.4f} + {:.4f}\t".format(m, s)

    print (outstring)        
        
    print ('test time: {}'.format(time.time() - tic))
    return result

def main(args):
    # load data
    data_path = "../data/mimic/output/records.pkl"
    voc_path = "../data/mimic/output/voc.pkl"

    # UMLS path
    kg_path = '../data/mimic/data/kg.pkl'
    kg_voc_path = '../data/mimic/data/code2cui.pkl'

    device = torch.device('cuda:{}'.format(args.device))
    # device = torch.device('cuda:{}'.format(0))
    args.device = device
    
    set_random_seed(args.seed)

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    ehr_adj = None # dill.load(open(ehr_adj_path, 'rb'))
    kg = dill.load(open(kg_path, 'rb'))
    kg_voc = dill.load(open(kg_voc_path, 'rb'))
    diag2cui, proc2cui, med2cui, kg2idx, rel2idx = kg_voc['diag2cui'], kg_voc['proc2cui'], kg_voc['med2cui'], kg_voc['kg2id'], kg_voc['rel2id']
    print("UMLS triple num:{}".format(len(kg)))
    print("UMLS node num: {}".format(len(list(kg2idx.kg2idx.keys()))))

    n_nodes, n_entities, n_leaf_nodes, kg_graph, adj_mat_list, norm_mat_list, mean_mat_list = data_loader(kg, kg_voc, voc)
    n_relations = len(rel2idx.kg2idx)

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    # frequency statistic
    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] += 1
    
    ## rare first
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_medications = sorted(data[i][j][2], key=lambda x:med_count[x])
            data[i][j][2] = cur_medications


    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    train_dataset = mimic_data(data_train)
    eval_dataset = mimic_data(data_eval)
    test_dataset = mimic_data(data_test)
    total_dataset = mimic_data(data)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch_v2_train)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch_v2_eval)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval)
    
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    model = IntentKG(voc_size, ehr_adj, n_nodes, args.n_intents, n_relations, n_entities, kg_graph, \
                     mean_mat_list[0], sim_regularity=args.sim_regularity, codeMI_regularity=args.codeMI_regularity,\
                        emb_dim=args.emb_dim, context_hops=args.context_hops, ind=args.ind, node_dropout_rate=args.node_dropout_rate, \
                            mess_dropout_rate=args.mess_dropout_rate, device=device, dim_hidden=args.emb_dim, ln=args.ln, kgloss_alpha=args.kgloss, num_heads=args.num_heads)
    
    

    if args.Test:
        args.seed = os.path.split(args.resume_path)[-1].split('_')[1]
        test_model(model, args.resume_path, device, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, TOKENS, args)
        return
    
    else:
        wandb.init(project="IntentKG_22_mimic-iv")

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=args.lr) #, weight_decay=0.01

    # args.power = 0.9
    args.num_steps = 50000  #50000
    args.weight_decay = 0.0005
    args.momentum = 0.9

    history = defaultdict(list)
    test_history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 200
    cu_iter = 0

    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} seed : {} bs:{} model : {} --------------------------'.format(epoch, args.seed, args.batch_size, args.model_name))
        model.train()
        for idx, data in enumerate(train_dataloader):
            diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, target_list = data

            diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
            procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
            dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
            stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
            dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
            stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
            # medications = medications.to(device)
            medications = pad_num_replace(medications, -1, MED_PAD_TOKEN).to(device)
            m_mask_matrix = m_mask_matrix.to(device)
            d_mask_matrix = d_mask_matrix.to(device)
            p_mask_matrix = p_mask_matrix.to(device)
            dec_disease_mask = dec_disease_mask.to(device)
            stay_disease_mask = stay_disease_mask.to(device)
            dec_proc_mask = dec_proc_mask.to(device)
            stay_proc_mask = stay_proc_mask.to(device)

            adjust_learning_rate(optimizer, cu_iter, args)
            cu_iter += 1
            output_logits, _, code_mi_loss, intent_cor = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
            bce_target = np.zeros([medications.shape[0], medications.shape[1], voc_size[2]])
            for b_i, med in enumerate(target_list):
                for v_i, m in enumerate(med):
                    bce_target[b_i, v_i, m] = 1
            labels = torch.Tensor(bce_target).to(device)
            loss = F.binary_cross_entropy_with_logits(output_logits, labels)

            loss = loss + code_mi_loss + intent_cor
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            llprint('\rtraining step: {} / {} loss:{:.4f} lr:{} sample_len:{}'.format(idx, len(train_dataloader), loss.item(), optimizer.param_groups[0]['lr'], diseases.shape))        
        

        print ()
        tic2 = time.time()
        ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, eval_dataloader, voc_size, epoch, device, TOKENS, args)
        print ('training time: {}, eval time: {}'.format(time.time() - tic, time.time() - tic2))

        history['ja'].append(ja)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)
        test_result = test_1(model, device, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, TOKENS, args)
        mean = test_result.mean(axis=0)
        std = test_result.std(axis=0)
        test_history['ja'].append(mean[0])
        test_history['avg_p'].append(mean[2])
        test_history['avg_r'].append(mean[3])
        test_history['avg_f1'].append(mean[4])
        test_history['prauc'].append(mean[1])
        test_history['med'].append(mean[5])
        wandb.log({"loss": loss,
                    "e_Jaccard_max":max((history['ja'])), 
                    "e_prauc_max":max(history['prauc']),
                    "e_avg_f1_max": max(history['avg_f1']),
                    "e_Jaccard":ja,
                    "e_prauc":prauc,
                    "e_avg_f1": avg_f1,
                    "e_avg_p":avg_p,
                    "e_avg_r":avg_r,
                    "t_Jaccard":mean[0],
                    "t_F1":mean[4], 
                    "t_PRAUC":mean[1], 
                    "t_MED":mean[5], 
                    "t_Jaccard_std":std[0],
                    "t_F1_std":std[4], 
                    "t_PRAUC_std":std[1], 
                    "t_MED_std":std[5],
                    "t_Jaccard_max":max(test_history['ja']),
                    "t_F1_max":max(test_history['avg_f1']), 
                    "t_PRAUC_max":max(test_history['prauc']), 
                    "t_MED_min":min(test_history['med'])
                    })

        if epoch >= 5:
            print ('Med: {}, Ja: {}, F1: {} PRAUC: {}'.format(
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
                ))

        saved_path = os.path.join("./saved_results/iv/{}".format(args.model_name), 'Bs_{}'.format(args.batch_size), 'lr_{}_hs_{}_ln_{}_kg_{}_intent_{}_ind_{}_CodeMI_{}_InCor_{}_ndr_{}_nhead{}'.format(args.lr, args.emb_dim, int(args.ln), args.kgloss, args.n_intents, args.ind, args.codeMI_regularity, args.sim_regularity, args.node_dropout_rate, args.num_heads), 'Seed_{}'.format(args.seed))
          
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        if best_ja < mean[1]: #if best_ja < ja: # 
            best_epoch = epoch
            best_ja = mean[1]
            best_saved_model = os.path.join(saved_path, \
            'Seed_{}_Epoch_{}_JA_{:.4}.model'.format(args.seed, epoch, ja))
            torch.save(model.state_dict(), open(best_saved_model, 'wb'))
            test_model(model, best_saved_model, device, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, TOKENS, args)
        print ('best_epoch: {}'.format(best_epoch))
        
        if epoch - best_epoch > 10:
            # test_model(model, best_saved_model, device, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, TOKENS, args)
            break


if __name__ == '__main__':
    main(args)


class DrugEncoder(nn.Module):
    def __init__(self, dim, num_drugs):
        super(DrugEncoder, self).__init__()
        self.gat_c = GATv2Conv(dim, dim)
        self.gat_i = GATv2Conv(dim, dim)
        self.w_m = nn.Parameter(torch.ones(1))
        self.drug_embed = nn.Parameter(torch.randn(num_drugs, dim))

    def forward(self, edge_index_c, edge_index_i):
        e_c = self.gat_c(self.drug_embed, edge_index_c)
        e_i = self.gat_i(self.drug_embed, edge_index_i)
        return e_c + self.w_m * e_i


class ClinicalPredictor(nn.Module):
    """
    Task heads for multi-label classification: diagnosis & prescription.
    """
    def __init__(self, input_dim, med_embed_dim):
        super().__init__()
        self.diagnosis_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, med_embed_dim)
        )
        self.prescription_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, med_embed_dim)
        )

    def forward(self, patient_emb, med_emb):
        diag_logits = torch.matmul(self.diagnosis_head(patient_emb), med_emb.T)
        presc_logits = torch.matmul(self.prescription_head(patient_emb), med_emb.T)
        return diag_logits, presc_logits


class TAKECare(nn.Module):
    """
    Full framework: Code init + Patient encoding + Prediction.
    """
    def __init__(self, concept_texts, path_dict, input_dim, med_embed_dim):
        super().__init__()
        self.code_initializer = CodeInit(concept_texts, path_dict, hidden_size=med_embed_dim)
        self.patient_encoder = PatRep(input_dim=med_embed_dim)
        self.predictor = ClinicalPredictor(med_embed_dim, med_embed_dim)
        self.drug_encoder = DrugEncoder(med_embed_dim)

    def forward(self, ehr_data, ance_feats, ance_mask, drug_co, drug_int):
        # --- Stage 1 ---
        med_embeds, mi_loss = self.code_initializer()

        # --- Stage 2 ---
        patient_emb = self.patient_encoder(
            code_embeddings=med_embeds,
            visit_sets=ehr_data['visits'],
            visit_times=ehr_data['times'],
            ance_features=ance_feats,
            ance_mask=ance_mask
        )

        # --- Stage 3 ---
        diag_logits, presc_logits = self.predictor(patient_emb, med_embeds)
        drug_embeds = self.drug_encoder(drug_co, drug_int)

        return {
            'diag_logits': diag_logits,
            'presc_logits': presc_logits,
            'med_embeds': med_embeds,
            'drug_embeds': drug_embeds,
            'mi_loss': mi_loss,
            'patient_emb': patient_emb
        }

def train(model, data_loader, optimizer, device, args):
    model.train()
    tracker = MetricTracker()

    for batch in data_loader:
        ehr_data = batch['ehr']
        ance_feat = batch['ance_feat'].to(device)
        ance_mask = batch['ance_mask'].to(device)
        drug_co = batch['drug_co'].to(device)
        drug_int = batch['drug_int'].to(device)
        labels_diag = batch['labels_diag'].to(device)
        labels_presc = batch['labels_presc'].to(device)

        out = model(ehr_data, ance_feat, ance_mask, drug_co, drug_int)
        loss_diag = F.binary_cross_entropy_with_logits(out['diag_logits'], labels_diag)
        loss_presc = F.binary_cross_entropy_with_logits(out['presc_logits'], labels_presc)
        loss_ddi = compute_ddi_loss(out['presc_logits'], batch['ddi_index'], args.ddi_weight)

        loss = loss_diag + loss_presc + args.alpha * out['mi_loss'] + args.beta * loss_ddi

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update('total_loss', loss.item())
        tracker.update('loss_diag', loss_diag.item())
        tracker.update('loss_presc', loss_presc.item())
        tracker.update('loss_ddi', loss_ddi.item())

    return tracker.average()


def evaluate(model, data_loader, device):
    model.eval()
    tracker = MetricTracker()
    with torch.no_grad():
        for batch in data_loader:
            ehr_data = batch['ehr']
            ance_feat = batch['ance_feat'].to(device)
            ance_mask = batch['ance_mask'].to(device)
            drug_co = batch['drug_co'].to(device)
            drug_int = batch['drug_int'].to(device)
            labels_diag = batch['labels_diag'].to(device)
            labels_presc = batch['labels_presc'].to(device)

            out = model(ehr_data, ance_feat, ance_mask, drug_co, drug_int)
            diag_pred = torch.sigmoid(out['diag_logits']) > 0.5
            presc_pred = torch.sigmoid(out['presc_logits']) > 0.5

            acc_diag = (diag_pred == labels_diag).float().mean().item()
            acc_presc = (presc_pred == labels_presc).float().mean().item()
            tracker.update('acc_diag', acc_diag)
            tracker.update('acc_presc', acc_presc)

    return tracker.average()
