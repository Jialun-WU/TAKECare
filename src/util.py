from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import warnings
import dill
from collections import Counter
from rdkit import Chem
from collections import defaultdict
import torch
warnings.filterwarnings('ignore')

def nfold_experiment(mimic3sample, epochs , ds_size_ratio, print_results=True, record_results=True):

    data = mimic3sample.samples
    co_occurrence_counts, groups1 = get_group_labels1(data)

    seeds = [123, 321, 54, 65, 367]


    list_top_k = [3,5,7,10, 15, 20, 30]
    metrics_dict = {'roc_auc_samples': [], 'pr_auc_samples': [], 'f1_samples': []}

    for group_name in groups1.keys():
        metrics_dict[f'roc_auc_samples_{group_name}'] = []
        metrics_dict[f'pr_auc_samples_{group_name}'] = []


    for k in list_top_k:
        metrics_dict[f'acc_at_k={k}'] = []
        metrics_dict[f'hit_at_k={k}'] = []
        for group_name in groups1.keys():
            metrics_dict[f'Group_acc_at_k={k}@' + group_name] = []
            metrics_dict[f'Group_hit_at_k={k}@' + group_name] = []


    for seed in seeds:
        print(f'----------------------seed:{seed}-----------------------')

        torch.manual_seed(seed)
        np.random.seed(seed)
        #random.seed(seed)
        # Set seed for CUDA operations
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        train_loader, val_loader, test_loader = preprocessing_seq_diag_pred(
            mimic3sample, train_ratio=0.8, val_ratio=0.2, test_ratio=0, batch_size=252, print_stats=False, seed=seed
        )
        print('preprocessing done!')

        # Stage 3: define model
        device = "cuda:0"

        if ds_size_ratio==1.0:
            ds_size_ratio_model = ''
        else:
            ds_size_ratio_model = '_' + str(ds_size_ratio)

        model = Mega(
            dataset=mimic3sample,
            feature_keys=["conditions", "drugs", "procedures"],
            label_key="label",
            mode="multilabel",
            embedding_dim=128,dropout=0.5,nheads=1,nlayers=1,
            G_dropout=0.1,n_G_heads=4,n_G_layers=1,
            threshold3=0.00, threshold2=0.02, threshold1=0.00,
            n_hap_layers=1, n_hap_heads=2, hap_dropout=0.2,
            llm_model='text-embedding-3-small', gpt_embd_path='../saved_files/gpt_code_emb/tx-emb-3-small/include_all_parents2/', #gpt_embd_path='../saved_files/gpt_code_emb/tx-emb-3-small/' => so far best results
            ds_size_ratio=ds_size_ratio_model,device=device, seed=seed,
        )
        model.to(device)


        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = torch.nn.DataParallel(model)

        model.to(device)

        # Stage 4: model training

        trainer = Trainer(model=model,
                          checkpoint_path=None,
                          metrics = ['roc_auc_samples', 'pr_auc_samples', 'f1_samples'],
                          enable_logging=True,
                          output_path=f"./output/OntoFAR_{ds_size_ratio}",
                          exp_name=f'EXP_:seed:{seed}',
                          device=device)

        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            optimizer_class =  torch.optim.Adam,
            optimizer_params = {"lr": 1e-3},
            weight_decay=0.0,
            monitor="pr_auc_samples",
            monitor_criterion='max',
            load_best_model_at_last=True
        )


        all_metrics = [

            "pr_auc_samples",
            "roc_auc_samples",
            "f1_samples",
        ]

        y_true, y_prob, loss = trainer.inference(val_loader)

        result = evaluate(y_true, y_prob, co_occurrence_counts, groups1, list_top_k=list_top_k, all_metrics=all_metrics)

        print ('\n', result)

        metrics_dict['pr_auc_samples'].append(result['pr_auc_samples'])
        metrics_dict['roc_auc_samples'].append(result['roc_auc_samples'])
        metrics_dict['f1_samples'].append(result['f1_samples'])

        for group_name in groups1.keys():
            metrics_dict[f'roc_auc_samples_{group_name}'].append(result[f'roc_auc_samples_{group_name}'])
            metrics_dict[f'pr_auc_samples_{group_name}'].append(result[f'pr_auc_samples_{group_name}'])


        for k in list_top_k:
            metrics_dict[f'acc_at_k={k}'].append(result[f'acc_at_k={k}'])
            metrics_dict[f'hit_at_k={k}'].append(result[f'hit_at_k={k}'])
            for group_name in groups1.keys():
                metrics_dict[f'Group_acc_at_k={k}@' + group_name].append(result[f'Group_acc_at_k={k}@' + group_name])
                metrics_dict[f'Group_hit_at_k={k}@' + group_name].append(result[f'Group_hit_at_k={k}@' + group_name])

    if print_results:
        print()
        print('mean pr_auc_samples:', np.mean(metrics_dict['pr_auc_samples']))
        print('max pr_auc_samples:', np.max(metrics_dict['pr_auc_samples']))
        print('min pr_auc_samples:', np.min(metrics_dict['pr_auc_samples']))
        print('CI pr_auc_samples:', calculate_confidence_interval(metrics_dict['pr_auc_samples']))

        print()

        print('mean roc_auc_samples:', np.mean(metrics_dict['roc_auc_samples']))
        print('max roc_auc_samples:', np.max(metrics_dict['roc_auc_samples']))
        print('min roc_auc_samples:', np.min(metrics_dict['roc_auc_samples']))
        print('CI roc_auc_samples:', calculate_confidence_interval(metrics_dict['roc_auc_samples']))
        print()

        print('mean f1_samples:', np.mean(metrics_dict['f1_samples']))
        print('max f1_samples:', np.max(metrics_dict['f1_samples']))
        print('min f1_samples:', np.min(metrics_dict['f1_samples']))
        print('CI f1_samples:', calculate_confidence_interval(metrics_dict['f1_samples']))
        print()

        for group_name in groups1:
            print()
            print(f'mean pr_auc_samples_{group_name}:', np.mean(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'max pr_auc_samples_{group_name}:', np.max(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'min pr_auc_samples_{group_name}:', np.min(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'CI pr_auc_samples_{group_name}:',
                  calculate_confidence_interval(metrics_dict[f'pr_auc_samples_{group_name}']))
            print()

            print(f'mean roc_auc_samples_{group_name}:', np.mean(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'max roc_auc_samples_{group_name}:', np.max(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'min roc_auc_samples_{group_name}:', np.min(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'CI roc_auc_samples_{group_name}:',
                  calculate_confidence_interval(metrics_dict[f'roc_auc_samples_{group_name}']))
            print()

        for k in list_top_k:
            print('------------------------------------------')

            print(f'mean acc_at_k={k}:', np.mean(metrics_dict[f'acc_at_k={k}']))
            print(f'max acc_at_k={k}:', np.max(metrics_dict[f'acc_at_k={k}']))
            print(f'min acc_at_k={k}:', np.min(metrics_dict[f'acc_at_k={k}']))
            print(f'CI acc_at_k={k}:', calculate_confidence_interval(metrics_dict[f'acc_at_k={k}']))
            print()

            print(f'mean hit_at_k={k}:', np.mean(metrics_dict[f'hit_at_k={k}']))
            print(f'max hit_at_k={k}:', np.max(metrics_dict[f'hit_at_k={k}']))
            print(f'min hit_at_k={k}:', np.min(metrics_dict[f'hit_at_k={k}']))
            print(f'CI hit_at_k={k}:', calculate_confidence_interval(metrics_dict[f'hit_at_k={k}']))
            print()

            for group_name in groups1:
                print(f'mean Group_acc_at_k={k}@{group_name}:',
                      np.mean(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'max Group_acc_at_k={k}@{group_name}:',
                      np.max(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'min Group_acc_at_k={k}@{group_name}:',
                      np.min(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'CI Group_acc_at_k={k}@{group_name}:',
                      calculate_confidence_interval(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print()

                print(f'mean Group_hit_at_k={k}@{group_name}:',
                      np.mean(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'max Group_hit_at_k={k}@{group_name}:',
                      np.max(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'min Group_hit_at_k={k}@{group_name}:',
                      np.min(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'CI Group_hit_at_k={k}@{group_name}:',
                      calculate_confidence_interval(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print()

    if record_results:
        with open(f'results_prompting/metrics_results_BestModel_OntoFAR_{ds_size_ratio}.txt', 'w') as file:
            file.write('\n')
            file.write(f'mean pr_auc_samples: {np.mean(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'max pr_auc_samples: {np.max(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'min pr_auc_samples: {np.min(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'CI pr_auc_samples: {calculate_confidence_interval(metrics_dict["pr_auc_samples"])}\n')
            file.write('\n')

            file.write(f'mean roc_auc_samples: {np.mean(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'max roc_auc_samples: {np.max(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'min roc_auc_samples: {np.min(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'CI roc_auc_samples: {calculate_confidence_interval(metrics_dict["roc_auc_samples"])}\n')
            file.write('\n')

            file.write(f'mean f1_samples: {np.mean(metrics_dict["f1_samples"])}\n')
            file.write(f'max f1_samples: {np.max(metrics_dict["f1_samples"])}\n')
            file.write(f'min f1_samples: {np.min(metrics_dict["f1_samples"])}\n')
            file.write(f'CI f1_samples: {calculate_confidence_interval(metrics_dict["f1_samples"])}\n')
            file.write('\n')

            for group_name in groups1:
                file.write('\n')
                file.write(
                    f'mean pr_auc_samples_{group_name}: {np.mean(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(
                    f'max pr_auc_samples_{group_name}: {np.max(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(
                    f'min pr_auc_samples_{group_name}: {np.min(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(
                    f'CI pr_auc_samples_{group_name}: {calculate_confidence_interval(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write('\n')

                file.write(
                    f'mean roc_auc_samples_{group_name}: {np.mean(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(
                    f'max roc_auc_samples_{group_name}: {np.max(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(
                    f'min roc_auc_samples_{group_name}: {np.min(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(
                    f'CI roc_auc_samples_{group_name}: {calculate_confidence_interval(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write('\n')

            for k in list_top_k:
                file.write('------------------------------------------\n')

                file.write(f'mean acc_at_k={k}: {np.mean(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'max acc_at_k={k}: {np.max(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'min acc_at_k={k}: {np.min(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'CI acc_at_k={k}: {calculate_confidence_interval(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write('\n')

                file.write(f'mean hit_at_k={k}: {np.mean(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'max hit_at_k={k}: {np.max(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'min hit_at_k={k}: {np.min(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'CI hit_at_k={k}: {calculate_confidence_interval(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write('\n')

                for group_name in groups1:
                    file.write(
                        f'mean Group_acc_at_k={k}@{group_name}: {np.mean(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'max Group_acc_at_k={k}@{group_name}: {np.max(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'min Group_acc_at_k={k}@{group_name}: {np.min(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'CI Group_acc_at_k={k}@{group_name}: {calculate_confidence_interval(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write('\n')

                    file.write('------------------------------------------\n')

                    file.write(
                        f'mean Group_hit_at_k={k}@{group_name}: {np.mean(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'max Group_hit_at_k={k}@{group_name}: {np.max(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'min Group_hit_at_k={k}@{group_name}: {np.min(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'CI Group_hit_at_k={k}@{group_name}: {calculate_confidence_interval(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write('\n')

    return

class InfoNCEMI(torch.nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super().__init__()
        self.p_mu = torch.nn.Sequential(
            torch.nn.Linear(x_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, y_dim)
        )
        self.p_logvar = torch.nn.Sequential(
            torch.nn.Linear(x_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, y_dim),
            torch.nn.Tanh()
        )

    def forward(self, x_samples, y_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()
        negative = - (mu.unsqueeze(1) - y_samples.unsqueeze(0)) ** 2 / 2. / logvar.exp().unsqueeze(1)
        upper_bound = positive.sum(dim=1).mean() - negative.sum(dim=2).mean()
        return upper_bound


def compute_MI(x1: torch.Tensor, x2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    sim_matrix = torch.matmul(x1, x2.T) / tau
    labels = torch.arange(x1.size(0)).to(x1.device)
    loss_i = F.cross_entropy(sim_matrix, labels)
    loss_j = F.cross_entropy(sim_matrix.T, labels)
    return (loss_i + loss_j) / 2


def compute_code_frequencies(data, key="diag"):
    counter = Counter()
    for patient in data:
        for visit in patient['visits']:
            counter.update(visit.get(key, []))
    return counter


def generate_visit_mask(visits, pad_token=0):
    max_len = max(len(v) for v in visits)
    mask = [[1]*len(v) + [0]*(max_len - len(v)) for v in visits]
    return torch.tensor(mask, dtype=torch.bool)


def convert_to_sequence(data, vocab):
    sequences = []
    for patient in data:
        patient_seq = []
        for visit in patient['visits']:
            encoded_visit = [vocab[code] for code in visit.get('diag', []) if code in vocab]
            patient_seq.append(encoded_visit)
        sequences.append(patient_seq)
    return sequences


class EHRDataset(torch.utils.data.Dataset):
    def __init__(self, patient_records):
        self.data = patient_records

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def split_by_patient(data, ratios=(0.7, 0.1, 0.2), seed=42):
    set_seed(seed)
    n = len(data)
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    return [data[i] for i in train_idx], [data[i] for i in val_idx], [data[i] for i in test_idx]


def build_medkg_graph(ehr_data, code2cui, cui_relations):
    nodes = set()
    edges = []
    for cui1, rel, cui2 in cui_relations:
        nodes.add(cui1)
        nodes.add(cui2)
        edges.append((cui1, cui2, rel))

    node2idx = {node: i for i, node in enumerate(sorted(nodes))}
    edge_index = [(node2idx[c1], node2idx[c2]) for c1, c2, _ in edges]
    return {
        'node2idx': node2idx,
        'edge_index': torch.tensor(edge_index).t().contiguous(),
        'relations': [rel for _, _, rel in edges]
    }

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def transform_split(X, Y):
    x_train, x_eval, y_train, y_eval = train_test_split(X, Y, train_size=2/3, random_state=1203)
    x_eval, x_test, y_eval, y_test = train_test_split(x_eval, y_eval, test_size=0.5, random_state=1203)
    return x_train, x_eval, x_test, y_train, y_eval, y_test

def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]  
    out_list = []   
    break_flag = False
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)]
    return out_list, sorted_predict


def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b]==1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score


    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score


    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score


    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            # try:
            #     all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
            # except:
            #     continue
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)
    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0
    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def sequence_metric_v2(y_gt, y_pred, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b]==1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score


    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score


    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score


    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)
    f1 = f1(y_gt, y_pred)
    # prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def ddi_rate_score(record, path='/home/lsc/model/lsc_code/useGeneration2Drug/COGNet/data/ddi_A_final.pkl'):
    # ddi rate
    ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt


def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)


def buildMPNN(molecule, med_voc, radius=1, device="cpu:0"):

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    MPNNSet, average_index = [], []

    print (len(med_voc.items()))
    for index, ndc in med_voc.items():

        smilesList = list(molecule[ndc])

        """Create each data with the above defined functions."""
        counter = 0 # counter how many drugs are under that ATC-3
        for smiles in smilesList:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                    fingerprint_dict, edge_dict)
                adjacency = Chem.GetAdjacencyMatrix(mol)
                # if fingerprints.shape[0] == adjacency.shape[0]:
                for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                    fingerprints = np.append(fingerprints, 1)
                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                MPNNSet.append((fingerprints, adjacency, molecular_size))
                counter += 1
            except:
                continue
        average_index.append(counter)

        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """

    N_fingerprint = len(fingerprint_dict)

    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        average_projection[i, col_counter : col_counter + item] = 1 / item
        col_counter += item

    return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)

def print_result(label, prediction):
    '''
    label: [real_med_num, ]
    logits: [20, med_vocab_size]
    '''
    label_text = " ".join([str(x) for x in label])
    predict_text = " ".join([str(x) for x in prediction])
    
    return "[GT]\t{}\n[PR]\t{}\n\n".format(label_text, predict_text)


from satsolver import *
def Post_DDI(pred_result, ddi_pair, ehr_train_pair):
    # input: numpy (seq, voc_size) (0~1)
    post_result, y_pred = np.zeros_like(pred_result), np.zeros_like(pred_result)
    y_pred[pred_result >= 0.5] = 1
    for k in range(pred_result.shape[0]):
        pred_idx = np.nonzero(y_pred[k])[0]
        tmp_dict = {idx:n for n, idx in enumerate(pred_idx)}
        pred_prob = {str(n):pred_result[k, idx] for n, idx in enumerate(pred_idx)}
        formula = two_cnf(pred_prob)
        ddi_list = []
        for (i, j) in ddi_pair:
            if i in pred_idx and j in pred_idx and i<j:
                if (i, j) not in ehr_train_pair:
                    # print(['~' + str(tmp_dict[i]), '~' + str(tmp_dict[j])])
                    formula.add_clause(['~' + str(tmp_dict[i]), '~' + str(tmp_dict[j])])
                    if i not in ddi_list:
                        ddi_list.append(i)
                    if j not in ddi_list:
                        ddi_list.append(j)
        f = two_sat_solver(formula)
        if f:
            pos = [list(tmp_dict.keys())[int(n)] for n, x in f.items() if x == 1] + [idx for idx in pred_idx if idx not in ddi_list]
            post_result[k, pos] = 1
        else:
            post_result[k] = pred_result[k]

    return post_result

