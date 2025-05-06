import dill
from util import ddi_rate_score

data_path = 'records_final.pkl'
data = dill.load(open(data_path, 'rb'))
records_med = []
for patient in data:
    new_patient = []
    for visit in patient:
        new_patient.append(visit[2])
    records_med.append(new_patient)
ddi_adj_path = 'DDI.pkl'
print('DDI rate in MIMIC-III: {:.4f}'.format(ddi_rate_score(records_med, ddi_adj_path)))