import pandas as pd
from tqdm import tqdm

test_image_dir = '../data2/v2/test/'

probs = {}
probs['no_ohem'] = pd.read_csv('csvs/768_noship_prob.csv')
probs['ohem'] = pd.read_csv('csvs/768_resnet_noship_prob.csv')

for prob_name, prob_pd in probs.items():
    prob_pd.sort_values(by='ImageId', inplace=True)
    prob_pd.reset_index(drop=True, inplace=True)

avg_probs = {}

target_probs = probs['no_ohem']
total = len(target_probs)
total_probs = len(probs)

summary_probs = []

for idx in tqdm(range(total)):
    avg = 0
    var = 0

    for prob_name, prob_pd in probs.items():
        entry = prob_pd.iloc[idx]
        avg += entry['Probability']

    avg = avg / total_probs

    for prob_name, prob_pd in probs.items():
        entry = prob_pd.iloc[idx]
        var += abs(avg - entry['Probability'])

    var = var / total_probs
    summary_probs += [{'ImageId': target_probs.iloc[idx]['ImageId'], 'Average': avg, 'Variance': var}]

summary_df = pd.DataFrame(summary_probs)[['ImageId', 'Average', 'Variance']]
summary_df.to_csv('csvs/summary_resnet_noship_prob.csv', index=False)
