import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import copy

def load_and_process_discrete_data_multiclass():
    d1 = pd.read_csv("student-mat.csv", sep=";")
    d2 = pd.read_csv("student-por.csv", sep=";")

    merge_cols = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason",
                  "nursery", "internet"]
    df = pd.merge(d1, d2, on=merge_cols, suffixes=('_mat', '_por'))

    df['G3_avg'] = (df['G3_mat'] + df['G3_por']) / 2

    def discretize_g3(g):
        g = round(g)
        if g < 10:
            return 0
        elif g < 15:
            return 1
        else:
            return 2

    df['G3_cat'] = df['G3_avg'].apply(discretize_g3)

    df['studytime'] = ((df['studytime_mat'] + df['studytime_por']) / 2).round().astype(int).clip(1, 4) - 1
    df['failures'] = ((df['failures_mat'] + df['failures_por']) / 2).round().astype(int).clip(0, 3)

    df['absences_val'] = (df['absences_mat'] + df['absences_por']) / 2

    def discretize_absences(a):
        a = round(a)
        if a == 0:
            return 0
        elif a <= 5:
            return 1
        elif a <= 15:
            return 2
        else:
            return 3

    df['absences'] = df['absences_val'].apply(discretize_absences)

    df['internet'] = df['internet'].map({'no': 0, 'yes': 1})

    h_mat = df['higher_mat'].apply(lambda x: 1 if x == 'yes' else 0)
    h_por = df['higher_por'].apply(lambda x: 1 if x == 'yes' else 0)
    df['higher'] = ((h_mat + h_por) / 2).round().astype(int)

    return {
        'internet': df['internet'].values,
        'studytime': df['studytime'].values,
        'failures': df['failures'].values,
        'higher': df['higher'].values,
        'absences': df['absences'].values,
        'G3': df['G3_cat'].values
    }

class ImprovedDiscreteBN_EM:
    def __init__(self, data_dict, z_dim=4, max_iter=50, tol=1e-5, seed=42):
        self.data = data_dict
        self.max_iter = max_iter
        self.tol = tol
        self.z_dim = z_dim
        self.n_samples = len(data_dict['internet'])

        self.history = {'delta': []}

        np.random.seed(seed)

        self.dims = {
            'internet': 2,
            'Z': self.z_dim,
            'studytime': 4,
            'failures': 4,
            'higher': 2,
            'absences': 4,
            'G3': 3
        }

        self.params = self._init_cpts()

    def _init_cpts(self):
        p = {}
        p['P_G3_Z'] = self._norm_rand((self.dims['Z'], 3))
        p['P_Z_I'] = self._norm_rand((2, self.dims['Z']))
        p['P_S_Z'] = self._norm_rand((self.dims['Z'], 4))
        p['P_F_Z'] = self._norm_rand((self.dims['Z'], 4))
        p['P_H_Z'] = self._norm_rand((self.dims['Z'], 2))
        p['P_A_Z'] = self._norm_rand((self.dims['Z'], 4))
        return p

    def _norm_rand(self, shape):
        m = np.random.rand(*shape) + 0.1
        return m / m.sum(axis=-1, keepdims=True)

    def e_step(self, data):
        N = len(data['internet'])
        log_p = np.zeros((N, self.dims['Z']))

        log_p += np.log(self.params['P_Z_I'][data['internet']])
        log_p += np.log(self.params['P_S_Z'][:, data['studytime']].T)
        log_p += np.log(self.params['P_F_Z'][:, data['failures']].T)
        log_p += np.log(self.params['P_H_Z'][:, data['higher']].T)
        log_p += np.log(self.params['P_A_Z'][:, data['absences']].T)

        if 'G3' in data:
            log_p += np.log(self.params['P_G3_Z'][:, data['G3']].T)

        max_log = log_p.max(axis=1, keepdims=True)
        exp_log = np.exp(log_p - max_log)
        return exp_log / exp_log.sum(axis=1, keepdims=True)

    def m_step(self, resp, data):
        alpha = 0.5

        counts = np.zeros((2, self.dims['Z']))
        for i in [0, 1]:
            mask = (data['internet'] == i)
            if np.any(mask): counts[i] = resp[mask].sum(axis=0)
        self.params['P_Z_I'] = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * self.dims['Z'])

        def update(name, vec, c_dim):
            counts = np.zeros((self.dims['Z'], c_dim))
            for v in range(c_dim):
                mask = (vec == v)
                if np.any(mask): counts[:, v] = resp[mask].sum(axis=0)
            self.params[name] = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * c_dim)

        update('P_S_Z', data['studytime'], 4)
        update('P_F_Z', data['failures'], 4)
        update('P_H_Z', data['higher'], 2)
        update('P_A_Z', data['absences'], 4)

        if 'G3' in data:
            update('P_G3_Z', data['G3'], 3)

    def fit(self):
        prev_params = copy.deepcopy(self.params)
        for i in range(self.max_iter):
            self.m_step(self.e_step(self.data), self.data)
            max_delta = 0.0
            for key in self.params:
                diff = np.max(np.abs(self.params[key] - prev_params[key]))
                if diff > max_delta:
                    max_delta = diff

            self.history['delta'].append(max_delta)

            if max_delta < self.tol:
                print(f"Converged at iteration {i}")
                break
            prev_params = copy.deepcopy(self.params)

    def predict(self, test_data):
        N = len(test_data['internet'])
        log_p = np.zeros((N, self.dims['Z']))

        log_p += np.log(self.params['P_Z_I'][test_data['internet']])
        log_p += np.log(self.params['P_S_Z'][:, test_data['studytime']].T)
        log_p += np.log(self.params['P_F_Z'][:, test_data['failures']].T)
        log_p += np.log(self.params['P_H_Z'][:, test_data['higher']].T)
        log_p += np.log(self.params['P_A_Z'][:, test_data['absences']].T)

        max_log = log_p.max(axis=1, keepdims=True)
        p_z = np.exp(log_p - max_log) / np.exp(log_p - max_log).sum(axis=1, keepdims=True)

        p_g3 = np.dot(p_z, self.params['P_G3_Z'])

        pred_cats = np.argmax(p_g3, axis=1)

        return pred_cats, p_g3

def analyze_factors(em, data_raw):
    results = {}
    p_internet = np.bincount(data_raw['internet'], minlength=2) / len(data_raw['internet'])
    p_z_given_i = em.params['P_Z_I']
    p_z_prior = np.dot(p_internet, p_z_given_i)

    expected_cat_given_z = np.sum(em.params['P_G3_Z'] * np.arange(3), axis=1)

    res_internet = []
    labels_internet = ['No', 'Yes']
    for val in [0, 1]:
        p_z = em.params['P_Z_I'][val]
        e_val = np.sum(p_z * expected_cat_given_z)
        res_internet.append(e_val)
    results['Internet'] = (labels_internet, res_internet)

    factors = {
        'Studytime': ('P_S_Z', ['<2h', '2-5h', '5-10h', '>10h']),
        'Failures': ('P_F_Z', ['0', '1', '2', '3+']),
        'Higher': ('P_H_Z', ['No', 'Yes']),
        'Absences': ('P_A_Z', ['None', 'Low', 'Med', 'High'])
    }

    for name, (param_name, labels) in factors.items():
        cpt = em.params[param_name]
        res_factor = []
        for v in range(len(labels)):
            likelihood = cpt[:, v]
            posterior_unnorm = likelihood * p_z_prior
            if posterior_unnorm.sum() == 0:
                posterior = p_z_prior
            else:
                posterior = posterior_unnorm / posterior_unnorm.sum()
            e_val = np.sum(posterior * expected_cat_given_z)
            res_factor.append(e_val)
        results[name] = (labels, res_factor)
    return results

data_full = load_and_process_discrete_data_multiclass()
idx = np.arange(len(data_full['internet']))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

def get_subset(d, idx):
    return {k: v[idx] for k, v in d.items()}

train_data = get_subset(data_full, train_idx)
test_data = get_subset(data_full, test_idx)

em = ImprovedDiscreteBN_EM(train_data, z_dim=4, seed=123)
em.fit()

preds, probs = em.predict(test_data)
true_vals = test_data['G3']

mse = mean_squared_error(true_vals, preds)
print(f"MSE (on Category Indices): {mse:.4f}")
print(f"RMSE (on Category Indices): {np.sqrt(mse):.4f}")

target_names = ['Fail', 'Pass', 'Good']
print("\nClassification Report:")
print(classification_report(true_vals, preds, target_names=target_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(true_vals, preds)
print(cm)

plt.figure(figsize=(6, 4))
plt.plot(em.history['delta'], marker='o', linestyle='-', color='purple')
plt.title('Convergence of EM (Max Parameter Change)')
plt.xlabel('Iteration')
plt.ylabel(r'Max $|\Delta \theta|$')
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_convergence.png')
print("Saved plot_convergence.png")

analysis_results = analyze_factors(em, data_full)
print("\nFactor Analysis Results (Expected Class Index 0-2):")
for k, v in analysis_results.items():
    print(f"{k}: {list(zip(v[0], np.round(v[1], 2)))}")

for factor_name, (labels, values) in analysis_results.items():
    plt.figure(figsize=(5, 4))
    bars = plt.bar(labels, values, color='lightgreen', edgecolor='black', width=0.6)
    plt.title(f'Impact of {factor_name}', fontsize=12)
    plt.ylabel('Expected Class (0=Fail, 1=Pass, 2=Good)')
    plt.ylim(0, 2)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    filename = f"plot_cat_{factor_name.lower()}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")