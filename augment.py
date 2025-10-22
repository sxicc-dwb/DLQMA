import pickle
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from tensorflow.keras import layers
from matplotlib import rcParams


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def augment_positive(spectra, ids4p, low=0.2, high=1.0):

    no = random.randint(0, len(ids4p) - 1)
    total_ratio = 0.0
    x = np.zeros_like(spectra[0]['fid'])


    ratio = random.uniform(low, high)
    x += ratio * spectra[ids4p[-1]]['fid']
    total_ratio += ratio


    q = random.randint(-20, 20)
    x = np.roll(x, q)


    for i in range(no):
        ratio = random.uniform(low, high)
        s = ratio * spectra[ids4p[i]]['fid']
        q = random.randint(-20, 20)
        s = np.roll(s, q)
        x += s
        total_ratio += ratio

    return x, total_ratio


def augment_negative(spectra, ids4n, low=0.2, high=1.0):

    no = random.randint(1, len(ids4n) - 1)
    total_ratio = 0.0
    x = np.zeros_like(spectra[0]['fid'])

    for i in range(no):
        ratio = random.uniform(low, high)
        s = ratio * spectra[ids4n[i]]['fid']
        q = random.randint(-20, 20)
        s = np.roll(s, q)
        x += s
        total_ratio += ratio

    return x, total_ratio


def data_augmentation(spectra, n, max_pc, noise_level=0.0001, concentration_components=1):

    p = spectra[0]['ppm'].shape[0]
    s = len(spectra)


    Rp = np.zeros((n, p), dtype=np.float32)
    Sp = np.zeros((n, p), dtype=np.float32)
    Rn = np.zeros((n, p), dtype=np.float32)
    Sn = np.zeros((n, p), dtype=np.float32)

    if concentration_components > 1:
        conc = np.zeros((2 * n, concentration_components), dtype=np.float32)
    else:
        conc = np.zeros(2 * n, dtype=np.float32)

    for i in tqdm(range(n), desc="waite"):

        n1 = np.random.normal(0, 1, p)
        n2 = np.random.normal(0, 1, p)
        n3 = np.random.normal(0, 1, p)
        n4 = np.random.normal(0, 1, p)


        ids4p = random.sample(range(s), min(max_pc, s - 1))
        Rp[i,] = spectra[ids4p[-1]]['fid'] + (n1 - np.min(n1)) * noise_level
        Sp[i,], conc_p = augment_positive(spectra, ids4p)
        Sp[i,] += (n2 - np.min(n2)) * noise_level


        ids4n = random.sample(range(s), min(max_pc + 1, s))
        Rn[i,] = spectra[ids4n[-1]]['fid'] + (n3 - np.min(n3)) * noise_level
        Sn[i,], conc_n = augment_negative(spectra, ids4n)
        Sn[i,] += (n4 - np.min(n4)) * noise_level


        if concentration_components > 1:

            conc[i, 0] = conc_p
            conc[n + i, :] = 0.0
        else:

            conc[i] = conc_p
            conc[n + i] = 0.0


    R = np.vstack((Rp, Rn))
    S = np.vstack((Sp, Sn))
    y = np.concatenate((np.ones(n), np.zeros(n)), axis=None)


    R, S, y, conc = shuffle(R, S, y, conc)

    return {'R': R, 'S': S, 'y': y, 'conc': conc}


def visualize_augmented_data(aug, num_samples=5):

    plt.figure(figsize=(15, 10))


    pos_indices = np.where(aug['y'] == 1)[0][:num_samples]

    for i, idx in enumerate(pos_indices):
        plt.subplot(2, num_samples, i + 1)
        plt.plot(aug['R'][idx])
        plt.plot(aug['S'][idx])
        plt.title(f"positive {i + 1}, conc: {aug['conc'][idx]:.2f}")

    neg_indices = np.where(aug['y'] == 0)[0][:num_samples]

    for i, idx in enumerate(neg_indices):
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.plot(aug['R'][idx])
        plt.plot(aug['S'][idx])
        plt.title(f"nagative {i + 1}, conc: {aug['conc'][idx]:.2f}")

    plt.tight_layout()
    plt.show()


def save_data(aug, file_path):


    os.makedirs(os.path.dirname(file_path), exist_ok=True)


    with open(file_path, 'wb') as f:
        pickle.dump(aug, f, protocol=4)

    print(f" {file_path}")


def load_data(file_path):

    with open(file_path, 'rb') as f:
        aug = pickle.load(f)


    if 'conc' not in aug:
        print(f"none")
        aug['conc'] = np.zeros(aug['y'].shape[0])

    print(f": {len(aug['R'])} ")
    return aug


def data_augmentation_classification_focused(spectra, n, max_pc, noise_level=0.0001):

    p = spectra[0]['ppm'].shape[0]
    s = len(spectra)
    

    aug_data = data_augmentation(spectra, n//2, max_pc, noise_level)
    

    difficult_R = np.zeros((n//2, p), dtype=np.float32)
    difficult_S = np.zeros((n//2, p), dtype=np.float32)
    difficult_y = np.zeros(n//2, dtype=np.float32)
    difficult_conc = np.zeros(n//2, dtype=np.float32)
    
    for i in tqdm(range(n//2), desc=" difficult"):

        n1 = np.random.normal(0, 1, p)
        n2 = np.random.normal(0, 1, p)
        

        if i < n//4:
            ids = random.sample(range(s), min(max_pc+2, s-1))
            difficult_R[i,] = spectra[ids[-1]]['fid'] + (n1 - np.min(n1)) * noise_level

            low_ratio = random.uniform(0.05, 0.15)
            difficult_S[i,] = low_ratio * spectra[ids[-1]]['fid']

            for j in range(min(max_pc+1, len(ids)-1)):
                ratio = random.uniform(0.5, 1.0)
                s_temp = ratio * spectra[ids[j]]['fid']
                q = random.randint(-20, 20)
                s_temp = np.roll(s_temp, q)
                difficult_S[i,] += s_temp
            difficult_S[i,] += (n2 - np.min(n2)) * noise_level
            difficult_y[i] = 1.0
            difficult_conc[i] = low_ratio
        

        else:
            ids = random.sample(range(s), min(max_pc+2, s))
            target_idx = random.randint(0, s-1)
            difficult_R[i,] = spectra[target_idx]['fid'] + (n1 - np.min(n1)) * noise_level
            difficult_S[i,] = np.zeros_like(spectra[0]['fid'])

            for j in range(min(3, len(ids))):
                if ids[j] != target_idx:
                    ratio = random.uniform(0.3, 1.0)
                    s_temp = ratio * spectra[ids[j]]['fid']
                    q = random.randint(-20, 20)
                    s_temp = np.roll(s_temp, q)
                    difficult_S[i,] += s_temp
            difficult_S[i,] += (n2 - np.min(n2)) * noise_level * 2
            difficult_y[i] = 0.0
            difficult_conc[i] = 0.0
    

    aug_data['R'] = np.vstack([aug_data['R'], difficult_R])
    aug_data['S'] = np.vstack([aug_data['S'], difficult_S])
    aug_data['y'] = np.concatenate([aug_data['y'], difficult_y])
    aug_data['conc'] = np.concatenate([aug_data['conc'], difficult_conc])
    

    aug_data['R'], aug_data['S'], aug_data['y'], aug_data['conc'] = shuffle(
        aug_data['R'], aug_data['S'], aug_data['y'], aug_data['conc']
    )
    
    return aug_data


if __name__ == "__main__":
    from readBruker import read_bruker_hs_base


    os.makedirs('aug', exist_ok=True)


    sample_count = 50000
    max_components = 5
    noise_level = 0.0001
    conc_components = 1


    spectra = read_bruker_hs_base('mydata/standards', False, True, False)


    aug_train = data_augmentation(
        spectra,
        sample_count,
        max_components,
        noise_level,
        conc_components
    )
    save_data(aug_train, 'aug/data_augment_train.pkl')


    aug_valid = data_augmentation(
        spectra,
        5000,
        max_components,
        noise_level,
        conc_components
    )
    save_data(aug_valid, 'aug/data_augment_valid.pkl')


    aug_test = data_augmentation(
        spectra,
        5000,
        max_components,
        noise_level,
        conc_components
    )
    save_data(aug_test, 'aug/data_augment_test.pkl')

    visualize_augmented_data(aug_train, 5)

    for dataset_name, file_path in [
        ('train', 'aug/data_augment_train.pkl'),
        ('valida', 'aug/data_augment_valid.pkl'),
        ('test', 'aug/data_augment_test.pkl')
    ]:
        aug = load_data(file_path)
        pos_count = np.sum(aug['y'] == 1)
        neg_count = np.sum(aug['y'] == 0)


        if isinstance(aug['conc'], np.ndarray) and len(aug['conc'].shape) > 1:
            print(f": {aug['conc'].shape}")
        else:
            print(f"conc: {aug['conc'].shape}")

    print("\ndone!")
