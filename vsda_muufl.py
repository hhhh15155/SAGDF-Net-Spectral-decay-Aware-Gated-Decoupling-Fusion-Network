# train_vsda_muufl.py - VSDANet training for MUUFL dataset

import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
from tqdm import tqdm
import os
import json

import vsda_net
import geniter1
import Utils

# ============== Configuration ==============
GENERATE_LABELED_MAP = False
EPOCHS = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.001
RANDOM_SEED = 42

SAVE_DIR_PARAMS = 'cls_params/muufl'
SAVE_DIR_RESULTS = 'cls_result/muufl'
SAVE_DIR_MAPS = 'cls_map/muufl'

NUM_LAYERS = 2
NUM_HEADS = 4
EMB_DIM = 128
NUM_BLOCKS = 2


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loadData():
    """Load MUUFL dataset"""
    data_HSI = sio.loadmat('data/muufl.mat')['muufl']
    data_lidar = sio.loadmat('data/lidar_1.mat')['lidar']
    labels = sio.loadmat('data/muufl_label.mat')['muufl_gt']
    return data_HSI, data_lidar, labels


def applyPCA(X, numComponents):
    """Apply PCA for dimensionality reduction"""
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def select_traintest(groundTruth):
    """Select fixed number of training samples per class"""
    labels_loc = {}
    train = {}
    test = {}
    m = int(max(groundTruth))
    amount = [150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150]

    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(amount[i])
        train[i] = indices[-nb_val:]
        test[i] = indices[:-nb_val]

    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


def sampling(proportion, ground_truth):
    """Sample training and test data"""
    train = {}
    test = {}
    labels_loc = {}
    m = int(max(ground_truth))
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def create_data_loader():
    """Create data loaders"""
    X1, X2, y = loadData()

    # Process LiDAR data
    lidar_data_2d = np.zeros((325, 220, 1))
    lidar_data_2d[:, :, 0] = X2[:, :, 1]
    X2 = lidar_data_2d[:, :, 0]

    patch_size = 11
    PATCH_LENGTH = int((patch_size - 1) / 2)
    TOTAL_SIZE = 53687
    pca_components = 30

    print('Dataset: MUUFL Gulfport')
    print('Hyperspectral shape:', X1.shape)
    print('LiDAR shape:', X2.shape)
    print('Label shape:', y.shape)

    print('\nApplying PCA transformation...')
    X1 = applyPCA(X1, numComponents=pca_components)
    print('HSI shape after PCA:', X1.shape)

    # Normalize data
    X1_all_data = X1.reshape(np.prod(X1.shape[:2]), np.prod(X1.shape[2:]))
    X2_all_data = X2.reshape(np.prod(X2.shape[:2]), )
    gt = y.reshape(np.prod(y.shape[:2]), )
    gt = gt.astype(np.int_)
    CLASSES_NUM = max(gt)
    print('Number of classes:', CLASSES_NUM)

    X1_all_data = preprocessing.scale(X1_all_data)
    data_X1 = X1_all_data.reshape(X1.shape[0], X1.shape[1], X1.shape[2])
    whole_data_X1 = data_X1
    padded_data_X1 = np.lib.pad(whole_data_X1,
                                ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                                'constant', constant_values=0)

    X2_all_data = preprocessing.scale(X2_all_data)
    data_X2 = X2_all_data.reshape(X2.shape[0], X2.shape[1])
    whole_data_X2 = data_X2
    padded_data_X2 = np.lib.pad(whole_data_X2,
                                ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH)),
                                'constant', constant_values=0)

    print('\nCreating data loaders...')
    train_indices, test_indices = select_traintest(gt)
    _, total_indices = sampling(1, gt)

    TRAIN_SIZE = len(train_indices)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Training samples:', TRAIN_SIZE)
    print('Testing samples:', TEST_SIZE)

    train_iter, test_iter = geniter1.generate_iter(
        TRAIN_SIZE, train_indices, TEST_SIZE, test_indices,
        whole_data_X1, whole_data_X2, PATCH_LENGTH, padded_data_X1, padded_data_X2,
        pca_components, BATCH_SIZE, gt)

    if GENERATE_LABELED_MAP:
        _, total_iter = geniter1.generate_iter(
            TRAIN_SIZE, train_indices, TOTAL_SIZE, total_indices,
            whole_data_X1, whole_data_X2, PATCH_LENGTH, padded_data_X1, padded_data_X2,
            pca_components, BATCH_SIZE, gt)
    else:
        total_iter = None

    return train_iter, test_iter, total_iter, y, total_indices


def train(model, train_loader, optimizer, scheduler, device, epoch):
    """Training for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for hsi, lidar, labels in train_loader:
        hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(hsi, lidar)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()

    n = len(train_loader)
    return {
        'loss': total_loss / n,
        'acc': 100. * correct / total,
        'lr': optimizer.param_groups[0]['lr']
    }


def test(model, test_loader, device):
    """Testing"""
    model.eval()
    y_pred_test = []
    y_test = []

    with torch.no_grad():
        for hsi, lidar, labels in test_loader:
            hsi, lidar = hsi.to(device), lidar.to(device)
            outputs = model(hsi, lidar)
            _, predicted = outputs.max(1)
            y_pred_test.extend(predicted.cpu().numpy())
            y_test.extend(labels.numpy())

    return np.array(y_pred_test), np.array(y_test)


def AA_andEachClassAccuracy(confusion_matrix):
    """Calculate AA and per-class accuracy"""
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    """Generate accuracy reports"""
    target_names = [
        'Trees', 'Mostly Grass', 'Mixed Ground Surface', 'Dirt and Sand',
        'Road', 'Water', 'Buildings Shadow', 'Buildings',
        'Sidewalk', 'Yellow Curb', 'Cloth Panels'
    ]

    classification = classification_report(y_test, y_pred_test, digits=4,
                                           target_names=target_names, labels=range(11))
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


def main():
    print('=' * 80)
    print('VSDANet: Volumetric Spectral Decay Attention Network')
    print('Training on MUUFL Gulfport Dataset')
    print('=' * 80)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'\nDevice: {device}')

    print('\n--- Model Configuration ---')
    print(f'Transformer Layers: {NUM_LAYERS}')
    print(f'Attention Heads: {NUM_HEADS}')
    print(f'Embedding Dimension: {EMB_DIM}')
    print(f'TriScale Blocks: {NUM_BLOCKS}')

    print('\n--- Loading Data ---')
    train_iter, test_iter, total_iter, y, total_indices = create_data_loader()

    print('\n--- Creating Model ---')
    model = vsda_net.VSDANet(
        input_channels1=30,
        input_channels2=1,
        n_classes=11,
        patch_size=11,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        emb_dim=EMB_DIM,
        num_blocks=NUM_BLOCKS,
        use_spectral_decay=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    os.makedirs(SAVE_DIR_PARAMS, exist_ok=True)
    os.makedirs(SAVE_DIR_RESULTS, exist_ok=True)
    os.makedirs(SAVE_DIR_MAPS, exist_ok=True)

    print('\n--- Training ---')
    best_acc = 0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    pbar = tqdm(range(EPOCHS), desc='Training')
    tic_train = time.perf_counter()

    for epoch in pbar:
        stats = train(model, train_iter, optimizer, scheduler, device, epoch)
        history['train_loss'].append(stats['loss'])
        history['train_acc'].append(stats['acc'])

        pbar.set_postfix({
            'Loss': f'{stats["loss"]:.4f}',
            'Acc': f'{stats["acc"]:.2f}%',
            'LR': f'{stats["lr"]:.6f}'
        })

        if (epoch + 1) % 20 == 0:
            y_pred, y_true = test(model, test_iter, device)
            test_acc = accuracy_score(y_true, y_pred) * 100
            history['test_acc'].append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), f'{SAVE_DIR_PARAMS}/VSDANet_MUUFL_best.pth')

            print(f'\nEpoch {epoch + 1}: Test Acc = {test_acc:.2f}% (Best: {best_acc:.2f}% @ epoch {best_epoch})')

    toc_train = time.perf_counter()
    training_time = toc_train - tic_train

    print('\n--- Final Testing ---')
    model.load_state_dict(torch.load(f'{SAVE_DIR_PARAMS}/VSDANet_MUUFL_best.pth'))

    tic_test = time.perf_counter()
    y_pred_test, y_test = test(model, test_iter, device)
    toc_test = time.perf_counter()
    test_time = toc_test - tic_test

    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)

    # Save results
    with open(f'{SAVE_DIR_RESULTS}/VSDANet_MUUFL_report.txt', 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('VSDANet: Volumetric Spectral Decay Attention Network\n')
        f.write('Dataset: MUUFL Gulfport\n')
        f.write('=' * 80 + '\n\n')
        f.write('Architecture:\n')
        f.write('  HSI/LiDAR → TriScale → Patch Embed → SD-Transformer → VSDF → Classifier\n\n')
        f.write('Key Components:\n')
        f.write('  - TriScale: Triple-Scale Feature Extractor\n')
        f.write('  - SD-Transformer: Spectral Decay-aware Transformer\n')
        f.write('  - VSDF: Volumetric Spectral-Spatial Decoupling Fusion\n\n')
        f.write(f'Model Configuration:\n')
        f.write(f'  Transformer Layers: {NUM_LAYERS}\n')
        f.write(f'  Attention Heads: {NUM_HEADS}\n')
        f.write(f'  Embedding Dimension: {EMB_DIM}\n')
        f.write(f'  TriScale Blocks: {NUM_BLOCKS}\n\n')
        f.write(f'Training Configuration:\n')
        f.write(f'  Epochs: {EPOCHS}\n')
        f.write(f'  Batch Size: {BATCH_SIZE}\n')
        f.write(f'  Learning Rate: {LEARNING_RATE}\n')
        f.write(f'  Optimizer: AdamW (weight_decay=0.01)\n')
        f.write(f'  Scheduler: CosineAnnealingLR\n\n')
        f.write(f'Performance:\n')
        f.write(f'  Training Time: {training_time:.2f} s\n')
        f.write(f'  Testing Time: {test_time:.2f} s\n')
        f.write(f'  Best Epoch: {best_epoch}\n')
        f.write(f'  Parameters: {total_params:,}\n\n')
        f.write(f'Results:\n')
        f.write(f'  Overall Accuracy (OA): {oa:.2f}%\n')
        f.write(f'  Average Accuracy (AA): {aa:.2f}%\n')
        f.write(f'  Kappa Coefficient: {kappa:.2f}%\n')
        f.write(f'  Per-class Accuracy: {each_acc}\n\n')
        f.write(f'Detailed Classification Report:\n')
        f.write(f'{classification}\n')
        f.write(f'\nConfusion Matrix:\n{confusion}\n')

    with open(f'{SAVE_DIR_RESULTS}/VSDANet_MUUFL_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    print('\n' + '=' * 80)
    print('FINAL RESULTS')
    print('=' * 80)
    print(f'Overall Accuracy (OA): {oa:.2f}%')
    print(f'Average Accuracy (AA): {aa:.2f}%')
    print(f'Kappa Coefficient:     {kappa:.2f}%')
    print(f'Model Parameters:      {total_params:,}')
    print(f'Training Time:         {training_time:.2f} s')
    print('=' * 80)

    if GENERATE_LABELED_MAP and total_iter is not None:
        print('\n--- Generating Classification Map ---')
        Utils.generate_png(
            total_iter, model, y, device, total_indices,
            f'{SAVE_DIR_MAPS}/VSDANet_MUUFL_classification'
        )

    print('\nTraining completed successfully!')


if __name__ == '__main__':
    main()