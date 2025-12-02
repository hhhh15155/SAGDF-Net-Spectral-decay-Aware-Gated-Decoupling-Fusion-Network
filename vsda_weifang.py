# train_vsda_weifang.py - VSDANet training for Weifang dataset

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
GENERATE_LABELED_MAP = True
EPOCHS = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.001
RANDOM_SEED = 42

SAVE_DIR_PARAMS = 'cls_params/weifang'
SAVE_DIR_RESULTS = 'cls_result/weifang'
SAVE_DIR_MAPS = 'cls_map/weifang'

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
    """Load Weifang dataset"""
    data_HSI = sio.loadmat('data/weifang/hsi.mat')['image_data']
    data_lidar = sio.loadmat('data/weifang/lidar.mat')['image_data']
    labels = sio.loadmat('data/weifang/label.mat')['image_data']
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
    amount = [10, 10, 10, 10, 10, 10, 10, 10]

    print(f'\n各类别训练/测试样本分配:')
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = min(amount[i], len(indices))
        train[i] = indices[:nb_val]
        test[i] = indices[nb_val:]

        print(f'  类别 {i + 1}: {nb_val:3d} 训练, {len(test[i]):5d} 测试 (总共 {len(indices):5d})')

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

    patch_size = 11
    PATCH_LENGTH = int((patch_size - 1) / 2)
    pca_components = 30

    print('Dataset: Weifang')
    print('Hyperspectral shape:', X1.shape)
    print('LiDAR shape:', X2.shape)
    print('Label shape:', y.shape)

    print('\nApplying PCA transformation...')
    X1 = applyPCA(X1, numComponents=pca_components)
    print('HSI shape after PCA:', X1.shape)

    X1_all_data = X1.reshape(np.prod(X1.shape[:2]), np.prod(X1.shape[2:]))
    X2_all_data = X2.reshape(np.prod(X2.shape[:2]), )
    gt = y.reshape(np.prod(y.shape[:2]), )
    gt = gt.astype(np.int_)

    total_pixels = len(gt)
    background_samples = np.sum(gt == 0)
    labeled_samples = np.sum(gt > 0)

    print(f'\n样本统计:')
    print(f'  总像素数: {total_pixels:,}')
    print(f'  背景样本 (类别0): {background_samples:,}')
    print(f'  标注样本 (类别1-8): {labeled_samples:,}')

    CLASSES_NUM = max(gt)
    print(f'  类别数: {CLASSES_NUM}')

    unique, counts = np.unique(gt[gt > 0], return_counts=True)
    print('\n各类别样本分布:')
    for cls, count in zip(unique, counts):
        print(f'  类别 {cls}: {count:,} 样本')

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

    print('\n过滤背景样本，只使用有标注样本...')
    labeled_indices = np.where(gt > 0)[0]
    gt_labeled = gt[labeled_indices]

    TOTAL_SIZE = len(labeled_indices)
    print(f'有效样本数: {TOTAL_SIZE:,}')

    print('\n划分训练集和测试集...')
    train_indices, test_indices = select_traintest(gt_labeled)

    train_indices_abs = labeled_indices[train_indices]
    test_indices_abs = labeled_indices[test_indices]

    TRAIN_SIZE = len(train_indices_abs)
    TEST_SIZE = len(test_indices_abs)
    print(f'\n总训练样本: {TRAIN_SIZE:,}')
    print(f'总测试样本: {TEST_SIZE:,}')
    print(f'训练/测试比例: {TRAIN_SIZE}/{TEST_SIZE} = {TRAIN_SIZE / TEST_SIZE:.2f}')

    train_iter, test_iter = geniter1.generate_iter(
        TRAIN_SIZE, train_indices_abs, TEST_SIZE, test_indices_abs,
        whole_data_X1, whole_data_X2, PATCH_LENGTH, padded_data_X1, padded_data_X2,
        pca_components, BATCH_SIZE, gt)

    if GENERATE_LABELED_MAP:
        _, total_iter = geniter1.generate_iter(
            TRAIN_SIZE, train_indices_abs, TOTAL_SIZE, labeled_indices,
            whole_data_X1, whole_data_X2, PATCH_LENGTH, padded_data_X1, padded_data_X2,
            pca_components, BATCH_SIZE, gt)
    else:
        total_iter = None

    return train_iter, test_iter, total_iter, y, labeled_indices


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
        'Class 1', 'Class 2', 'Class 3', 'Class 4',
        'Class 5', 'Class 6', 'Class 7', 'Class 8'
    ]

    classification = classification_report(y_test, y_pred_test, digits=4,
                                           target_names=target_names, labels=range(8))
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


def main():
    print('=' * 80)
    print('VSDANet: Volumetric Spectral Decay Attention Network')
    print('Training on Weifang Dataset')
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
        n_classes=8,
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
    history = {'train_loss': [], 'train_acc': []}

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

    toc_train = time.perf_counter()
    training_time = toc_train - tic_train

    # 保存最终模型
    torch.save(model.state_dict(), f'{SAVE_DIR_PARAMS}/VSDANet_Weifang_final.pth')
    print(f'\n最终模型已保存到 {SAVE_DIR_PARAMS}/VSDANet_Weifang_final.pth')

    print('\n--- Final Testing ---')
    tic_test = time.perf_counter()
    y_pred_test, y_test = test(model, test_iter, device)
    toc_test = time.perf_counter()
    test_time = toc_test - tic_test

    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)

    # Save results
    with open(f'{SAVE_DIR_RESULTS}/VSDANet_Weifang_report.txt', 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('VSDANet: Volumetric Spectral Decay Attention Network\n')
        f.write('Dataset: Weifang\n')
        f.write('=' * 80 + '\n\n')
        f.write('Dataset Information:\n')
        f.write('  Image Size: 271 x 626\n')
        f.write(f'  Total Pixels: 169,646\n')
        f.write(f'  Background Samples (Class 0): 83,949\n')
        f.write(f'  Labeled Samples (Class 1-8): 85,697\n')
        f.write('  HSI Bands: 126 (PCA to 30)\n')
        f.write('  LiDAR Bands: 1\n')
        f.write('  Classes: 8\n')
        f.write('  Note: Training and testing only on labeled samples\n\n')
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
        f.write(f'  Training Time: {training_time:.2f} s ({training_time / 60:.2f} min)\n')
        f.write(f'  Testing Time: {test_time:.2f} s\n')
        f.write(f'  Parameters: {total_params:,}\n\n')
        f.write(f'Results (Final Model):\n')
        f.write(f'  Overall Accuracy (OA): {oa:.2f}%\n')
        f.write(f'  Average Accuracy (AA): {aa:.2f}%\n')
        f.write(f'  Kappa Coefficient: {kappa:.2f}%\n')
        f.write(f'  Per-class Accuracy: {each_acc}\n\n')
        f.write(f'Detailed Classification Report:\n')
        f.write(f'{classification}\n')
        f.write(f'\nConfusion Matrix:\n{confusion}\n')

    with open(f'{SAVE_DIR_RESULTS}/VSDANet_Weifang_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    print('\n' + '=' * 80)
    print('FINAL RESULTS')
    print('=' * 80)
    print(f'Overall Accuracy (OA): {oa:.2f}%')
    print(f'Average Accuracy (AA): {aa:.2f}%')
    print(f'Kappa Coefficient:     {kappa:.2f}%')
    print(f'Model Parameters:      {total_params:,}')
    print(f'Training Time:         {training_time:.2f} s ({training_time / 60:.2f} min)')
    print('=' * 80)

    if GENERATE_LABELED_MAP and total_iter is not None:
        print('\n--- Generating Classification Map ---')
        Utils.generate_png(
            total_iter, model, y, device, total_indices,
            f'{SAVE_DIR_MAPS}/VSDANet_Weifang_classification'
        )

    print('\nTraining completed successfully!')


if __name__ == '__main__':
    main()
