# train_vsda_binzhou.py - VSDANet training for Binzhou dataset

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
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

# 保存目录（Binzhou）
SAVE_DIR_PARAMS = 'cls_params/binzhou'
SAVE_DIR_RESULTS = 'cls_result/binzhou'
SAVE_DIR_MAPS = 'cls_map/binzhou'

# VSDANet 结构参数（可按需要调整）
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
    """
    加载 Binzhou 数据集

    根据你的转换脚本，三个 .mat 中变量名都是 'image_data'，形状为：
        HSI:   (126, 638, 321) -> (638, 321, 126)  (bands first, 需转置)
        Label: (638, 321)
        LiDAR:(638, 321)
    """
    # 如果你放在 data/binzhou 下，可改成：
    data_HSI = sio.loadmat('data/binzhou/hsi.mat')['image_data']
    labels  = sio.loadmat('data/binzhou/label.mat')['image_data']
    data_lidar = sio.loadmat('data/binzhou/lidar.mat')['image_data']



    # HSI 当前是 (C, H, W)，转换成 (H, W, C)，以和 HHK 脚本流程一致
    if data_HSI.ndim == 3 and data_HSI.shape[0] < data_HSI.shape[-1]:
        data_HSI = np.transpose(data_HSI, (1, 2, 0))  # (126,638,321)->(638,321,126)

    # LiDAR & Label 已经是 (H, W)，无需转置
    return data_HSI.astype(np.float32), data_lidar.astype(np.float32), labels.astype(np.int32)


def applyPCA(X, numComponents):
    """Apply PCA for dimensionality reduction"""
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


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


def select_traintest(groundTruth):
    """
    Select fixed number of training samples per class
    Binzhou 数据集共有 5 个前景类别（标签 1~5），
    每类默认取 150 个作为训练，其余为测试。
    """
    labels_loc = {}
    train = {}
    test = {}
    m = int(max(groundTruth))  # = 5

    # 如果想调整每类样本数，修改此处
    amount = [10, 10, 10, 10, 10]

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


def create_data_loader():
    """Create data loaders for Binzhou dataset"""
    X1, X2, y = loadData()

    patch_size = 11
    PATCH_LENGTH = int((patch_size - 1) / 2)
    pca_components = 30  # 对 126 个光谱波段做 PCA -> 30 维

    print('Dataset: Binzhou')
    print('Hyperspectral data shape:', X1.shape)
    print('LiDAR data shape:', X2.shape)
    print('Label shape:', y.shape)

    print('\nPCA transformation...')
    X1 = applyPCA(X1, numComponents=pca_components)
    print('Data shape after PCA:', X1.shape)

    height, width, bands = X1.shape

    # ---------- 展平数据 ----------
    X1_all_data = np.reshape(X1, [height * width, bands])

    # LiDAR 是单通道 (H, W) -> (H*W, 1)
    X2_all_data = np.reshape(X2, [height * width, 1])

    # 标签展平
    gt = y.reshape(np.prod(y.shape[:2]), )
    gt = gt.astype(np.int_)

    # 0 为背景，1~5 为前景类别
    CLASSES_NUM = int(max(gt))  # 5
    print('Number of foreground classes (without background):', CLASSES_NUM)

    # ---------- 归一化 ----------
    minMax = preprocessing.MinMaxScaler()
    X1_all_data = minMax.fit_transform(X1_all_data)
    data_X1 = X1_all_data.reshape(X1.shape[0], X1.shape[1], X1.shape[2])
    whole_data_X1 = data_X1
    padded_data_X1 = np.lib.pad(
        whole_data_X1,
        ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
        'constant',
        constant_values=0
    )

    minMax = preprocessing.MinMaxScaler()
    X2_all_data = minMax.fit_transform(X2_all_data)
    data_X2 = X2_all_data.reshape(X2.shape[0], X2.shape[1])
    whole_data_X2 = data_X2
    padded_data_X2 = np.lib.pad(
        whole_data_X2,
        ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH)),
        'constant',
        constant_values=0
    )

    # ---------- 构造训练 / 测试索引 ----------
    print('\nCreating train & test data...')
    train_indices, test_indices = select_traintest(gt)

    # total_indices：所有前景标记像元索引（不含背景 0）
    _, total_indices = sampling(1, gt)
    TRAIN_SIZE = len(train_indices)
    TEST_SIZE = len(test_indices)
    TOTAL_SIZE = len(total_indices)  # = 所有带标签像素数（44525）

    print('Total labeled samples:', TOTAL_SIZE)
    print('Train size:', TRAIN_SIZE)
    print('Test size:', TEST_SIZE)

    # ---------- 迭代器 ----------
    train_iter, test_iter = geniter1.generate_iter(
        TRAIN_SIZE, train_indices,
        TEST_SIZE, test_indices,
        whole_data_X1, whole_data_X2,
        PATCH_LENGTH, padded_data_X1, padded_data_X2,
        pca_components, BATCH_SIZE, gt
    )

    if GENERATE_LABELED_MAP:
        _, total_iter = geniter1.generate_iter(
            TRAIN_SIZE, train_indices,
            TOTAL_SIZE, total_indices,
            whole_data_X1, whole_data_X2,
            PATCH_LENGTH, padded_data_X1, padded_data_X2,
            pca_components, BATCH_SIZE, gt
        )
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


def AA_andEachClassAccuracy(confusion_matrix_):
    """Calculate AA and per-class accuracy"""
    list_diag = np.diag(confusion_matrix_)
    list_raw_sum = np.sum(confusion_matrix_, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    """Generate accuracy reports for Binzhou dataset"""
    # Binzhou 有 5 个前景类别，对应标签 1~5，这里给个占位名字
    target_names = [
        'Class 1',
        'Class 2',
        'Class 3',
        'Class 4',
        'Class 5'
    ]

    classification = classification_report(
        y_test, y_pred_test, digits=4,
        target_names=target_names, labels=range(1, 6)
    )
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test, labels=range(1, 6))
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


def main():
    print('=' * 70)
    print('VSDANet Training on Binzhou Dataset')
    print('=' * 70)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    print('\n--- Model Configuration ---')
    print(f'Transformer Layers: {NUM_LAYERS}')
    print(f'Attention Heads: {NUM_HEADS}')
    print(f'Embedding Dimension: {EMB_DIM}')
    print(f'TriScale Blocks: {NUM_BLOCKS}')

    print('\n--- Loading Data ---')
    train_iter, test_iter, total_iter, y, total_indices = create_data_loader()

    print('\n--- Creating Model ---')
    model = vsda_net.VSDANet(
        input_channels1=30,   # PCA 后的光谱维度
        input_channels2=1,    # LiDAR 单通道
        n_classes=5,          # Binzhou：5 个前景类别（不含背景 0）
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

    pbar = tqdm(range(EPOCHS))
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
                torch.save(model.state_dict(), f'{SAVE_DIR_PARAMS}/VSDANet_Binzhou_best.pth')

            print(f'\nEpoch {epoch + 1}: Test Acc = {test_acc:.2f}% (Best: {best_acc:.2f}% at epoch {best_epoch})')

    toc_train = time.perf_counter()
    training_time = toc_train - tic_train

    print('\n--- Final Testing ---')
    model.load_state_dict(torch.load(f'{SAVE_DIR_PARAMS}/VSDANet_Binzhou_best.pth'))

    tic_test = time.perf_counter()
    y_pred_test, y_test = test(model, test_iter, device)
    toc_test = time.perf_counter()
    test_time = toc_test - tic_test

    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)

    # Save results
    with open(f'{SAVE_DIR_RESULTS}/VSDANet_Binzhou_report.txt', 'w') as f:
        f.write('=' * 70 + '\n')
        f.write('VSDANet: Binzhou Dataset (5 classes + background)\n')
        f.write('=' * 70 + '\n\n')
        f.write('Architecture: TriScale → SD-Transformer → VSDF → Classifier\n\n')
        f.write(f'Model Configuration:\n')
        f.write(f'  Transformer Layers: {NUM_LAYERS}\n')
        f.write(f'  Attention Heads: {NUM_HEADS}\n')
        f.write(f'  Embedding Dimension: {EMB_DIM}\n')
        f.write(f'  TriScale Blocks: {NUM_BLOCKS}\n\n')
        f.write(f'Training Time: {training_time:.2f} s\n')
        f.write(f'Test Time: {test_time:.2f} s\n')
        f.write(f'Best Epoch: {best_epoch}\n')
        f.write(f'Parameters: {total_params:,}\n')
        f.write(f'OA: {oa:.2f}%\n')
        f.write(f'AA: {aa:.2f}%\n')
        f.write(f'Kappa: {kappa:.2f}%\n')
        f.write(f'Each Accuracy: {each_acc}\n')
        f.write(f'\n{classification}\n')
        f.write(f'\nConfusion Matrix:\n{confusion}\n')

    with open(f'{SAVE_DIR_RESULTS}/VSDANet_Binzhou_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    print('\n' + '=' * 70)
    print('RESULTS:')
    print(f'OA: {oa:.2f}%')
    print(f'AA: {aa:.2f}%')
    print(f'Kappa: {kappa:.2f}%')
    print(f'Parameters: {total_params:,}')
    print('=' * 70)

    if GENERATE_LABELED_MAP and total_iter is not None:
        print('\n--- Generating Classification Map ---')
        Utils.generate_png(
            total_iter, model, y, device, total_indices,
            f'{SAVE_DIR_MAPS}/VSDANet_Binzhou_classification'
        )

    print('\nTraining completed!')


if __name__ == '__main__':
    main()
