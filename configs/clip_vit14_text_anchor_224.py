work_dir = 'output/best_results'
note = ''
gpu_ids = range(0, 1)
model = dict(
    type='CLSLoraTextAnchor',
    encoder=dict(
        type='weight/clip/ViT-L-14.pt',
        pretrained=True,
        num_classes=0),
    text_feat_pth='text_features_by_class4.pt',
    feat_dim=[768, 16],
    test_cfg=dict(return_label=True, return_feature=False),
    train_cfg=dict(
        w_cls=0.5,
        w_contrastive=0.5,
        lora_modules=['out_proj', 'c_fc', 'c_proj'],
        weight_balance_strategy='inv'))
data_root = './'
img_norm_cfg = dict(
    mean=[122.7709383, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True)
data = dict(
    train=dict(
        type='DFDMultiDataset',
        data_root='/workspace/iccv2025_face_antispoofing/',
        ann_files=[
            'Protocol-train_mannual_clean_addReply_removepos_rev50x50all.txt',
            'PulID_output.txt', 'consistentID_output_onlyreal_train.txt',
            'train_reply_h264_aug.txt', 'train_reply_h264_aug.txt',
            'train_reply_h264_aug.txt', 'train_reply_h264_aug.txt',
            'train_reply_h264_aug.txt', 'inf8b_resin_mask_results.txt'
        ],
        crop_method='4_lmks_of_72',
        img_prefix=['/workspace/iccv2025_face_antispoofing/'],
        enlarge=1.0,
        test_mode=False,
        img_size=(224, 224),
        lamk_jit=dict(is_used=False, jitter_range=0.008, freeze_ratio=0.3),
        pipeline=dict(
            RandomFlip=dict(hflip_ratio=0.5, vflip_ratio=0),
            RandomCrop=dict(crop_ratio=0.2, crop_range=(0.1, 0.1)),
            Resize=dict(scale=(224, 224)),
            Normalize=dict(
                mean=[122.7709383, 116.7460125, 104.09373615],
                std=[68.5005327, 66.6321579, 70.32316305],
                to_rgb=True))),
    val=dict(
        type='DFDMultiDataset',
        data_root='/workspace/iccv2025_face_antispoofing/',
        ann_files=['Protocol-val-test.txt'],
        img_prefix=['/workspace/iccv2025_face_antispoofing/'],
        test_mode=True,
        enlarge=1.0,
        img_size=(224, 224),
        crop_method='4_lmks_of_5',
        pipeline=dict(
            Resize=dict(scale=(224, 224)),
            Normalize=dict(
                mean=[122.7709383, 116.7460125, 104.09373615],
                std=[68.5005327, 66.6321579, 70.32316305],
                to_rgb=True))),
    test=dict(
        type='DFDMultiDataset',
        data_root='/workspace/iccv2025_face_antispoofing',
        ann_files=['Protocol-val-test_infer.txt'],
        img_prefix=['/workspace/iccv2025_face_antispoofing/'],
        test_mode=True,
        enlarge=1.0,
        img_size=(224, 224),
        crop_method='4_lmks_of_5',
        pipeline=dict(
            Resize=dict(scale=(224, 224)),
            Normalize=dict(
                mean=[122.7709383, 116.7460125, 104.09373615],
                std=[68.5005327, 66.6321579, 70.32316305],
                to_rgb=True))),
    train_loader=dict(
        num_gpus=1,
        shuffle=True,
        samples_per_gpu=128,
        workers_per_gpu=16,
        sampler=None),
    test_loader=dict(
        num_gpus=1, shuffle=False, samples_per_gpu=32, workers_per_gpu=2))
log_cfg = dict(
    interval=10,
    filename=None,
    plog_cfg=dict(loss_types='all', eval_types=['acer']))
eval_cfg = dict(
    interval=40000,
    score_type='acer',
    tsne_cfg=dict(marks=None, filename='tsne.png'))
optim_cfg = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0005)
sched_cfg = dict(
    type='CosineAnnealingLR', every_iter=True, eta_min=1e-07, warmup=200)
check_cfg = dict(
    interval=50,
    save_topk=3,
    load_from=None,
    resume_from=None,
    pretrain_from=None)
freeze_cfg = None
total_epochs = 10
seed = 42
if model['type'] == 'CLSLoraTextAnchor':
    model['train_cfg']['ann_files'] = data['train']['ann_files']
    data['train_loader']['seed'] = seed
    data['test_loader']['seed'] = seed