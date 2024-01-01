# set args.load_gallery to None the first time you run and True then in argument_parser.py

import argparse
import os
import numpy as np
from tqdm import tqdm
import faiss
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataloader import Data, DataQuery
from model import Extractor
from argument_parser import add_base_args, add_eval_args
from utils import split_labels, compute_NDCG, get_target_attr
import constants as C
from test import init_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    add_eval_args(parser)
    args = parser.parse_args()
    if not args.use_cpu and not torch.cuda.is_available():
        print('Warning: Using CPU')
        args.use_cpu = True
    else:
        torch.cuda.set_device(args.gpu_id)

    file_root = args.file_root
    img_root_path = args.img_root

    # load test dataset
    print('Loading gallery...')
    gallery_data = Data(file_root, img_root_path,
                        transforms.Compose([
                            transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                        ]), mode='test')

    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False,
                                                 sampler=torch.utils.data.SequentialSampler(gallery_data),
                                                 num_workers=args.num_threads,
                                                 drop_last=False)

    model = Extractor(gallery_data.attr_num, backbone=args.backbone, dim_chunk=args.dim_chunk)

    if args.load_pretrained_extractor:
        print('load {path} \n'.format(path=args.load_pretrained_extractor))
        model.load_state_dict(torch.load(args.load_pretrained_extractor))
    else:
        print(
            'Pretrained extractor not provided. Use --load_pretrained_extractor or the model will be randomly initialized.')

    if not os.path.exists(args.feat_dir):
        os.makedirs(args.feat_dir)

    if not args.use_cpu:
        model.cuda()

    model.eval()

    # indexing the gallery

    # gallery_feat = []  # all the test disentangled features (20000 x 4080)
    # with torch.no_grad():
    #     for i, (img, _) in enumerate(tqdm(gallery_loader)):
    #         if not args.use_cpu:
    #             img = img.cuda()
    #
    #         dis_feat, _ = model(img)
    #         gallery_feat.append(torch.cat(dis_feat, 1).squeeze().cpu().numpy())
    #
    # if args.save_gallery:
    #     np.save(os.path.join(args.feat_dir, 'gallery_feat.npy'), gallery_feat)
    #     print('Saved indexed features at {dir}/gallery_feat.npy'.format(dir=args.feat_dir))
    # gallery_feat = np.concatenate(gallery_feat, axis=0).reshape(-1, args.dim_chunk * len(gallery_data.attr_num))
    gallery_feat = np.load('../disentangledFeaturesExtractor/feat_test_senzaNorm.npy')

    # indexing the query
    query_inds = np.loadtxt(os.path.join(file_root, args.query_inds), dtype=int)  # query's single manipulation vector
    gt_labels = np.loadtxt(os.path.join(file_root, args.gt_labels), dtype=int)  # target's ground_truth labels
    ref_idxs = np.loadtxt(os.path.join(file_root, args.ref_ids), dtype=int)  # ids query images

    assert (query_inds.shape[0] == gt_labels.shape[0]) and (query_inds.shape[0] == ref_idxs.shape[0])

    query_fused_feats = []
    print('Loading test queries...')
    query_data = DataQuery(file_root, img_root_path,
                           args.ref_ids, args.query_inds,
                           transforms.Compose([
                               transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                           ]), mode='test')
    query_loader = torch.utils.data.DataLoader(query_data, batch_size=args.batch_size, shuffle=False,
                                               sampler=torch.utils.data.SequentialSampler(query_data),
                                               num_workers=args.num_threads,
                                               drop_last=False)

    # Initialize MAN
    task = "clothesManipulatorTask"
    MAN_NET = init_model(task)
    if args.load_pretrained_MAN:
        print('load {path} \n'.format(path=args.load_pretrained_MAN))
        MAN_NET.net.load_state_dict(torch.load(args.load_pretrained_MAN))
    else:
        print(
            'Pretrained MAN not provided. Use --load_pretrained_MAN or the model will be randomly initialized.')


    with torch.no_grad():
        for i, (img, indicator) in enumerate(tqdm(query_loader)):
            indicator = indicator.float()
            if not args.use_cpu:
                img = img.cuda()
            dis_feat, _ = model(img)
            query = torch.reshape(torch.from_numpy(torch.cat(dis_feat, 1).squeeze().cpu().numpy()), (1, 12, 340))
            MAN_NET.net.init_sequence_query(args.batch_size, query)
            MAN_NET.net(indicator)
            net_memory = MAN_NET.net.get_memory()
            memory = net_memory.memory.numpy().reshape(4080)

            query_fused_feats.append(memory)

    if args.save_output:
        np.save(os.path.join(args.feat_dir, 'query_fused_feats.npy'), np.concatenate(query_fused_feats, axis=0))
        print('Saved query features at {dir}/query_fused_feats.npy'.format(dir=args.feat_dir))

    # evaluate the top@k results
    print("Evaluate top@k")
    output_feat = np.array(np.concatenate(query_fused_feats, axis=0)).reshape(-1, args.dim_chunk * len(
        gallery_data.attr_num))
    dim = args.dim_chunk * len(gallery_data.attr_num)  # dimension
    num_database = gallery_feat.shape[0]  # number of images in database
    num_query = output_feat.shape[0]  # number of queries

    database = gallery_feat
    queries = output_feat
    index = faiss.IndexFlatL2(dim)
    index.add(database)
    k = args.top_k
    _, knn = index.search(queries, k)

    # load the GT labels for all gallery images
    label_data = np.loadtxt(os.path.join(file_root, 'labels_test.txt'), dtype=int)

    # compute top@k acc
    hits = 0
    for q in tqdm(range(num_query)):
        neighbours_idxs = knn[q]
        for n_idx in neighbours_idxs:
            if (label_data[n_idx] == gt_labels[q]).all():
                hits += 1
                break
    print('Top@{k} accuracy: {acc}'.format(k=k, acc=hits / num_query))

    # compute NDCG

    print("Compute NDCG")
    ndcg = []
    ndcg_target = []  # consider changed attribute only
    ndcg_others = []  # consider other attributes

    for q in tqdm(range(num_query)):
        rel_scores = []
        target_scores = []
        others_scores = []

        neighbours_idxs = knn[q]
        indicator = query_inds[q]
        target_attr = get_target_attr(indicator, gallery_data.attr_num)
        target_label = split_labels(gt_labels[q], gallery_data.attr_num)

        for n_idx in neighbours_idxs:
            n_label = split_labels(label_data[n_idx], gallery_data.attr_num)
            # compute matched_labels number
            match_cnt = 0
            others_cnt = 0

            for i in range(len(n_label)):
                if (n_label[i] == target_label[i]).all():
                    match_cnt += 1
                if i == target_attr:
                    if (n_label[i] == target_label[i]).all():
                        target_scores.append(1)
                    else:
                        target_scores.append(0)
                else:
                    if (n_label[i] == target_label[i]).all():
                        others_cnt += 1

            rel_scores.append(match_cnt / len(gallery_data.attr_num))
            others_scores.append(others_cnt / (len(gallery_data.attr_num) - 1))

        ndcg.append(compute_NDCG(np.array(rel_scores)))
        ndcg_target.append(compute_NDCG(np.array(target_scores)))
        ndcg_others.append(compute_NDCG(np.array(others_scores)))

    print('NDCG@{k}: {ndcg}, NDCG_target@{k}: {ndcg_t}, NDCG_others@{k}: {ndcg_o}'.format(k=k,
                                                                                          ndcg=np.mean(ndcg),
                                                                                          ndcg_t=np.mean(ndcg_target),
                                                                                          ndcg_o=np.mean(ndcg_others)))

"""
NET 500_200.pth
Top@10 accuracy: 0.32838938792563194 -> 32.84%
Top@30 accuracy: 0.5129169277905439 -> 51.29%

NDCG@30: 0.7447023391723633 -> 0.7447
NDCG_target@30: 0.3207034468650818 -> 0.3207
NDCG_others@30: 0.7991127967834473 -> 0.7991

################################################

NET 700_200.pth
Top@10 accuracy: 0.3357704895202284 -> 33.58%
Top@20 accuracy: 0.4495508669312722 -> 44.96%
Top@30 accuracy: 0.5200194972494951 -> 52.00%
Top@40 accuracy: 0.5671610612074368 -> 56.72%
Top@50 accuracy: 0.6017686790613467 -> 60.18%

NDCG@30: 0.7448408603668213 -> 0.7448
NDCG_target@30: 0.3259589374065399 -> 0.3259
NDCG_others@30: 0.7986679077148438 -> 0.7987

#################################################

NET 500_200_mix_1_3.pth
Top@10 accuracy: 0.32504700229789013 -> 32.50%
Top@30 accuracy: 0.5003829816865121 -> 50.03%

NDCG@30: 0.7458316683769226 -> 0.7458
NDCG_target@30: 0.30914244055747986 -> 0.3091
NDCG_others@30: 0.801673173904419 -> 0.8017
"""
