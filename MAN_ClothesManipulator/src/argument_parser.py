# args for eval.py

def add_base_args(parser):  # common arguments for all CLIs
    parser.add_argument('--dataset_name', type=str,
                        default='Shopping100k',
                        choices=['Shopping100k'],
                        help='Select dataset Shopping100k')
    parser.add_argument('--backbone', type=str,
                        default='alexnet',
                        choices=['alexnet', 'resnet'],
                        help='Select pretrained backbone architecture (alexnet or resnet)')
    parser.add_argument('--file_root', type=str,
                        default='../splits/Shopping100k',
                        help='Path for pre-processed files')
    parser.add_argument('--img_root', type=str,
                        default='../Shopping100k/Images',
                        help='Path for raw images')
    parser.add_argument('--num_threads', type=int,
                        default=16,
                        help='Number of threads for fetching data (default: 16)')
    parser.add_argument('--batch_size', type=int,
                        default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--dim_chunk', type=int,
                        default=340,
                        help='Dimension of each attribute-specific embedding')
    parser.add_argument('--load_pretrained_extractor', type=str,
                        default='../models/Shopping100k/extractor_best.pkl',
                        help='Load pretrained weights of disentangled representation learner')
    parser.add_argument('--load_pretrained_MAN', type=str,
                        default='../nets/last.pth',
                        help='Load pretrained memory augmented network')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Do not use cuda')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Which GPU to use (if any available)')

def add_eval_args(parser):
    parser.add_argument('--ref_ids', type=str,
                        default='ref_test.txt',
                        help='list of query image id')
    parser.add_argument('--gt_labels', type=str,
                        default='gt_test.txt',
                        help='list of target labels')
    parser.add_argument('--query_inds', type=str,
                        default='indfull_test.txt',
                        help='list of indicators')
    parser.add_argument('--top_k', type=int,
                        default=30,
                        help='top K neighbours')
    parser.add_argument('--save_gallery', action='store_true',
                        default=False,
                        help='Save the gallery feature')
    parser.add_argument('--save_output', action='store_true',
                        default=False,
                        help='Save the output feature')
    parser.add_argument('--feat_dir', type=str,
                        default='eval_out',
                        help='Path to store gallery feature and fused feature')
