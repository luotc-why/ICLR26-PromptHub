import os.path
from tqdm import tqdm
from trainer import val_pascal_dataloader
from evaluate.reasoning_dataloader import *
import torchvision
from evaluate.mae_utils import *
import argparse
from pathlib import Path
from evaluate.segmentation_utils import *
from PIL import Image
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from models.train_models import _generate_result_for_canvas, PGVP
from evaluate_detection.voc_orig import CLASS_NAMES


def get_args():
    parser = argparse.ArgumentParser('PromptHub for segmentation', add_help=False)
    parser.add_argument('--mae_model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument("--mode", type=str, default='spimg_spmask',
                        choices=['no_vp', 'spimg_spmask', 'spimg', 'spimg_qrimg', 'qrimg', 'spimg_spmask_qrimg'],
                        help="mode of adding vp on img.")
    parser.add_argument('--output_dir', default=f'./output_samples')
    parser.add_argument('--device', default='cuda:7',
                        help='device to use for training / testing')
    parser.add_argument('--base_dir', default='./pascal-5i', help='pascal base dir')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--task', default='segmentation', choices=['segmentation', 'detection'])
    parser.add_argument('--ckpt', default='./weights/checkpoint-1000.pth', help='model checkpoint')
    parser.add_argument('--save_base_dir', default='./VisualICL', help='/prefix/VisualICL/')
    parser.add_argument('--vq_ckpt_dir', default='/data/luotianci/TO_JPSX/VisualICL/weights/vqgan', help="dir for vq-gan's config and model ckpt")
    parser.add_argument('--dataset_type', default='pascal')
    parser.add_argument('--simidx', default=1, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--split', default='trn', type=str)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--feature_name', default='features_vit-laion2b_pixel-level_trn', type=str)
    parser.add_argument('--percentage', default='', type=str)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--G_pre_mean', action='store_true')
    parser.add_argument('--G_copy_another', action='store_true')
    parser.add_argument('--G_only_div', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--fsl', action='store_true')
    parser.add_argument('--save_examples', action='store_true', help='whether save the example in val')
    parser.add_argument('--sigma', default=0.1, type=float)
    parser.add_argument('--intra_loss',type=int, default=1)
    parser.add_argument('--add_zx_loss', action='store_true')
    # training settings
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--lr", type=float, default=40,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Number of training steps.")
    parser.add_argument("--loss-function", type=str, default='CrossEntropy',
                        help="loss function for training")
    parser.add_argument("--scheduler", type=str, default='cosinewarm',
                        help="scheduler for training")
    parser.add_argument("--arr", type=str, default='a1',
                        help="the setting of arrangements of canvas")
    parser.add_argument("--p-eps", type=int, default=1,
                        help="Number of pad weight hyperparameter [0, 1].")
    parser.add_argument("--vp-model", type=str, default='pad',
                        help="type of the VP Prompter.")
    parser.add_argument("--loss_mean",type=int, default=0)
    # Number of images for few-shot training
    parser.add_argument("--n-shot", type=int, default=16,
                        help="Number of images for fsl.")
    parser.add_argument("--choice", type=str, default='Cut',
                        help="choose prompt composer")
    parser.add_argument('--align_s',type=int, default=1)
    parser.add_argument('--align_q',type=int, default=0)
    parser.add_argument('--kernel_size', default=7, type=int)
    parser.add_argument("--loss_choice", type=str, default='cos',
                        help="choose prompt composer")
    parser.add_argument("--lamba", type=float, default='0.5',
                        help="choose prompt composer")
    parser.add_argument("--pos", type=str, default='after',
                        help="choose prompt composer")
    parser.add_argument('--save_model_path',
                        help='model checkpoint')
    parser.add_argument("--gama", type=float, default='0.2',
                        help="choose prompt composer")

    return parser


def test_for_generate_results(args):
    padding = 1
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])

    val_dataset = {
        'pascal': val_pascal_dataloader.DatasetPASCAL,
    }[args.dataset_type](args.base_dir, fold=args.fold, split=args.split, image_transform=image_transform,
                         mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, random=args.random, cluster=args.cluster,
                         feature_name=args.feature_name, percentage=args.percentage, seed=args.seed, mode=args.mode,
                         arr=args.arr, simidx=args.simidx, args=args)

    dataloaders = {}
    dataloaders['val'] = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print('val datalaoder: ', len(dataloaders['val']))

    print("load data over")

    # MAE_VQGAN model
    vqgan = prepare_model(args.ckpt, arch=args.mae_model, vq_ckpt_dir=args.vq_ckpt_dir)

    if args.vp_model == 'Prompt':
        print('load pad prompter.')
        VP = PGVP(args=args, vqgan=vqgan.to(args.device), mode=args.mode, arr=args.arr)
    else:
        raise ValueError("No VP model is loaded, Please use 'pad'")

    if args.mode != 'no_vp':
        checkpoint = torch.load(args.save_model_path,map_location=args.device)
        VP.PromptGenerator.load_state_dict(checkpoint["visual_prompt_dict"])

        VP.eval()
        VP.to(args.device)

    setting = f'{args.mode}_fold{args.fold}_{args.task}_{args.arr}_{args.simidx}'
    eg_save_path = f'{args.output_dir}/{args.vp_model}_output_examples/{args.mode}'
    os.makedirs(eg_save_path, exist_ok=True)

    print(f'This is the mode of {args.mode}.')
    print(f'This is the arrangement of {args.arr}.')

    eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
    examples_save_path = eg_save_path + f'/{setting}/'
    os.makedirs(examples_save_path, exist_ok=True)

    with open(os.path.join(examples_save_path, 'log.txt'), 'w') as log:
        log.write(str(args) + '\n')

    image_number = 0

    # Inference phase
    for i, data in enumerate(tqdm(dataloaders["val"])):
        len_dataloader = len(dataloaders["val"])
        support_features = data['support_features']
        query_img_features = data['query_img_features']
        support_features = support_features.to(args.device, dtype=torch.float32)
        query_img_features = query_img_features.to(args.device, dtype=torch.float32)
        support_img, support_mask, query_img, query_mask, grid_stack = \
            data['support_img'], data['support_mask'], data['query_img'], data['query_mask'], data['grid_stack']
        support_img = support_img.to(args.device, dtype=torch.float32)
        support_mask = support_mask.to(args.device, dtype=torch.float32)
        query_img = query_img.to(args.device, dtype=torch.float32)
        query_mask = query_mask.to(args.device, dtype=torch.float32)
        grid_stack = grid_stack.to(args.device, dtype=torch.float32)

        _, canvas_pred_tokens, canvas_label = VP(support_img.unsqueeze(1), support_mask.unsqueeze(1), query_img, query_mask, grid_stack.unsqueeze(1), 
                            query_img_features, support_features)


        original_image_list, generated_result_list = _generate_result_for_canvas(args, vqgan.to(args.device),
                                                                                 canvas_pred_tokens, canvas_label, args.arr)
        for index in range(len(original_image_list)):

            Image.fromarray(generated_result_list[index]).save(
                examples_save_path + f'generated_image_{image_number}.png')

            original_image = round_image(original_image_list[index], [WHITE, BLACK])
            generated_result = round_image(generated_result_list[index], [WHITE, BLACK], t=args.t)

            current_metric = calculate_metric(args, original_image, generated_result, fg_color=WHITE, bg_color=BLACK)
            with open(os.path.join(examples_save_path, 'log.txt'), 'a') as log:
                log.write(str(image_number) + '\t' + str(current_metric) + '\n')
            image_number += 1

            for i, j in current_metric.items():
                eval_dict[i] += (j / len(val_dataset))

    print('val metric: {}'.format(eval_dict))
    with open(os.path.join(examples_save_path, 'log.txt'), 'a') as log:
        log.write('all\t' + str(eval_dict) + '\n')


if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    test_for_generate_results(args)
