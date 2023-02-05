#!/usr/bin/env python3

import cv2

import numpy as np
import torch
from cvl.utils import iou
from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import NCCTracker, MOSSE, DeepMOSSE
from cvl.features import colornames_image, alexnetFeatures
import argparse
import matplotlib.pyplot as plt
import pickle 

def create_parser():
    parser = argparse.ArgumentParser(
        description="Different ways to use the finetuned model.")
    parser.add_argument("--NCC", action="store_true")
    parser.add_argument("--MOSSE", action="store_true")
    parser.add_argument("--show_tracking", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--colornames", action="store_true")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--deep_features", action="store_true")
    parser.add_argument("--use_subset", action="store_true")
    parser.add_argument("--search_region", action="store_true", help='Whether to use twice the bbox as search area')
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--r_search", type=float, default=0, help="Value of 0 indicates using the full boundary box as search. Otherwise it is a square region of width r_search.")
    parser.add_argument("--filter_region", type=float, default=2, help="Value of 1 means using the boundary box as region for the filters to run over. Otherwise it is scaled with factor filter_region.")
    parser.add_argument("--sigma", type=float, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.02)


    return parser


def main(args):
    dataset_path = "Mini-OTB"
    dataset = OnlineTrackingBenchmark(dataset_path)

    if len([x for x in [args.grayscale, args.colornames, args.rgb, args.deep_features] if x]) > 1:
        raise ValueError('Only one of grayscale, colornames, rgb or deep features can be used.')

    if args.use_subset:
        data_subset_to_test = [5]
    else:
        data_subset_to_test = list(range(0,30))

    assert set(data_subset_to_test).issubset(set(range(0,30)))

    if args.show_tracking:
        cv2.namedWindow("tracker")
    if args.NCC:
        tracker = NCCTracker()
        tracker_name = 'NCC_grayscale'
    elif args.MOSSE:
        tracker = MOSSE(learning_rate=args.learning_rate, sigma=args.sigma, filter_region_scale=args.filter_region, r_search=args.r_search, deep_features=args.deep_features)
        tracker_name = 'MOSSE'

    if args.deep_features:
    #     tracker = DeepMOSSE(filter_region_scale=args.filter_region, r_search=args.r_search)
        net = alexnetFeatures(pretrained=True, layer=args.layer)
    #     tracker_name = f'DeepMOSSE_layer_{net.layer}'

    if args.grayscale:
        representation = '_grayscale'
    elif args.colornames:
        representation = '_colornames'
    elif args.rgb:
        representation = '_rgb'
    elif args.deep_features:
        representation = f'_deep_features_{args.layer}'
    
    config = f'_filter_region_{args.filter_region}_r_search_{args.r_search}'

    savename = tracker_name + representation + config

    score_dict = {}
    original_image_shape = None # only set if deep features used
    deep_image_shape = None
    for i in data_subset_to_test:
        a_seq = dataset[i]
        iou_score = []
        for frame_idx, frame in enumerate(a_seq):
            print(f"{i} {tracker_name}: {frame_idx} / {len(a_seq)}")
            image_color = frame['image']
            height, width, channels = image_color.shape
            tracker.channels = channels

            if args.grayscale:
                image = np.sum(image_color, 2) / 3
                # add extra dimension to make MOSSE generalize to any number of channels
                image = np.expand_dims(image, axis=2)

            elif args.colornames:
                image = colornames_image(image_color)

            elif args.rgb:
                image = image_color

            elif args.deep_features:
                image = image_color
                original_image_shape = image.shape    
                image = net(torch.tensor(image).unsqueeze(0).permute(
                        0, 3, 1, 2).float()).squeeze().permute(1,2,0).detach().numpy()
                deep_image_shape = image.shape

            if frame_idx == 0:
                bbox = frame['bounding_box']

                if bbox.width % 2 == 0:
                    bbox.width += 1

                if bbox.height % 2 == 0:
                    bbox.height += 1

                current_position = bbox
                tracker.start(image, bbox, deep_image_shape=deep_image_shape, original_image_shape=original_image_shape)

            else:
                tracker.detect(image)
                tracker.update(image)

            if args.show_tracking:
                bbox = tracker.region
                pt0 = (bbox.xpos, bbox.ypos)
                pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
                image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_color, pt0, pt1,
                              color=(0, 255, 0), thickness=3)
                cv2.imshow("tracker", image_color)
                cv2.waitKey(0)

            ground_truth = frame['bounding_box']
            iou_score.append(iou(ground_truth, bbox))
        score_dict[i] = iou_score
    
    if args.use_subset:
        with open(f'results/ubset/{tracker_name}.pkl', 'wb') as f:
            pickle.dump(score_dict, f)
    else:
        with open(f'results/full/{savename}.pkl', 'wb') as f:
            pickle.dump(score_dict, f)



if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
