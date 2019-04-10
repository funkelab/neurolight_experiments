import numpy as np


def intersection_over_union(pred, gt, labels=None):
    iou = {}
    dice = {}

    pred_channels = pred.shape[3] if pred.ndim == 4 else 1
    gt_channels = gt.shape[3] if gt.ndim == 4 else 1

    for i in range(gt_channels):

        if gt.ndim == 4:
            gt_channel = gt[:, :, :, i]
        else:
            gt_channel = gt

        gt_mask = gt_channel > 0
        overlay = np.array([np.tile(gt_channel[gt_mask].flatten(), pred_channels),
                            pred[gt_mask].flatten()])
        matches, counts = np.unique(overlay, axis=1, return_counts=True)
        gt_labels = np.unique(gt_channel[gt_mask])

        if labels is None:
            label_list = gt_labels
        else:
            for gt_label in gt_labels:
                if gt_label not in labels:
                    print('WARNING: not all labels are included in given list!')
            label_list = labels

        for gt_label in label_list:

            idx = np.logical_and(matches[0] == gt_label, matches[1] > 0)
            if np.sum(idx) == 0:
                iou[gt_label] = 0
                dice[gt_label] = 0
                continue

            pred_labels = matches[1, idx]
            pred_label = pred_labels[np.argmax(counts[idx])]

            gt_label_mask = gt_channel == gt_label
            pred_label_mask = pred == pred_label
            for j in range(pred_channels):
                if np.sum(pred_label_mask[:, :, :, j]) > 0:
                    pred_label_mask = pred_label_mask[:, :, :, j]
                    break

            iou[gt_label] = np.sum(np.logical_and(gt_label_mask, pred_label_mask)) / float(
                np.sum(np.logical_or(gt_label_mask, pred_label_mask)))
            #dice[gt_label] = 2 * np.sum(np.logical_and(gt_label_mask, pred_label_mask)) / float(
            #    np.sum(pred_label_mask) + np.sum(gt_label_mask))

    return iou