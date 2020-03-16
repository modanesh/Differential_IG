import os

import numpy as np
import cv2
import colorsys
import matplotlib.pyplot as plt


def convert_to_gray_scale(attributions):
    return np.average(attributions, axis=3)


def compute_threshold_by_top_percentage(attributions, percentage=60, plot_distribution=True):
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    if (cum_sum >= percentage).min().data:
        threshold_idx = None
        threshold = 0
    else:
        threshold_idx = np.where(cum_sum >= percentage)[0][0]
        threshold = sorted_attributions[threshold_idx]
    if plot_distribution:
        raise NotImplementedError 
    return threshold


def polarity_function(attributions, polarity):
    if polarity == 'positive':
        return np.clip(attributions, 0, 1)
    elif polarity == 'negative':
        return np.clip(attributions, -1, 0)
    else:
        raise NotImplementedError


def overlay_function(attributions, image):
    return np.clip(0.7 * image + 0.5 * attributions, 0, 255)


def visualize(attributions, image, overlay=True, mask_mode=False):
    if overlay:
        if mask_mode == False:
            attributions = overlay_function(attributions, image)
        else:
            attributions = attributions * image.cpu().numpy()
    return attributions


def plot_feature_vector(attributions, results_path, env_name):
    attributions = np.average(attributions, axis=0)
    fig, ax = plt.subplots()
    plt.imshow(attributions, cmap='hot')
    plt.colorbar()
    plt.ylabel("Input Features", fontsize=12, fontweight='bold')
    plt.gca().axes.get_xaxis().set_visible(False)
    labels = [item.get_text() for item in ax.get_yticklabels()]

    if env_name == 'Acrobot-v1':
        # labels for Acrobot's features
        plt.yticks(np.arange(0, 6, 1))
        labels[0] = "cos(joint1)"
        labels[1] = "sin(joint1)"
        labels[2] = "cos(joint2)"
        labels[3] = "sin(joint2)"
        labels[4] = "joint1\nvelocity"
        labels[5] = "joint2\nvelocity"
    elif env_name == 'CartPole-v1':
        # labels for CartPole's features
        plt.yticks(np.arange(0, 4, 1))
        labels[0] = "cart\nposition"
        labels[1] = "cart\nvelocity"
        labels[2] = "pole\nangle"
        labels[3] = "pole\nvelocity"
    elif env_name == 'LunarLander-v2':
        # labels for LunarLander's features
        plt.yticks(np.arange(0, 8, 1))
        labels[0] = "Pos X"
        labels[1] = "Pos Y"
        labels[2] = "Velocity X"
        labels[3] = "Velocity Y"
        labels[4] = "Angle"
        labels[5] = "Angular Velocity"
        labels[6] = "Left-leg Pos"
        labels[7] = "Right-leg Pos"

    ax.set_yticklabels(labels)
    plt.savefig(os.path.join(results_path, "diff_IGs.jpg"), bbox_inches='tight')
    plt.show()