import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def integrated_gradient(inputs, model, predict_and_gradients, original_image_x, before_Ttanh_output, steps, device, baseline, results_path, feed_tTanh=True, feature_type='image', env_type='atari'):
    scaled_inputs = []
    for k in range(steps + 1):
        step_k_image = baseline + (float(k) / steps) * (inputs - baseline)
        scaled_inputs.append(step_k_image)
    if env_type == 'atari':
        plot_inputs_together(results_path, scaled_inputs)

    grads = predict_and_gradients(scaled_inputs, model, original_image_x, before_Ttanh_output, device, feed_tTanh=feed_tTanh)
    avg_grads = np.average(grads, axis=0)
    if feature_type == 'image':
        avg_grads = np.transpose(avg_grads, (0, 2, 3, 1))
    integrated_grad = []
    for j in range(len(avg_grads)):
        if feature_type == 'image':
            integrated_grad.append((inputs.reshape(80, 80, 1) - baseline.reshape(80, 80, 1)).cpu().numpy() * avg_grads[j])
        elif feature_type == 'vector':
            integrated_grad.append((inputs.reshape(inputs.shape[1], inputs.shape[0]) - baseline.reshape(baseline.shape[1], baseline.shape[0])).cpu().numpy() * avg_grads[j].reshape(baseline.shape[1], baseline.shape[0]))
    avg_intgrads = np.array(integrated_grad)
    avg_grads = np.array(avg_grads)
    return avg_intgrads, avg_grads


def plot_inputs_together(path, steps_images):
    nrows, ncols = 5, 10
    figsize = [80, 80]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, axi in enumerate(ax.flat):
        axi.imshow(steps_images[i].cpu().numpy()[0].reshape(steps_images[i].shape[2:]))
        axi.set_title("step " + str(i), fontsize=45)
        axi.set_yticklabels([])
        axi.set_xticklabels([])

    plt.tight_layout(True)
    plt.savefig(path + "/scaled_inputs.jpg")
    fig.clf()


def mask_diff_ig(img_integrated_gradient_overlay, unmasked_integrated_gradient, input_image, original_image_x, baseline_x, input_img_path, save_path):
    original_image_x = original_image_x.cpu().detach().numpy()[0].tolist()
    baseline_x = baseline_x.cpu().detach().numpy()[0].tolist()
    different_bit_values_ig = []
    different_bit_values_uig = []
    for i in range(len(original_image_x)):
        if original_image_x[i] != baseline_x[i]:
            different_bit_values_ig.append(img_integrated_gradient_overlay[i])
            different_bit_values_uig.append(unmasked_integrated_gradient[i])

    mask = np.average(np.array(different_bit_values_ig), axis=0)
    unmask = np.average(np.array(different_bit_values_uig), axis=0)
    input_image = cv2.resize(input_image.cpu().numpy().reshape(80, 80), (160, 160))
    mask = cv2.resize(mask, (160, 160))
    unmask = cv2.resize(unmask, (160, 160))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[0].imshow(mask, 'hot', interpolation='None', alpha=0.6)
    ax[0].set_title("masked_diff_IG")
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[1].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[1].imshow(unmask, 'hot', interpolation='None', alpha=0.6)
    ax[1].set_title("unmasked_diff_IG")
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    plt.tight_layout(True)
    plt.savefig(os.path.join(save_path, "diff_IGs.jpg"))
    fig.clf()

