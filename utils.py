import os
import random

import gym
import numpy as np
import matplotlib.pyplot as plt
from model_def import *
import torch.nn.functional as F
from env_wrapper import atari_wrapper


def calculate_outputs_and_gradients_steps(inputs, model, original_image_x, input_before_quantized, device, grad_clip=1, feed_tTanh=False):
    gradients = []
    for i in range(len(inputs)):
        bits_gradients = []
        input = torch.tensor(inputs[i], requires_grad=True)
        output_c, output_x, output_before_quantized = model(input.to(device))
        for j in range(len(output_x[0])):
            model.zero_grad()
            if feed_tTanh:
                loss = nn.MSELoss()(output_before_quantized[0][j], input_before_quantized[0][j].to(device))
            else:
                loss = nn.MSELoss()(output_x[0][j], original_image_x[0][j])
            loss.backward(retain_graph=True)
            gradient = input.grad.detach().cpu().numpy()[0].tolist()
            bits_gradients.append(gradient)
            input.grad.data.fill_(0)
        gradients.append(bits_gradients)
    gradients = np.array(gradients)
    assert gradients[0][0].shape == inputs[0][0].shape
    return gradients


def generate_entrie_images(img_origin, img_integrad_overlay, bit_values, img_file_name, results_path):
    for i in range(len(img_integrad_overlay)):
        if i < 10:
            bit_i = '0' + str(i)
        else:
            bit_i = str(i)
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        plt.imshow(img_integrad_overlay[i].reshape(img_origin.shape[:-1]), cmap='hot')
        fig.savefig(results_path + '/IG_bit' + bit_i + '_bv' + str(int(bit_values[0][i].data)) + '_' + img_file_name + ".jpg")

    plot_GIs_together(results_path, img_integrad_overlay)


def plot_GIs_together(path, IGs):
    nrows, ncols = 10, 10
    figsize = [20, 20]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, axi in enumerate(ax.flat):
        axi.imshow(IGs[i].reshape(IGs[i].shape[:2]), cmap='hot')
        axi.set_title("bit: " + str(i))
        axi.set_yticklabels([])
        axi.set_xticklabels([])
    plt.tight_layout(True)
    plt.savefig(path + "/IGs.jpg")
    fig.clf()


def gather_observations(env_name, gru_size, bhx_size, ox_size, bgru_net_path, device, episodes=1, env_type='atari'):
    if os.path.exists('./inputs/' + str(env_name) + '/observations.pt'):
        observations = torch.load('./inputs/' + str(env_name) + '/observations.pt', map_location=device)
        return observations

    if env_type == 'atari':
        env = atari_wrapper(env_name)
        env.seed(0)
        obs = env.reset()
        gru_net = GRUNet(len(obs), gru_size, int(env.action_space.n))
        bhx_net = HxQBNet(gru_size, bhx_size)
        ox_net = ObsQBNet(gru_net.input_c_features, ox_size)
        bgru_net = MMNet(gru_net, bhx_net, ox_net)
    elif env_type == 'classic_control':
        env = gym.make(env_name)
        env.seed(0)
        obs = env.reset()
        gru_net = ControlGRUNet(len(obs), gru_size, int(env.action_space.n))
        bhx_net = ControlHxQBNet(gru_size, bhx_size)
        ox_net = ControlObsQBNet(gru_net.input_c_features, ox_size)
        bgru_net = ControlMMNet(gru_net, bhx_net, ox_net)

    if cuda:
        bgru_net = bgru_net.cuda()

    bgru_net.load_state_dict(torch.load(bgru_net_path, map_location='cpu'))
    bgru_net.eval()
    bgru_net.eval()
    max_actions = 10000
    random.seed(0)
    x = set([])
    observations = []
    with torch.no_grad():
        for ep in range(episodes):
            done = False
            obs = env.reset()
            curr_state = bgru_net.init_hidden()
            if cuda:
                curr_state = curr_state.cuda()
            curr_state_x = bgru_net.state_encode(curr_state)
            ep_reward = 0
            ep_actions = []
            record_changes = []
            while not done:
                # env.render()

                curr_action = bgru_net.get_action_linear(curr_state_x, decode=True)
                prob = F.softmax(curr_action, dim=1)
                curr_action = int(prob.max(1)[1].cpu().data.numpy()[0])
                obs = torch.Tensor(obs).unsqueeze(0)
                if cuda:
                    obs = obs.cuda()
                critic, logit, next_state, (next_state_c, next_state_x), (_, obs_x, obs_tanh) = bgru_net((obs, curr_state),
                                                                                                    inspect=True)
                observations.append(obs)
                prob = F.softmax(logit, dim=1)
                next_action = int(prob.max(1)[1].cpu().data.numpy())

                obs, reward, done, _ = env.step(next_action)

                done = done if len(ep_actions) <= max_actions else True
                # a quick hack to prevent the agent from stucking
                max_same_action = 5000
                if len(ep_actions) > max_same_action:
                    actions_to_consider = ep_actions[-max_same_action:]
                    if actions_to_consider.count(actions_to_consider[0]) == max_same_action:
                        done = True
                curr_state = next_state
                curr_state_x = next_state_x

                ep_reward += reward
                x.add(''.join([str(int(i)) for i in next_state.cpu().data.numpy()[0]]))

    torch.save(observations, './inputs/' + str(env_name) + '/observations.pt')
    return observations