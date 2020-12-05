import os
import gym
import argparse
from model_def import *
from visualization import plot_feature_vector
from integrated_gradients import integrated_gradient, mask_diff_ig
from utils import calculate_outputs_and_gradients_steps, gather_observations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='differential-IG')
    parser.add_argument('--env_type', type=str, default='inception', help='the type of network')
    parser.add_argument('--input_index', type=str, help='input image index')
    parser.add_argument('--baseline_index', type=str, help='baseline iamge index')
    parser.add_argument('--env', type=str, help='environment')
    parser.add_argument('--qbn_sizes', nargs=2, type=int, help='HX_QBN size and OX_QBN size')
    parser.add_argument('--gru_size', type=str, help='GRU cell size')
    parser.add_argument('--env_seed', type=int, default=1, help='environment seed')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = args.env
    qbn_sizes = str((args.qbn_sizes[0], args.qbn_sizes[1])).replace(" ", "")
    model_path = "./inputs/" + env_name + "/model_bgru_" + args.gru_size + "_" + str(qbn_sizes) + ".p"

    saved_observations = gather_observations(env_name, int(args.gru_size), args.qbn_sizes[0], args.qbn_sizes[1],
                                             model_path, device, episodes=1, env_type=args.env_type)

    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + args.env_type):
        os.mkdir('results/' + args.env_type)

    results_path = "input_" + args.input_index + "_baseline_" + args.baseline_index
    if not os.path.exists(os.path.join('results/', args.env_type, env_name)):
        os.mkdir(os.path.join('results/', args.env_type, env_name))
    if not os.path.exists(os.path.join('results/', args.env_type, env_name, results_path)):
        os.mkdir(os.path.join('results/', args.env_type, env_name, results_path))
    results_path = os.path.join('results/', args.env_type, env_name, results_path)

    env = gym.make(env_name)
    env.seed(args.env_seed)
    obs = env.reset()
    gru_net = ControlGRUNet(len(obs), int(args.gru_size), int(env.action_space.n))
    ox_net = ControlObsQBNet(gru_net.input_c_features, int(args.qbn_sizes[1]))
    model = ControlMMNet(gru_net, None, ox_net)
    pretrained_ox_dict = {k[8:]: v for k, v in torch.load(model_path, map_location='cpu').items() if k.startswith("obx_net")}
    model.obx_net.load_state_dict(pretrained_ox_dict)
    pretrained_gru_dict = {k[8:]: v for k, v in torch.load(model_path, map_location='cpu').items() if k.startswith("gru_net")}
    model.gru_net.load_state_dict(pretrained_gru_dict)

    model.eval()
    model.to(device)
    input_image = saved_observations[int(args.input_index)]
    baseline_image = saved_observations[int(args.input_index)]
    ii_f, ii_x, ii_q = model(input_image.to(device), inspect=False)
    bi_f, bi_x, bi_q = model(baseline_image.to(device), inspect=False)
    attributions, unmasked_attributions = integrated_gradient(input_image, model, calculate_outputs_and_gradients_steps,
                                                              ii_x, ii_q, steps=10, device=device, baseline=baseline_image,
                                                              results_path=results_path, feed_tTanh=True, feature_type='vector',
                                                              env_type=args.env_type)
    input_image_path = os.path.join("./inputs/", args.env, args.input_index + ".jpg")
    plot_feature_vector(attributions, results_path, args.env)