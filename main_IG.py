from utils import calculate_outputs_and_gradients_steps, gather_observations
from integrated_gradients import integrated_gradient, mask_diff_ig
from env_wrapper import atari_wrapper
from model_def import *
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='differential-IG')
    parser.add_argument('--cuda', action='store_true', help='if use the cuda to do the accelartion')
    parser.add_argument('--env_type', type=str, default='inception', help='the type of network')
    parser.add_argument('--input_index', type=str, help='input image index')
    parser.add_argument('--baseline_index', type=str, help='baseline iamge index')
    parser.add_argument('--env', type=str, help='environment')
    parser.add_argument('--qbn_sizes', nargs=2, type=int, help='HX_QBN size and OX_QBN size')
    parser.add_argument('--gru_size', type=str, help='GRU cell size')
    parser.add_argument('--env_seed', type=int, default=1, help='environment seed')
    args = parser.parse_args()

    env_name = args.env
    qbn_sizes = str((args.qbn_sizes[0], args.qbn_sizes[1])).replace(" ", "")
    model_path = "../results/Atari/" + env_name + "/gru_" + args.gru_size + "_hx_" + str(qbn_sizes) + "_bgru/model.p"

    saved_observations, saved_observations_x, saved_observations_tanh = gather_observations(env_name, int(args.gru_size), args.qbn_sizes[0], args.qbn_sizes[1], model_path, episodes=1, cuda=args.cuda, env_type=args.env_type)

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

    env = atari_wrapper(env_name)
    env.seed(args.env_seed)
    obs = env.reset()
    gru_net = GRUNet(len(obs), int(args.gru_size), int(env.action_space.n))
    ox_net = ObsQBNet(gru_net.input_c_features, int(args.qbn_sizes[1]))
    model = MMNet(gru_net, None, ox_net)
    pretrained_ox_dict = {k[8:]: v for k, v in torch.load(model_path, map_location='cpu').items() if k.startswith("obx_net")}
    model.obx_net.load_state_dict(pretrained_ox_dict)
    pretrained_gru_dict = {k[8:]: v for k, v in torch.load(model_path, map_location='cpu').items() if k.startswith("gru_net")}
    model.gru_net.load_state_dict(pretrained_gru_dict)

    model.eval()
    if args.cuda:
        model.cuda()
    input_image = saved_observations[int(args.input_index)]
    baseline_image = saved_observations[int(args.baseline_index)]
    ii_f, ii_x, ii_q = model(input_image.cuda(), inspect=False)
    bi_f, bi_x, bi_q = model(baseline_image.cuda(), inspect=False)
    attributions, unmasked_attributions = integrated_gradient(input_image, model, calculate_outputs_and_gradients_steps, ii_x, ii_q, steps=50, cuda=args.cuda, baseline=baseline_image, results_path=results_path, feed_tTanh=True, env_type=args.env_type)
    input_image_path = os.path.join("./inputs/", args.env, args.input_index + ".jpg")
    mask_diff_ig(attributions, unmasked_attributions, input_image, ii_x, bi_x, input_image_path, results_path)
