import argparse
from lib.utils import yaml2config
from networks import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/gan_iam.yml",
        help="Configuration file to use",
    )

    parser.add_argument(
        "--ckpt",
        nargs="?",
        type=str,
        help="checkpoint for evaluation",
    )

    parser.add_argument(
        "--mode",
        nargs="?",
        type=str,
        help="mode: [rand] [style] [interp] [text]",
    )

    args = parser.parse_args()
    cfg = yaml2config(args.config)

    model = get_model(cfg.model)(cfg, args.config)
    model.load(args.ckpt, cfg.device)
    if args.mode == 'style':
        model.eval_style()
    elif args.mode == 'rand':
        model.eval_rand()
    elif args.mode == 'interp':
        model.eval_interp()
    elif args.mode == 'text':
        model.eval_text()
    else:
        print('Unsupported mode: {} | [rand] [style] [interp]'.format(cfg.mode))
