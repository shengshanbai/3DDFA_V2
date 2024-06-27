import argparse
import lmdb_util
import yaml
from TDDFA import TDDFA
import cv2
from utils.depth import depth
from tqdm import tqdm


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    tddfa = TDDFA(gpu_mode=True, **cfg)
    db_env = lmdb_util.read_lmdb(args.db_dir)
    image_keys = lmdb_util.read_folder(db_env, "/train/real")
    for image_key in tqdm(image_keys):
        image_np = lmdb_util.read_image(db_env, f"/train/real/{image_key}")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        params = tddfa.pred_crop_face(image_np)
        v3d = tddfa.reconv_dense(
            params, [0, 0, image_np.shape[1], image_np.shape[0]])
        image_depth = depth(image_np, [v3d,], tddfa.tri, show_flag=False,
                            wfp=None, with_bg_flag=False)
        cv2.imwrite(f"./{args.output_dir}/{image_key}", image_depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", default="/home/ssbai/datas/anti-spoof_db")
    parser.add_argument(
        "--output_dir", default="/home/ssbai/datas/anti-spoof/train/real_depth")
    parser.add_argument('-c', '--config', type=str,
                        default='configs/mb1_120x120.yml')
    args = parser.parse_args()
    main(args)
