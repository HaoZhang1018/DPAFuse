import argparse
import logging
import os.path
import sys
import time
import numpy as np
import torch

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model
model = create_model(opt)
device = model.device

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_times = []
    fused_dir = os.path.join(dataset_dir, "fused")
    recx_dir = os.path.join(dataset_dir, "recX")
    recy_dir = os.path.join(dataset_dir, "recY")
    util.mkdir(fused_dir)
    util.mkdir(recx_dir)
    util.mkdir(recy_dir)

    for i, test_data in enumerate(test_loader):
        img_path = test_data["X_path"][0]  
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{i+1}/{len(test_loader)}] {img_name}")

        X, Y = test_data["X"], test_data["Y"]
        with torch.no_grad():
            latent_X, hidden_X = model.encode(X.to(device))
            latent_Y, hidden_Y = model.encode(Y.to(device))
            latent_X = model.re_model_X(latent_X.to(device))
            latent_Y = model.re_model_Y(latent_Y.to(device))

            X_fea = [latent_X] + hidden_X
            Y_fea = [latent_Y] + hidden_Y

            tic = time.time()
            model.feed_data(X_fea, Y_fea)
            model.test(current_step=0)
            toc = time.time()
            test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        fused = util.tensor2img(visuals["Fused"].squeeze())
        rec_x = util.tensor2img(visuals["Rec_X_img"].squeeze())
        rec_y = util.tensor2img(visuals["Rec_Y_img"].squeeze())
        suffix = opt["suffix"] if opt["suffix"] else ""
        util.save_img(fused, os.path.join(fused_dir, img_name + suffix + ".png"))
        util.save_img(rec_x, os.path.join(recx_dir, img_name + suffix + ".png"))
        util.save_img(rec_y, os.path.join(recy_dir, img_name + suffix + ".png"))

    print(f"Average test time: {np.mean(test_times):.4f}s per image")
    
