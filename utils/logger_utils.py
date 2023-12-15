import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from utils.image_utils import psnr
from argparse import Namespace
from icecream import ic
import csv

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        ic("report")
        headers = [
            "iteration",
            "set",
            "l1_loss",
            "psnr",
            "ssim",
            "lpips",
            "file_size",
            "elapsed",
        ]
        csv_path = os.path.join(scene.model_path, "metric.csv")
        # Check if the CSV file exists, if not, create it and write the header
        file_exists = os.path.isfile(csv_path)
        save_path = os.path.join(
            scene.model_path,
            "point_cloud/iteration_" + str(iteration),
            "point_cloud.ply",
        )
        # Check if the file exists
        if os.path.exists(save_path):
            # Get the size of the file
            file_size = os.path.getsize(save_path)
            file_size_mb = file_size / 1024 / 1024  # Convert bytes to kilobytes
        else:
            file_size_mb = None

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header

        torch.cuda.empty_cache()
        validation_configs = ({"name": "test", "cameras": scene.getTestCameras()},)
        #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type="vgg").mean().double()

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                lpips_test /= len(config["cameras"])
                # sys.stderr.write(f"Iteration  {iteration} Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test} SSIM {ssim_test} LPIPS {lpips_test}\n")
                # sys.stderr.flush()
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                        iteration,
                        config["name"],
                        l1_test,
                        psnr_test,
                        ssim_test,
                        lpips_test,
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - lpips",
                        lpips_test,
                        iteration,
                    )
                if config["name"] == "test":
                    with open(csv_path, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=headers)
                        writer.writerow(
                            {
                                "iteration": iteration,
                                "set": config["name"],
                                "l1_loss": l1_test.item(),
                                "psnr": psnr_test.item(),
                                "ssim": ssim_test.item(),
                                "lpips": lpips_test.item(),
                                "file_size": file_size_mb,
                                "elapsed": elapsed,
                            }
                        )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )

        torch.cuda.empty_cache()
