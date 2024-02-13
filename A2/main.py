from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from a2.data.utils import predict
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    h_cfg = HydraConfig.get()
    run_dir = Path(h_cfg.run.dir)
    predict(
        camera_id=cfg.camera.id,
        save_dir=run_dir.joinpath(cfg.save.dir),
        num_frames_to_save=cfg.camera.num_frames,
        ground_truth_label=cfg.processing.ground_truth_label,
        start_delay_seconds=cfg.camera.start_delay_seconds,
        width=cfg.camera.width,
        height=cfg.camera.height,
        roi_width=cfg.camera.roi.width,
        roi_height=cfg.camera.roi.height,
        gamma=cfg.processing.gamma,
        rotations=cfg.processing.template_matching.rotations,
        scales=cfg.processing.template_matching.scales,
        fps=cfg.camera.fps,
        template_labels_file=cfg.processing.template_matching.labels_file,
        template_images_dir=cfg.processing.template_matching.images_dir,
    )


if __name__ == "__main__":
    run()
