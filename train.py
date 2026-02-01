import os
import hydra
from omegaconf import DictConfig, OmegaConf
from csref.core.trainer import Trainer
from csref.utils.distributed import seed_everything
from csref.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@hydra.main(config_path="configs", config_name="config_csa", version_base="1.3.2")
def main(cfg: DictConfig):
    # Setup global seed
    if hasattr(cfg.train, 'seed'):
        seed_everything(cfg.train.seed)
    
    # Debug info
    if os.environ.get("RANK", "0") == "0":
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    trainer = Trainer(cfg)
    trainer.setup()
    trainer.fit()

if __name__ == "__main__":
    main()
