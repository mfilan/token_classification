import hydra
from hydra.core.config_store import ConfigStore
from config import NERConfig
from typing import Optional
cs = ConfigStore.instance()
cs.store(name='ner_config', node=NERConfig)


@hydra.main(config_path='conf', config_name='config', version_base="1.2")
def main(cfg: Optional[NERConfig]):
    print(cfg.dataset.label2id)


if __name__ == "__main__":
    main()
