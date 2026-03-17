import argparse
from Configs.parse_config import ConfigParser

def main(config):
    train_dataset = config.init_dataset(split="train")
    val_dataset = config.init_dataset(split="val")
    model = config.init_model()
    criterion = config.get_loss()
    trainer = config.init_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion
    )
    trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config JSON")
    args = parser.parse_args()
    config = ConfigParser.from_json(args.config)
    main(config)