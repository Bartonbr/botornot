import pandas as pd
from botornot.data.data_build import build_training_data
from botornot.model.model import bot_or_not
from botornot.training.train import ModelTrainer
from pprint import pprint


def orchestrate_model_build(id_data_location, bids_data_location, model_output_path):
    print("Data Read...")
    raw_data = pd.read_csv(id_data_location)
    bids_data = pd.read_csv(bids_data_location)

    print("Generating Features...")
    joined_data = pd.merge(raw_data, bids_data, on='bidder_id')
    training_data, training_y = build_training_data(joined_data)

    print("Training Model...")
    trainer = ModelTrainer(bot_or_not())
    trainer.train_model(training_data, training_y)

    print("Training Results:" )
    pprint(trainer.train_results)

    print("Testing Results:")
    pprint(trainer.test_results)

    print("Done, model persisted to: ", model_output_path)
    trainer.persist_model(model_output_path)


