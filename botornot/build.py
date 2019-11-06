from botornot.model.model_build import orchestrate_model_build

if __name__ == "__main__":
    orchestrate_model_build("C:/Users/Bryan/data/humanorbot/train.csv",
                            "C:/Users/Bryan/data/humanorbot/bids.csv",
                            "../models/model.pkl")
