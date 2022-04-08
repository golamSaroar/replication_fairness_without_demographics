import os
from absl import app

dataset = "uci_adult"
adversary_loss = "ce_loss"
adversary_label = True
batch_size = 32
learning_rate = 0.001

def get_model_metadata():
    model_name = "adversarial_reweighting"
    base_dir = "/temp"
    config = "{}/{}/{}_{}_{}_{}_{}".format(dataset, model_name, adversary_loss, adversary_label, str(batch_size), str(learning_rate), str(learning_rate))
    directory = os.path.join(base_dir, config)
    return directory, model_name

def instantiate():
    model_dir, model_name = get_model_metadata()
    # TODO: Load Dataset
    # Work In Progress

def main(_):
  instantiate()

if __name__ == "__main__":
  app.run(main)