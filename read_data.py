import wandb

wandb.login()

api = wandb.Api()
artifact = api.artifact('fencepainters/raifhack/RawDataset:latest')
artifact.download(root="data/initial_data")
