import wandb

wandb.login()

api = wandb.Api()
artifact = api.artifact('fencepainters/raifhack/SplitDataset:latest')
artifact.download(root="eden/data/split_data")
