# $(FL)^2$: Overcoming Few Labels in Federated Semi-Supervised Learning (NeurIPS '24)
This is the official PyTorch implementation of "$(FL)^2$: Overcoming Few Labels in Federated Semi-Supervised Learning (NeurIPS '24)" by [Seungjoo Lee](https://seungjoo.com), Thanh-Long V. Le, [Jaemin Shin](https://jaemin-shin.github.io/), and [Sung-Ju Lee](https://sites.google.com/site/wewantsj/).

## Getting Started
- We manage environment with conda. Please install [Anaconda](https://www.anaconda.com/) first.
- Create conda environment:
    ```ruby
    conda create -n <env_name> python==3.8.18
    conda activate <env_name>
    ```
- Install the required package
    ```ruby
    pip install -r requirements.txt
    ```
## Instruction
- To run a specific experiment, modify the experiment's hyperparameters and other configuration, please look at the config file at `src/config/config.yml`.

- To run $(FL)^2$, set the `algorithm` as `flfl` and `mixup` value as 0.

    ```ruby
    # From FLFL directory
    cd src
    python train_classifier_flfl.py
    ```
