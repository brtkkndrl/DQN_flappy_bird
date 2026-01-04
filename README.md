# Playing Flappy Bird with DQN

This projects implements training of Deep Q-Networks (DQN) to play the Flappy Bird game. During experiments 3 models were trained: **DQN**, **Double DQN (DDQN)**, and **Dueling Double DQN (Dueling DDQN)**. The trained models can be found in the **Releases** section.

| Training history      |    Evaluation history |
|-----------------------|-----------------------|
| ![Training history graph](img/train_comparison.png) | ![Evaluation history graph](img/eval_comparison.png) |

### Architecture:

The project uses a Convolutional Neural Network (CNN) similar to the one described in the paper ["Playing Atari with Deep Reinforcement Learning"](http://arxiv.org/abs/1312.5602) by Mnih et al. (2013). Input to the CNN is a 72x106x4 image, created by stacking cropped, rescaled, grayscale game frames. The output consists of 2 Q-values corresponding to the possible actions: _jump_ or _no-jump_.

### Resources:

[DQN framework](https://github.com/brtkkndrl/DQN)

[flappy-bird-gymnasium environment](https://github.com/markub3327/flappy-bird-gymnasium)

### Installation

Create and activate virtual enviroment:

```
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

###  Running the code

Test trained model in real time:

```
python3 flappy_dqn.py --mode test --dirpath models/flappy_dueling_ddqn/ --test_fps 30
```

Training new model:
```
python3 flappy_dqn.py --mode train --hyperparams base_hyperparams.json
```

Finetuning existing model:
```
python3 flappy_dqn.py --mode train --dirpath models/flappy_dueling_ddqn/ --hyperparams tune_hyperparams.json
```

Show instructions:
```
python3 flappy_dqn.py --help
```
