# Playing Flappy Bird with DQN

This projects implements training of a Deep Q-Networks (DQN) to play the Flappy Bird game. During experimtents 3 models were trained; **DQN**, **DDQN**. **Dueling DDQN**. Trained models can be found in **Releases**.

| Training history      |    Evaluation history |
|-----------------------|-----------------------|
| ![Training history graph](img/train_comparison.png) | ![Evaluation history graph](img/eval_comparison.png) |

### Architecture:

Project uses CNN similar to the one used in paper ["Playing Atari with Deep Reinforcement Learning"](http://arxiv.org/abs/1312.5602) by Mnih et al. (2013). Input to the CNN is a 72x106x4 image, created by stacking cropped, rescaled, gray-scale game frames. The output consists of 2 q-values for possible actions: _jump_ or _no-jump_.

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