![](images/cubegpt4-logo.webp)

# CubeGPT-3 and CubeGPT-4: Scaling up Transformer-based Rubik's Cube Solvers
A research report about approaching optimality.

## Introduction

Solving Rubik’s Cubes is hard! There are on the order of 10^19 possible Rubik’s Cube states, and only one winning state. Any standard algorithms like A* search, which explore a significant fraction of possible states, are rendered useless. We have to prune trees extremely aggressively, which is where an estimator trained via machine learning comes in.

![](images/scrambled_to_solved.png)

All 3D renders in this project made using a 3D renderer I built, by hand.

So does the AI work? Yes! For instance, it finds the following 18-move solution to the cube pictured above:

https://github.com/user-attachments/assets/dd51b826-f864-45e7-99cc-ccca08a05a2e

### Previous Work

This project builds significantly off of the DeepCubeA and CayleyPy papers. It also builds off <a href=https://github.com/stanleytheli/cubegpt>my previous work on training transformers to solve Rubik’s Cubes, CubeGPT-2</a>. Essentially, the strongest model I tested in solving was CubeGPT-2g-PFT, an out-of-the-box transformer architecture running the GeLU activation trained over 25M samples. The strongest model I trained was CubeGPT-CLS-t154m-PFT, which was a classifier model trained on just over 200M samples. 

In this work, CubeGPT-3 and CubeGPT-4, I improve the model architecture and training pipeline, and devote more compute to larger-scale experiments, getting significantly closer to optimality. We get multiple step improvements as a result. Experiments also reveal more about the nature of training Rubik's Cube models.

## Training

### CubeGPT-2 → CubeGPT-3:

CubeGPT-3 updates over CubeGPT-2 using more modern research and explicitly implements the symmetries of the Rubik's Cube.

![](images/iteration_loss.webp)

In order of being added:

* **GPU optimization**: Moving the data pipeline onto the GPU sped up training ~2x and inference ~3x. TF32 matmuls add another ~1.5x for very little loss in quality.
* **No classifier token**: We now apply a linear layer over the whole token sequence instead of predicting from a special 'cls-token'.
* **Canonicalization**: Rotating or mirroring a cube and relabeling the colors gives the same puzzle. That is 48 symmetries, which we enforce with a cheap preprocessing step.
* **Gated MLP**: f(x) = C(Ax * SiLU(Bx)) instead of f(x) = B(GeLU(Ax)).
* **Mixture of Experts (MoE)**: Deepseek V3-style, 8 experts with 1 active, using explicitly tuned expert biases for load balancing. This adds capacity while holding the number of active parameters the same as CubeGPT-2.

Trained for 307M samples, this is CubeGPT-3-8A1-t300m.

### CubeGPT-3 → CubeGPT-3.1

CubeGPT-3.1 is the same architecture as CubeGPT-3, but with a better learning rate scheduler. We find that the LR scheduler for CubeGPT-3 was significantly too aggressive, cutting LR too early for improvements to "settle in". 

We implement an explicit LR schedule which warms up, holds a high learning rate (5.6e-4) for the most of the run, then cools down to zero near the end. Holding a higher LR allows us to train on higher batches better, so we also increase batch size to 1024. 

![](images/readme/lr_vs_samples.png)

That alone takes the loss from 0.1308 to 0.1073, and significantly improves all metrics.

![](images/readme/loss_vs_samples.png)

### CubeGPT-3.1 → CubeGPT-4

We notice two problems with CubeGPT-3.1: (1) the MoE routing is very noisy which causes routing to change very fast in training, and (2) several of the layers are effectively unused. We measure this by ablating the layers, removing them replacing with just their average value, then measuring the loss. In CubeGPT-3.1, the first layer's MoE FFN can be removed while not impacting the loss at all, and several more have less than 5% impact.

![](images/readme/block_usage.png)

We introduce the CubeGPT-4 architecture which stacks attention layers and then puts the FFNs after, in series. We get rid of MoE because the routing changing so fast indicated that expert specialization wasn't helping. Experiments show that these architectures are able to achieve equal and lower loss than the standard transformer. These architectures also run significantly faster because they less active parameters and no MoE routing overhead. We see that CubeGPT-4 makes more usage of all of its layers.

![](images/readme/experiments_architecture_new_recipe.png)


We also optimize the training pipeline, introducing prenorms instead of postnorms, and using AdamW with hyperparameters more suited to high learning rates.

![](images/readme/experiments_training.png)

We also train for much longer, with a more explicit learning rate schedule. The first 230M samples are at 4e-4, the next 300M at 2e-4, the next 300M at 1e-4, and then the final stretch of 300M where the learning rate linearly decays to zero.

![](images/readme/loss_vs_log_samples.png)

The result is a model with a fifth of the parameters of CubeGPT-3.1 that is better on every metric.

|Model|Training Examples (Millions)|Mean Squared Error|Within-3-Accuracy| Within-4-Accuracy|Accuracy (When rounding to nearest int)|
| --- | --- | --- | --- | --- | --- |
|CubeGPT-2g-PFT | 26.2 | 4.70 | 82.49% | 91.37% | 37.88% |
|CubeGPT-CLS-t154mPFT | 204.8 | 4.30 | 83.96% | 92.40% | 42.50% |
|CubeGPT-3-8A1-t300m | 307.2 |  4.01 | 84.8% | 93.0% | 44.9%
|CubeGPT-3-1-t300m | 307.2 |  3.01 | 88.83% | 95.56% | 50.91%
|CubeGPT-4-t1b | 1,152 | **2.68** | **90.22%** | **96.30%** | **53.84%**



## Solving

To solve a cube we run a beam search guided by the model's distance estimates, keeping the best `width` states at every step. MAO is "moves above optimal": the average solution length minus the average optimal solution length (about 17.7).

|Model| Width | Solve Rate | Average Moves | MAO |
| ---| --- | --- | --- | --- |
|CubeGPT-2g-PFT| 4000 | 99% | 21.1 | 3.4 |
|CubeGPT-3-8A1-t300m| 4000 | 100% | 20.3 | 2.6 |
|CubeGPT-3-8A1-t100mPFT| 4000 | 99% | 20.2 | 2.5 |
|CubeGPT-3-8A1-t100mPFT| 16000 | 100% | 19.6 | 1.9 |
|CubeGPT-3-8A1-t100mPFT| 64000 | 100% | 19.0 | 1.3 |
|CubeGPT-3-1-t300m| 4000 | 100% | 19.13 | 1.43 |
|CubeGPT-3-1-t300m| 16000 | 100% | 18.62 | 0.92 |
|CubeGPT-3-1-t300m| 64000 | 100% | 18.23 | 0.53 |
|CubeGPT-4-t1b| 4000 | 100% | **18.78** | **1.08** |
|CubeGPT-4-t1b| 16000 | 100% | **18.35** | **0.65** |
|CubeGPT-4-t1b| 64000 | 100% | **18.15** | **0.45** |

The CubeGPT-3.1 and CubeGPT-4 rows use the same 200 random scrambles (100 at width 64000). The CubeGPT-2 and CubeGPT-3 rows are from earlier runs on different scrambles.

![](images/readme/width_scaling.png)

Better models bring significant gains. CubeGPT-4 at width 4000 outperforms CubeGPT-3 at width 64000, CubeGPT-4. At width 4000, CubeGPT-4 takes less than five seconds per solve. Scaling is also buying less and less, even on the log scale, because we are bumping up against the theoretical optimal average of 17.7. Earlier models failed to solve small numbers of cubes; the new models are more reliable and do not fail to solve in our experiments.

## More Solve Videos

https://github.com/user-attachments/assets/d71554ae-2c3d-4c95-9c00-17c6c1f9edbc

https://github.com/user-attachments/assets/bf7bce2e-1663-4972-af68-6b5380861241
