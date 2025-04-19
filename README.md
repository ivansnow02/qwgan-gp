# Quantum Generative Adversarial Networks (QGANs) Implementations

---

[中文](README_zh.md)

## English

This repository provides implementations of various Quantum Generative Adversarial Network (QGAN) models using PyTorch. It focuses on generating handwritten digits based on the UCI Optical Recognition of Handwritten Digits dataset.

### Features

* **Classic QGAN:** Implementation based on the original QGAN concepts.
* **Improved QGAN:** Classic QGAN with enhancements like dropout in the discriminator.
* **Quantum WGAN-GP:** Implementation using the Wasserstein GAN with Gradient Penalty (WGAN-GP) framework adapted for quantum generators.
* **Quantum WGAN-GP with Minibatch Discrimination:** WGAN-GP QGAN incorporating Minibatch Discrimination in the critic to potentially improve generation diversity.

### Requirements

* Python 3.x
* PyTorch (`torch`, `torchvision`)
* NumPy
* Matplotlib
* TensorBoard (`tensorboard`)
* UCIMLRepo (`ucimlrepo`)
* PennyLane

You can install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ivansnow02/qwgan-gp.git
   cd qwgan-gp
   ```

2. Install the dependencies as listed in the Requirements section.

### Usage

Run the main training script `train.py` from the command line, specifying the desired configuration:

```bash
python train.py --config <config_name>
```

Available configurations (`<config_name>`):

* `classic`
* `improved`
* `wgan_gp`
* `wgan_gp_mbd`

Example:

```bash
python train.py --config wgan_gp
```

### Configuration

Detailed training parameters for each model type (e.g., learning rates, batch size, number of iterations, quantum circuit parameters, image size) are defined within configuration dictionaries at the end of the `train.py` script (e.g., `config_classic`, `config_wgan_gp`). You can modify these dictionaries to experiment with different settings.

### Results and Monitoring

* **TensorBoard:** Training progress, loss values, performance metrics (like D(x), D(G(z)), Critic scores), and generated image samples are logged using TensorBoard. Launch TensorBoard in the project's root directory:

  ```bash
  tensorboard --logdir runs
  ```

  Then navigate to `http://localhost:6006` (or the port specified by TensorBoard) in your web browser.

* **Saved Files:** Generated images and model checkpoints are saved periodically during training within the `runs/<run_name>/` directory, where `<run_name>` corresponds to the configuration being run (e.g., `runs/qwgan_gp/`).
