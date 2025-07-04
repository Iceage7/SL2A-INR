# SL²A-INR: Single-Layer Learnable Activation for Implicit Neural Representation  

This repository contains the implementation for the paper:  
**"SL²A-INR: Single-Layer Learnable Activation for Implicit Neural Representation"**

---

## 📄 Abstract

Implicit Neural Representation (INR), leveraging a neural network to transform coordinate input into corresponding attributes, has recently driven significant advances in several vision-related domains. However, the performance of INR is heavily influenced by the choice of the nonlinear activation function used in its multilayer perceptron (MLP) architecture. To date, multiple nonlinearities have been investigated, but current INRs still face limitations in capturing high-frequency components and diverse signal types. 

We show that these challenges can be alleviated by introducing a novel approach in INR architecture. Specifically, we propose **SL²A-INR**, a hybrid network that combines a single-layer learnable activation function with an MLP that uses traditional ReLU activations. Our method performs superior across diverse tasks, including image representation, 3D shape reconstruction, and novel view synthesis. Through comprehensive experiments, SL²A-INR sets new benchmarks in accuracy, quality, and robustness for INR.

---

## 🔧 Installation

Install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Train the model using the `train.py` script. You need to specify the path to the input data and the INR model to use.

### 🔢 Training Arguments

- `--input`: Path to the input image (e.g., `./data/00.png`)  
- `--inr_model`: The INR model to train. Options include:  
  `gauss`, `relu`, `siren`, `wire`, `finer`, and our proposed `sl2a`.

### 🧪 Example

To train the **SL²A-INR** model on a sample image:

```bash
python train.py --input ./data/00.png --inr_model sl2a
```

For more options and configurations, please refer to the `train.py` file.

---

## ✅ To-Do

- [ ] Add Novel View Synthesis (NeRF) code.

---

## 🙏 Acknowledgements

We thank the authors of the following repositories for their publicly available code, which greatly supported our research:

- [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN)
- [WIRE](https://github.com/vishwa91/wire)
- [FINER](https://github.com/liuzhen0212/FINER)

---

## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@article{heidari2024sl,
  title={SL\textsuperscript{2}A-INR: Single-Layer Learnable Activation for Implicit Neural Representation},
  author={Heidari, Moein and Rezaeian, Reza and Azad, Reza and Merhof, Dorit and Soltanian-Zadeh, Hamid and Hacihaliloglu, Ilker},
  journal={arXiv preprint arXiv:2409.10836},
  year={2024}
}
```
---

