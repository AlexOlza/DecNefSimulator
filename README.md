# DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models

### Alexander Olza, Roberto Santana and David Soto
[ArXiV preprint](https://arxiv.org/abs/2511.14555)
<!-- Required for howfairis -->
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/12727/badge)](https://bestpractices.coreinfrastructure.org/projects/12727)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B%20%20%E2%97%8F%20%20%E2%97%8F-yellow)](https://fair-software.eu)

*In compliance with FAIR practices, DecNefSimulator is registred in the NITRC public software registry, [here](https://www.nitrc.org/projects/decnefsimulator)*
*I am working on pip packaging. Stay tuned!* 😺


Decoded Neurofeedback (DecNef) is a flourishing non-invasive approach to brain modulation with wide-ranging applications in neuromedicine and cognitive neuroscience. However, progress in DecNef research remains constrained by subject-dependent learning variability, reliance on indirect measures to quantify progress, and the high cost and time demands of experimentation.

We present DecNefSimulator, a modular and interpretable simulation framework that formalizes DecNef as a machine learning problem. Beyond providing a virtual laboratory, DecNefSimulator enables researchers to model, analyze and understand neurofeedback dynamics. Using latent variable generative models as simulated participants, DecNefSimulator allows direct observation of internal cognitive states and systematic evaluation of how different protocol designs and subject characteristics influence learning.

We demonstrate how this approach can (i) reproduce empirical phenomena of DecNef learning, (ii) identify conditions under which DecNef feedback fails to induce learning, and (iii) guide the design of more robust and reliable DecNef protocols *in silico* before human implementation.

In summary, DecNefSimulator bridges computational modeling and cognitive neuroscience, offering a principled foundation for methodological innovation, robust protocol design, and ultimately, a deeper understanding of DecNef-based brain modulation.

If this work has been useful to you, please include a citation in your publication :blush:

```bibtex
@article{olza2026decnefsimulator,
  title={DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models},
  author={Olza, Alexander and Santana, Roberto and Soto, David},
  eprint={2511.14555},
  archivePrefix={arXiv},
  year={2026},
  url={https://arxiv.org/abs/2511.14555}
}
```

## Features :sparkles:
In our paper, we model the human participant with a VAE and a simple learning behaviour, but other researchers can implement new custom models acording to their expert knowledge. DecNefSimulator is highly customizable, and we encourage users to implement their custom components.

Click [here](https://github.com/AlexOlza/DecNefSimulator/tree/main/run_exp#readme) to learn the basic usage!


## Installation and resources :wrench:

This project uses Python 3.10. We recommend using pip to manage dependencies. We make use of `pytorch` and have conducted our experiments on a NVIDIA GeForce RTX 2080 GPU; however, the code is prepared to run in CPU as well.

The auxiliary repository [DecNefSimulator_install](https://github.com/AlexOlza/DecNefSimulator_install) provides an installation script (`install.sh`), and is also be used to manage new dependencies during active development. Given that Python 3 and pip are available, DecNefSimulator is set up from scratch in a dedicated environment by running the installation script. We kindly advise you to follow that procedure. 
<details>
<summary>
You may click this dropdown for a summary of what will happen during the install.
</summary>
1. External repos [MindSimulator](https://github.com/Noirebao/MindSimulator) and [MindEye2](https://github.com/MedARC-AI/MindEyeV2) will be cloned
2. Required packages will be installed via pip. This includes:
   - scikit-learn, pytorch and other Machine Learning libraries
   - nilearn and nibabel: neuroimaging libraries
   - matplotlib, seaborn and other plotting libraries
3. External repo [CLIP](https://github.com/openai/CLIP) (from openAI) and package `dalle2-pytorch` will be installed via git, without dependencies. This is required by MindSimulator.
4. Installed packages will be written to a `requirements.txt` file.
</details>

Additionally, for experiments with synthetic fMRI, MindEye models and data must be downloaded from [HuggingFace](https://huggingface.co/datasets/pscotti/mindeyev2), specifically, files `betas_all_subj0X_fp32_renorm.hdf5` (where `X` is the subject index), `file`and `file`. For optional result visualization using neuroimaging libraries, we used the file `brain_region_masks.hdf5`.

## License :scroll:
This repository is distributed under the AGPL 3.0 license. See the [LICENSE](./LICENSE) file for more details, or visit the [AGPL 3.0 homepage](https://www.gnu.org/licenses/agpl-3.0.en.html).
