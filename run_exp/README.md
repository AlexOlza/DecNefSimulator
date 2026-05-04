
## How to use DecNefSimulator 🧠✨

### Preparing DecNefSimulator for systematic use

**1. Training the generative model:**
The unsupervised model will learn the semantic structure of the data. This is analogous to how humans learn similarities, differences and relationships in our surroundings. The output is an abstract cognitive space in which semantic relationships between experienced concepts are reflected. In DecNefSimulator, by training generative model, we obtain a latent space with semantic structure.
 
   - *Input:* A sufficiently big and diverse dataset from the modality of interest, given resource constraints (e.g. real or synthetic fMRI data from several categories for the same subject)
   - *How to:* Run `train_VAE.py` or adapt it to suit your needs if you wish to employ a non-VAE model
   - *Output:* A trained latent variable generative model, saved to disk in pytorch format

**2. Declaring a learning rule:** In a real world scenario, the human participant has some kind of decision process determining their reaction to the feedback, whether they are aware of their decision-making or it happens inadvertedly. In DecNefSimulator, we model this effect by proposing a learning rule that represents the specific subject's strategy. Some examples of learning rules are given in [the documentation about components](https://github.com/AlexOlza/DecNefSimulator/blob/main/components/README.md).
  - *Input:* The experimenter's assumptions on cognitive behaviour, backed by previous literature or expert knowledge
  - *Output:* A python function that takes a point in the cognitive space and a feedback value, and provides a new point on the cognitive space

 
 **3. Training the feedback system:** In empirical DecNef, the feedback system is trained on fMRI data acquired while the subject is exposed to relevant and known stimuli (for example, during a visual perception task involving cats -- the target category -- and dogs -- the alternative --). Similarly, in DecNefSimulator, we train the feedback system of choice using data (images or fMRI activations) from the target category and, if needed, an alternative "control" category. This process is analogous in both empirical and simulated DecNef.
   - *Input:* Data from the target category and, depending on the choice of feedback system, auxiliary data (e.g. if the feedback will be based on a binary classifier, this requires choosing and providing data from an alternative class)
   - *How to:* Run `train_classifier.py` or adapt it for you custom implemented feedback algorithms
   - *Output:* A trained feedback model saved to disk in pytorch format


### Conducting systematic simulations
Once the components described above are available, the function `compute_single_trajectory` from the `protocols`module makes a simulation of the induction stage of DecNef training. This function can be used systematically with different starting cognitive states an different randomness conditions (as shown in our paper), or it can also be used to study other aspects of interest (for instance, the effect of different learning rules using the same randomness conditions and the same feedback system, and/or the effect of slight perturbations of the same learning rule).

```python
import numpy as np
from components.generators import CustomGenerativeModel
from components.classifiers import CustomFeedbackSystem
from components.update_rules import custom_learning_rule
from protocols.decnef_loops import compute_single_trajectory

generator = CustomGenerativeModel(*args, **kwargs).load("filename_of_pretrained_generative_model.pt")
classifier = CustomFeedbackSystem(*args, **kwargs).load("filename_of_pretrained_generative_model.pt")
L = custom_learning_rule

        generated_images,\
        trajectory,\
        probabilities,\
        all_probabilities,\
        sigma =  compute_single_trajectory(generator, classifier,
                                           trajectory_random_seed,          # Fixed seed for reproducibility
                                           train_loader,
                                           target_class,                    # Integer denoting the target class of DecNef training
                                           update_rule_func=L,              # Learning rule
                                           p_scale_func,
                                           z_current= torch.Tensor(z0),     # The initial cognitive state
                                           trajectory_name=trajectory_name, # A string for result saving
                                           n_iter = 500,                    # Number of DecNef iterations
                                           lambda_ = lambda_,               # Parameter required by the learning rule
                                           )
        np.savez_compressed(trajectory_filename, 
                            generated_images = generated_images,
                            trajectory = trajectory,
                            probabilities = probabilities,
                            all_probabilities = all_probabilities,
                            sigma = sigma
                            )

```
