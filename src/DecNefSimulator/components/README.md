# We are changing the repo layout for packaging! This will be moved to src/DecNefSimulator/components/README.md
# Components of DecNefSimulator

DecNefSimulator is modular and flexible, in a sense that the theoretical aspects of the framework remain independent from the actual models it uses for simulating the human participant and the learning process. Therefore, we encourage other researchers to develop their own custom components. This text contains explanations on what the different kind of components are, and which minimum requirements they must meet.

## Generator
These are the latent variable generative models, trained in an unsupervised way, which then define the semantic structure of the artificial participant's cognitive space. The paper uses a VAE. A custom Generator class could be:

```python
class CustomGenerator1(torch.nn.Module):
    def __init__(self, z_dim: int, device: str='cuda'):
      """
        Parameters
        ----------
        z_dim : int
            DESCRIPTION: Latent space dimension
        device : str, optional
            DESCRIPTION: Whether to work in CPU or GPU
      """
      super(CustomGenerator1, self).__init__()
      """
      Define the architecture of the custom generative model here
      """
      pass
    
    def forward(self, X): # Required for backpropagation
        pass     
   
    def fit(self, train_loader, epochs): # Required
        """
        Define the training procedure
        """
        pass

    def compute_prototypes(self, data_loader): # Optional
        pass 
```

## Feedback system
The feedback system is defined by the choice of a ML algorithm, generally pretrained on a supervised way on data from the target class and (customarily, despite introducing bias) some other auxiliary class(es). To this date, the common feedback mechanisms are probabilistic classifiers. Hence, we have implemented two for our paper, one suited for images (`CNNClassifier`) and another one for tabular data, in our case voxel activations (`ElasticNetLinearClassifier`). We encourage researchers to implement custom alternatives. The minimal backbone of a feedback model is:

```python
class CustomFeedbackSystem()
  def __init__():
      self.model = ...... # Custom architecture here
  def forward(self, X):
      pass
```

## Update rule
The update rule is a function that encodes the artificial participant's decision-making process. It can be any function of the current state and feedback value, reflecting neurscience-based knowledge or assumptions. We have implemented several of them, but we encourage custom proposals. The minimal requirements are:

```python
def custom_rule(z_current, feedback_current):
    z_next = (whatever custom behaviour we want to study, using feedback)
    return z_next
```

  
  
