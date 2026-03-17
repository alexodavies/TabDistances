**PTaRL**
=========

A regularization-based framework that enhances prediction by constructing and projecting into a prototype-based space.


Functions
~~~~~~~~~

.. code-block:: python

    def run_one_epoch(model, data_loader, loss_func, model_type, config, regularize, ot_weight, diversity_weight, r_weight, diversity, optimizer=None)

Runs one training epoch with optional OT (Optimal Transport) and diversity regularization.

**Parameters:**

* **model** - Neural network model.
* **data_loader** - Data loader for training.
* **loss_func** - Loss function.
* **model_type** *(str)* - Model type (e.g., 'ot' for optimal transport).
* **config** - Configuration dictionary.
* **regularize** *(bool)* - Whether to apply regularization.
* **ot_weight** *(float)* - Weight for OT loss.
* **diversity_weight** *(float)* - Weight for diversity loss.
* **r_weight** *(float)* - Weight for regularization loss.
* **diversity** *(bool)* - Whether to apply diversity regularization.
* **optimizer** *(Optional)* - Optimizer for training.

**Returns:**

* **float** - Average loss for the epoch.


.. code-block:: python

    def run_one_epoch_val(model, data_loader, loss_func, model_type, config, is_regression)

Runs one validation epoch.

**Parameters:**

* **model** - Neural network model.
* **data_loader** - Data loader for validation.
* **loss_func** - Loss function.
* **model_type** *(str)* - Model type.
* **config** - Configuration dictionary.
* **is_regression** *(bool)* - Whether the task is regression.

**Returns:**

* **tuple** - Predictions and ground truth labels.


.. code-block:: python

    def fit_Ptarl(args, model, train_loader, val_loader, loss_func, model_type, config, regularize, is_regression, ot_weight, diversity_weight, r_weight, diversity, seed, save_path)

Fits a PTaRL model with training and validation.

**Parameters:**

* **args** - Command line arguments.
* **model** - Neural network model.
* **train_loader** - Training data loader.
* **val_loader** - Validation data loader.
* **loss_func** - Loss function.
* **model_type** *(str)* - Model type.
* **config** - Configuration dictionary.
* **regularize** *(bool)* - Whether to apply regularization.
* **is_regression** *(bool)* - Whether the task is regression.
* **ot_weight** *(float)* - Weight for OT loss.
* **diversity_weight** *(float)* - Weight for diversity loss.
* **r_weight** *(float)* - Weight for regularization loss.
* **diversity** *(bool)* - Whether to apply diversity regularization.
* **seed** *(int)* - Random seed.
* **save_path** *(str)* - Path to save model.


.. code-block:: python

    def test(model, test_loader, no_ot=False)

Tests a trained model.

**Parameters:**

* **model** - Trained model.
* **test_loader** - Test data loader.
* **no_ot** *(bool, optional, Default is False)* - Whether to disable OT.

**Returns:**

* **tuple** - Predictions and ground truth labels.


.. code-block:: python

    def generate_topic(model, train_loader, n_clusters)

Generates topics from trained model.

**Parameters:**

* **model** - Trained model.
* **train_loader** - Training data loader.
* **n_clusters** *(int)* - Number of clusters/topics.

**Returns:**

* **np.ndarray** - Generated topics. 


**References:**

Hangting Ye, Wei Fan, Xiaozhuang Song, Shun Zheng, He Zhao, Dandan Guo, and Yi Chang. **PTARL: Prototype-based Tabular Representation Learning via Space Calibration**. In *Proceedings of the Twelfth International Conference on Learning Representations*, 2024. `<https://openreview.net/pdf?id=G32oY4Vnm8>`_