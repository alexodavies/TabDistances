**TabPTM**
==========

A general method for tabular data that standardizes heterogeneous datasets using meta-representations, allowing a pre-trained model to generalize to unseen datasets without additional training.


Functions
~~~~~~~~~

.. code-block:: python

    def prepare_meta_feature(X, Y, args)

Prepares class centers for classification tasks by sampling from training data.

**Parameters:**

* **X** *(dict)* - Dataset splits (keys: 'train', 'val', 'test').
* **Y** *(dict)* - Labels for dataset splits.
* **args** - Command-line arguments (must contain `centers_num` and `seed`).

**Returns:**

* **centers** *(list)* - List of numpy arrays where each array contains sampled centers for a class.


.. code-block:: python

    def prepare_meta_feature_regression(X, Y, args, dataname=None, is_meta=False)

Prepares sampled data points for regression tasks.

**Parameters:**

* **X** *(dict)* - Dataset splits.
* **Y** *(dict)* - Target values for dataset splits.
* **args** - Command-line arguments (must contain `centers_num` and `seed`).
* **dataname** *(str, optional, Default is None)* - Dataset name.
* **is_meta** *(bool, optional, Default is False)* - Whether this is meta-data.

**Returns:**

* **centers** *(np.ndarray)* - Sampled data points concatenated with targets.


.. code-block:: python

    def to_tensors(data: ArrayDict) -> Dict[str, torch.Tensor]

Converts numpy arrays in a dictionary to PyTorch tensors.

**Parameters:**

* **data** *(dict)* - Dictionary with numpy array values.

**Returns:**

* **dict** - Dictionary with PyTorch tensors.


.. code-block:: python

    class TabPTMData(Dataset)

Dataset class for tabular data with numerical features.

**Parameters:**

* **dataset** - Dataset object (must have `is_regression` attribute).
* **X** *(dict)* - Feature splits.
* **Y** *(dict)* - Label splits.
* **y_info** - Label information.
* **part** *(str)* - Data split ('train', 'val', 'test').

**Methods:**

* **get_dim_in(self)** - Returns the input feature dimension.
* **get_categories(self)** - Returns categorical feature information (always None for this class).
* **__len__(self)** - Returns the number of samples in the dataset.
* **__getitem__(self, i)** - Retrieves a data sample and its label.


**References:**

Han-Jia Ye, Qi-Le Zhou, Huai-Hong Yin, De-Chuan Zhan, and Wei-Lun Chao. **Rethinking Pre-Training in Tabular Data: A Neighborhood Embedding Perspective**. arXiv:2311.00055 [cs.LG], 2025. `<https://arxiv.org/abs/2311.00055>`_