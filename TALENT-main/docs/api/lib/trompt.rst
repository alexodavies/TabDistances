**Trompt**
==========

A prompt-based deep neural network for tabular data that separates learning into intrinsic column features and sample-specific feature importance.


Functions
~~~~~~~~~

.. code-block:: python

    class LinearEmbeddings(nn.Module)

Linear embeddings for continuous features.

**Parameters:**

* **n_features** *(int)* - Number of continuous features.
* **d_embedding** *(int)* - Embedding dimension.

**Input Shape:**

`(*, n_features)`

**Output Shape:**

`(*, n_features, d_embedding)`


.. code-block:: python

    class CategoricalEmbeddings1d(nn.Module)

Embeddings for categorical features with support for unknown categories.

**Parameters:**

* **cardinalities** *(list[int])* - List of category counts for each feature.
* **d_embedding** *(int)* - Embedding dimension.

**Input Shape:**

`(*, n_cat_features)`

**Output Shape:**

`(*, n_cat_features, d_embedding)`


.. code-block:: python

    class ImportanceGetter(nn.Module)

Computes feature importance scores using prompts and input features (Figure 3 Part 1).

**Parameters:**

* **P** *(int)* - Number of prompts.
* **C** *(int)* - Total number of features (numerical + categorical).
* **d** *(int)* - Embedding dimension.

**Input:**

* **O** *(Tensor)* - Previous output tensor.

**Output:**

* **Tensor** - Feature importance matrix.


.. code-block:: python

    class TromptEmbedding(nn.Module)

Combines numerical and categorical embeddings (Figure 3 Part 2).

**Parameters:**

* **n_num_features** *(int)* - Number of numerical features.
* **cat_cardinalities** *(list[int])* - List of category counts for categorical features.
* **d** *(int)* - Embedding dimension.

**Inputs:**

* **x_num** *(Tensor)* - Numerical feature tensor.
* **x_cat** *(Tensor)* - Categorical feature tensor.

**Output:**

* **Tensor** - Combined embeddings.


.. code-block:: python

    class Expander(nn.Module)

Expands input features using a linear layer and group normalization (Figure 3 Part 3).

**Parameters:**

* **P** *(int)* - Number of prompts.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Expanded tensor.


.. code-block:: python

    class TromptCell(nn.Module)

Complete Trompt cell that combines embedding, importance calculation, and expansion.

**Parameters:**

* **n_num_features** *(int)* - Number of numerical features.
* **cat_cardinalities** *(list[int])* - List of category counts for categorical features.
* **P** *(int)* - Number of prompts.
* **d** *(int)* - Embedding dimension.

**Inputs:**

* **x_num** *(Tensor)* - Numerical feature tensor.
* **x_cat** *(Tensor)* - Categorical feature tensor.
* **O** *(Tensor)* - Previous output tensor.

**Output:**

* **Tensor** - Processed tensor.


.. code-block:: python

    class TromptDecoder(nn.Module)

Decodes the output of the Trompt cells into final predictions.

**Parameters:**

* **d** *(int)* - Input dimension.
* **d_out** *(int)* - Output dimension.

**Input:**

* **o** *(Tensor)* - Input tensor from Trompt cells.

**Output:**

* **Tensor** - Decoded predictions. 


**Reference**

Kuan-Yu Chen, Ping-Han Chiang, Hsin-Rung Chou, Ting-Wei Chen, and Tien-Hao Chang. **Trompt: Towards a Better Deep Neural Network for Tabular Data**. arXiv:2305.18446 [cs.LG], 2023. `<https://arxiv.org/abs/2305.18446>`_


