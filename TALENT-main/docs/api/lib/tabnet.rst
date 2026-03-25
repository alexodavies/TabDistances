**TabNet**
==========

A tree-mimic method using sequential attention for feature selection, offering interpretability and self-supervised learning capabilities.


Functions
~~~~~~~~~

.. code-block:: python

    class TabNetClassifier(TabModel)

TabNet model for classification tasks.

**Parameters:**

* **optimizer_fn** *(torch.optim.Optimizer)* - PyTorch optimizer function.
* **optimizer_params** *(dict)* - Parameters for the optimizer.
* **scheduler_fn** *(torch.optim.lr_scheduler._LRScheduler)* - PyTorch scheduler function.
* **scheduler_params** *(dict)* - Parameters for the scheduler.
* **mask_type** *(str)* - Type of mask to use ('sparsemax' or 'entmax').
* **n_d** *(int)* - Dimension of prediction layer.
* **n_a** *(int)* - Dimension of attention layer.
* **n_steps** *(int)* - Number of decision steps.
* **gamma** *(float)* - Relaxation parameter for feature selection.
* **n_ind** *(int)* - Number of independent features.
* **n_shared** *(int)* - Number of shared features.
* **cat_idxs** *(list)* - List of categorical feature indices.
* **cat_dims** *(list)* - List of categorical feature dimensions.
* **cat_emb_dim** *(int)* - Embedding dimension for categorical features.
* **n_independent** *(int)* - Number of independent Gated Linear Units.
* **n_shared** *(int)* - Number of shared Gated Linear Units.
* **epsilon** *(float)* - Epsilon for numerical stability.
* **virtual_batch_size** *(int)* - Virtual batch size for Ghost Batch Normalization.
* **momentum** *(float)* - Momentum for batch normalization.
* **clip_value** *(float)* - Gradient clipping value.
* **verbose** *(int)* - Verbosity level.


.. code-block:: python

    class TabNetRegressor(TabModel)

TabNet model for regression tasks.

**Parameters:**

* **optimizer_fn** *(torch.optim.Optimizer)* - PyTorch optimizer function.
* **optimizer_params** *(dict)* - Parameters for the optimizer.
* **scheduler_fn** *(torch.optim.lr_scheduler._LRScheduler)* - PyTorch scheduler function.
* **scheduler_params** *(dict)* - Parameters for the scheduler.
* **mask_type** *(str)* - Type of mask to use ('sparsemax' or 'entmax').
* **n_d** *(int)* - Dimension of prediction layer.
* **n_a** *(int)* - Dimension of attention layer.
* **n_steps** *(int)* - Number of decision steps.
* **gamma** *(float)* - Relaxation parameter for feature selection.
* **n_ind** *(int)* - Number of independent features.
* **n_shared** *(int)* - Number of shared features.
* **cat_idxs** *(list)* - List of categorical feature indices.
* **cat_dims** *(list)* - List of categorical feature dimensions.
* **cat_emb_dim** *(int)* - Embedding dimension for categorical features.
* **n_independent** *(int)* - Number of independent Gated Linear Units.
* **n_shared** *(int)* - Number of shared Gated Linear Units.
* **epsilon** *(float)* - Epsilon for numerical stability.
* **virtual_batch_size** *(int)* - Virtual batch size for Ghost Batch Normalization.
* **momentum** *(float)* - Momentum for batch normalization.
* **clip_value** *(float)* - Gradient clipping value.
* **verbose** *(int)* - Verbosity level.


.. code-block:: python

    class TabNetNetwork(nn.Module)

The main TabNet network architecture.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **output_dim** *(int)* - Output dimension.
* **n_d** *(int)* - Dimension of prediction layer.
* **n_a** *(int)* - Dimension of attention layer.
* **n_steps** *(int)* - Number of decision steps.
* **gamma** *(float)* - Relaxation parameter for feature selection.
* **n_ind** *(int)* - Number of independent features.
* **n_shared** *(int)* - Number of shared features.
* **cat_idxs** *(list)* - List of categorical feature indices.
* **cat_dims** *(list)* - List of categorical feature dimensions.
* **cat_emb_dim** *(int)* - Embedding dimension for categorical features.
* **n_independent** *(int)* - Number of independent Gated Linear Units.
* **n_shared** *(int)* - Number of shared Gated Linear Units.
* **epsilon** *(float)* - Epsilon for numerical stability.
* **virtual_batch_size** *(int)* - Virtual batch size for Ghost Batch Normalization.
* **momentum** *(float)* - Momentum for batch normalization.
* **mask_type** *(str)* - Type of mask to use ('sparsemax' or 'entmax').


.. code-block:: python

    class TabNetNoEmbeddings(nn.Module)

TabNet network without embeddings for numerical features only.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **output_dim** *(int)* - Output dimension.
* **n_d** *(int)* - Dimension of prediction layer.
* **n_a** *(int)* - Dimension of attention layer.
* **n_steps** *(int)* - Number of decision steps.
* **gamma** *(float)* - Relaxation parameter for feature selection.
* **n_ind** *(int)* - Number of independent features.
* **n_shared** *(int)* - Number of shared features.
* **n_independent** *(int)* - Number of independent Gated Linear Units.
* **n_shared** *(int)* - Number of shared Gated Linear Units.
* **epsilon** *(float)* - Epsilon for numerical stability.
* **virtual_batch_size** *(int)* - Virtual batch size for Ghost Batch Normalization.
* **momentum** *(float)* - Momentum for batch normalization.
* **mask_type** *(str)* - Type of mask to use ('sparsemax' or 'entmax').


.. code-block:: python

    class TabNetDecoder(nn.Module)

Decoder for TabNet pretraining.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **output_dim** *(int)* - Output dimension.
* **n_d** *(int)* - Dimension of prediction layer.
* **n_steps** *(int)* - Number of decision steps.
* **gamma** *(float)* - Relaxation parameter for feature selection.
* **n_ind** *(int)* - Number of independent features.
* **n_shared** *(int)* - Number of shared features.
* **n_independent** *(int)* - Number of independent Gated Linear Units.
* **n_shared** *(int)* - Number of shared Gated Linear Units.
* **epsilon** *(float)* - Epsilon for numerical stability.
* **virtual_batch_size** *(int)* - Virtual batch size for Ghost Batch Normalization.
* **momentum** *(float)* - Momentum for batch normalization.
* **mask_type** *(str)* - Type of mask to use ('sparsemax' or 'entmax'). 


**References:**

Sercan O. Arik and Tomas Pfister. **TabNet: Attentive Interpretable Tabular Learning**. arXiv:1908.07442 [cs.LG], 2020. `<https://arxiv.org/abs/1908.07442>`_
