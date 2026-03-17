**TabPFN**
==========

A general model which involves the use of pre-trained deep neural networks that can be directly applied to any tabular task.


Functions
~~~~~~~~~

.. code-block:: python

    class TransformerModel(nn.Module)

Main transformer model for TabPFN.

**Parameters:**

* **encoder** - Input encoder for features.
* **n_out** *(int)* - Output dimension.
* **ninp** *(int)* - Input dimension.
* **nhead** *(int)* - Number of attention heads.
* **nhid** *(int)* - Hidden dimension.
* **nlayers** *(int)* - Number of transformer layers.
* **dropout** *(float, optional, Default is 0.0)* - Dropout rate.
* **style_encoder** - Style encoder for additional features.
* **y_encoder** - Target encoder.
* **pos_encoder** - Positional encoder.
* **decoder** - Output decoder.
* **input_normalization** *(bool, optional, Default is False)* - Whether to normalize input.
* **init_method** - Weight initialization method.
* **pre_norm** *(bool, optional, Default is False)* - Whether to use pre-normalization.
* **activation** *(str, optional, Default is 'gelu')* - Activation function.
* **recompute_attn** *(bool, optional, Default is False)* - Whether to recompute attention.
* **num_global_att_tokens** *(int, optional, Default is 0)* - Number of global attention tokens.
* **full_attention** *(bool, optional, Default is False)* - Whether to use full attention.
* **all_layers_same_init** *(bool, optional, Default is False)* - Whether all layers have same initialization.
* **efficient_eval_masking** *(bool, optional, Default is True)* - Whether to use efficient evaluation masking.


.. code-block:: python

    class SeqBN(nn.Module)

Sequential batch normalization layer.

**Parameters:**

* **d_model** *(int)* - Model dimension.


.. code-block:: python

    class TransformerEncoderDiffInit(Module)

Transformer encoder with different initialization for each layer.

**Parameters:**

* **encoder_layer_creator** - Function to create encoder layers.
* **num_layers** *(int)* - Number of encoder layers.
* **norm** - Layer normalization component.


.. code-block:: python

    class EmbeddingEncoder(nn.Module)

Embedding encoder for categorical features.

**Parameters:**

* **num_embeddings** *(int)* - Number of embeddings.
* **embedding_dim** *(int)* - Embedding dimension.
* **padding_idx** *(int, optional)* - Padding index.


.. code-block:: python

    class LinearEncoder(nn.Module)

Linear encoder for numerical features.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **output_dim** *(int)* - Output dimension.


.. code-block:: python

    class TransformerEncoderLayer(nn.Module)

Single transformer encoder layer.

**Parameters:**

* **d_model** *(int)* - Model dimension.
* **nhead** *(int)* - Number of attention heads.
* **dim_feedforward** *(int)* - Feedforward dimension.
* **dropout** *(float, optional, Default is 0.1)* - Dropout rate.
* **activation** *(str, optional, Default is 'relu')* - Activation function.
* **pre_norm** *(bool, optional, Default is False)* - Whether to use pre-normalization.
* **recompute_attn** *(bool, optional, Default is False)* - Whether to recompute attention. 

**References:**

Noah Hollmann and Samuel Müller and Katharina Eggensperger and Frank Hutter. **TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second**. arXiv:2207.01848 [cs.LG], 2023. `<https://arxiv.org/abs/2207.01848>`_


