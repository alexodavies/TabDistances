**TabICL**
==========

A comparable tabular foundation model with performance on par with TabPFN v2.


Functions
~~~~~~~~~

.. code-block:: python

    class TabICLClassifier(ClassifierMixin, BaseEstimator)

Tabular In-Context Learning classifier with scikit-learn interface.

**Parameters:**

* **n_estimators** *(int, optional, Default is 32)* - Number of estimators for ensemble predictions.
* **norm_methods** *(Optional[str | List[str]], optional, Default is None)* - Normalization methods to apply.
* **feat_shuffle_method** *(str, optional, Default is "latin")* - Feature permutation strategy.
* **class_shift** *(bool, optional, Default is True)* - Whether to apply cyclic shifts to class labels.
* **outlier_threshold** *(float, optional, Default is 4.0)* - Z-score threshold for outlier detection.
* **softmax_temperature** *(float, optional, Default is 0.9)* - Temperature for softmax function.
* **average_logits** *(bool, optional, Default is True)* - Whether to average logits or probabilities.
* **use_hierarchical** *(bool, optional, Default is True)* - Whether to enable hierarchical classification.
* **use_amp** *(bool, optional, Default is True)* - Whether to use automatic mixed precision.
* **batch_size** *(Optional[int], optional, Default is 8)* - Batch size for inference.
* **model_path** *(Optional[str | Path], optional, Default is None)* - Path to pre-trained model.
* **allow_auto_download** *(bool, optional, Default is True)* - Whether to allow auto-download.
* **checkpoint_version** *(str, optional, Default is "tabicl-classifier-v1.1-0506.ckpt")* - Checkpoint version.
* **device** *(Optional[str | torch.device], optional, Default is None)* - Device for computation.
* **random_state** *(int | None, optional, Default is 42)* - Random seed.
* **n_jobs** *(Optional[int], optional, Default is None)* - Number of jobs for parallel processing.
* **verbose** *(bool, optional, Default is False)* - Whether to print verbose output.
* **inference_config** *(Optional[InferenceConfig | Dict], optional, Default is None)* - Inference configuration.

**Methods:**

* **fit(self, X, y)** - Fit the classifier.
* **predict(self, X)** - Predict class labels.
* **predict_proba(self, X)** - Predict class probabilities.
* **_batch_forward(self, Xs, ys, shuffle_patterns=None)** - Forward pass for batch processing.


.. code-block:: python

    class TransformToNumerical

Transforms categorical features to numerical representations.

**Parameters:**

* **norm_methods** *(List[str])* - List of normalization methods.
* **feat_shuffle_method** *(str)* - Feature shuffling method.
* **class_shift** *(bool)* - Whether to apply class shifts.
* **outlier_threshold** *(float)* - Outlier detection threshold.

**Methods:**

* **transform(self, X, y)** - Transform input data.


.. code-block:: python

    class EnsembleGenerator

Generates ensemble members with different transformations.

**Parameters:**

* **n_estimators** *(int)* - Number of ensemble members.
* **norm_methods** *(List[str])* - Normalization methods.
* **feat_shuffle_method** *(str)* - Feature shuffling method.
* **class_shift** *(bool)* - Whether to apply class shifts.

**Methods:**

* **generate(self, X, y)** - Generate ensemble members.


.. code-block:: python

    class TabICL(nn.Module)

TabICL neural network model.

**Parameters:**

* **config** - Model configuration.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Model predictions.


.. code-block:: python

    class InferenceConfig

Configuration for TabICL inference.

**Parameters:**

* **max_classes** *(int)* - Maximum number of classes.
* **max_features** *(int)* - Maximum number of features.
* **model_dim** *(int)* - Model dimension.
* **num_heads** *(int)* - Number of attention heads.
* **num_layers** *(int)* - Number of layers.


.. code-block:: python

    def softmax(x, axis: int = -1, temperature: float = 0.9)

Computes softmax with temperature scaling.

**Parameters:**

* **x** *(Tensor)* - Input tensor.
* **axis** *(int, optional, Default is -1)* - Axis for softmax computation.
* **temperature** *(float, optional, Default is 0.9)* - Temperature parameter.

**Returns:**

* **Tensor** - Softmax output with temperature scaling.


**References:**

Jingang Qu and David Holzmüller and Gaël Varoquaux and Marine Le Morvan. **TabICL: A Tabular Foundation Model for In-Context Learning on Large Data**. arXiv:2502.05564 [cs.LG], 2025. `<https://arxiv.org/abs/2502.05564>`_
