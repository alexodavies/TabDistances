====================================
Dummy Classifier
====================================

Dummy Classifier classical method implementation.

This section contains the Dummy Classifier implementation for classification tasks. Dummy Classifier is a classifier that makes predictions using simple rules, useful as a baseline for comparison with more sophisticated classifiers.

.. automodule:: TALENT.model.classical_methods.dummy
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: DummyMethod
   :noindex:

   Dummy Classifier method for classification tasks.
   
   **Key Features:**
   
   - Uses sklearn's DummyClassifier for classification
   - Provides baseline predictions for comparison
   - Supports both binary and multiclass classification
   - Automatically handles data preprocessing including normalization and encoding
   - Saves trained model to pickle file for later use
   - Simple baseline classifier
   
   **Algorithm:**
   
   Dummy Classifier is a classifier that makes predictions using simple rules. It is useful as a baseline for comparison with more sophisticated classifiers. It can use various strategies like most frequent, stratified, uniform, constant, etc.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the Dummy Classifier method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the Dummy Classifier model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for Dummy Classifier
      
      **Model Creation:**
      
      - Creates `DummyClassifier` classifier
      - Configures parameters like strategy, random_state, etc.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the Dummy Classifier model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Fits the Dummy Classifier model
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained Dummy Classifier model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      * **test_logit** (*array-like*) -- Test predictions (class labels for classification)
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained Dummy Classifier model
      3. **Prediction:** Generates baseline predictions using simple rules
      4. **Output:** Returns predicted class labels

**Evaluation Metrics:**

- **For classification:** returns Accuracy, Avg_Precision, Avg_Recall, F1 metrics

**References:**

``[1] scikit-learn developers. (2023). DummyClassifier. scikit-learn documentation.`` 