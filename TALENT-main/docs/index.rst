.. TALENT documentation master file

========================================================
TALENT: A Tabular Analytics and Learning Toolbox
========================================================

.. image:: ../resources/TALENT-LOGO.png
   :width: 1000px
   :align: center

Welcome to **TALENT**, a comprehensive machine learning toolbox designed to enhance model performance on tabular data. 

TALENT integrates advanced deep learning models, classical algorithms, and efficient hyperparameter tuning, offering robust preprocessing capabilities to optimize learning from tabular datasets. The toolbox is user-friendly and adaptable, catering to both novice and expert data scientists.

.. important::
   If you use any content of this repo for your work, please make sure to cite the relevant papers as described in the `Citing TALENT` section below.

==========================
Citing TALENT
==========================

If you use **TALENT** in your research, please consider citing the following works:

.. code-block:: bibtex

    @article{ye2024closerlookdeeplearning,
             title={A Closer Look at Deep Learning on Tabular Data}, 
             author={Han-Jia Ye and Si-Yang Liu and Hao-Run Cai and Qi-Le Zhou and De-Chuan Zhan},
             journal={arXiv preprint arXiv:2407.00956},
             year={2024}
    }

    @article{liu2024talenttabularanalyticslearning,
             title={TALENT: A Tabular Analytics and Learning Toolbox}, 
             author={Si-Yang Liu and Hao-Run Cai and Qi-Le Zhou and Han-Jia Ye},
             journal={arXiv preprint arXiv:2407.04057},
             year={2024}
    }

==========================
What's New
==========================

Here are the recent updates to **TALENT**:

- [2025-06]🌟 Add `TabAutoPNPNet <https://www.mdpi.com/2079-9292/14/6/1165>`_ (Electronics 2025)

- [2025-06]🌟 Add `TabICL <https://arxiv.org/abs/2502.05564>`_ (ICML 2025). The current code is based on TabICL v0.1.2.

- [2025-05]🌟 Check out our three papers `MMTU <https://github.com/LAMDA-Tabular/MMTU>`_, `Tabular-Temporal-Shift <https://github.com/LAMDA-Tabular/Tabular-Temporal-Shift>`_, and `BETA <https://github.com/LAMDA-Tabular/BETA>`_ accepted at ICML 2025!

- [2025-04]🌟 Check out our new survey `Representation Learning for Tabular Data: A Comprehensive Survey <https://arxiv.org/abs/2504.16109>`_ (`Repo <https://github.com/LAMDA-Tabular/Tabular-Survey>`_). We organize existing methods into three main categories according to their generalization capabilities: specialized, transferable, and general models, which provides a comprehensive taxonomy for deep tabular representation methods.🚀🚀🚀

- [2025-02]🌟 Add `T2Gformer <https://arxiv.org/abs/2211.16887>`_ (AAAI 2023).

- [2025-02]🌟 Add `TabPFN v2 <https://doi.org/10.1038/s41586-024-08328-6>`_ (Nature).

- [2025-02]🌟 Thanks to `Hengzhe Zhang <https://hengzhe-zhang.github.io/>`_ for providing a `Scikit-Learn compatible wrapper <https://github.com/hengzhe-zhang/scikit-TALENT>`_ for TALENT!

- [2025-01]🌟 Check out our new baseline `ModernNCA <https://openreview.net/pdf?id=JytL2MrlLT>`_ (**ICLR 2025**), inspired by traditional **Neighbor Component Analysis**, which outperforms both tree-based and other deep tabular models, while also reducing training time and model size!🚀🚀🚀

- [2025-01]🌟 Check out our `latest version of the benchmark paper <https://arxiv.org/abs/2407.00956>`_ for updated and expanded results and analysis!

- [2025-01]🌟We have curated and released `new benchmark datasets <https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z?usp=drive_link>`_, along with updated `results <https://6sy666.github.io/TALENT-Results/>`_ of the dataset across a broader range of methods. This update focuses on enhancing dataset quality, including removing duplicates, and correcting tasks where bin-class was mistakenly treated as regression. We have also separated the larger datasets and formed the basic benchmark (300 datasets, including 120 bin-class, 80 multi-class, and 100 regression), and the large benchmark (22 datasets).

- [2024-12]🌟 Add `TabM <https://arxiv.org/abs/2410.24210>`_ (ICLR 2025).

- [2024-09]🌟 Add `Trompt <https://arxiv.org/abs/2305.18446>`_ (ICML 2023).

- [2024-09]🌟 Add `AMFormer <https://arxiv.org/abs/2402.02334>`_ (AAAI 2024).

- [2024-08]🌟 Add `GRANDE <https://arxiv.org/abs/2309.17130>`_ (ICLR 2024).

- [2024-08]🌟 Add `Excelformer <https://arxiv.org/abs/2301.02819>`_ (KDD 2024).

- [2024-08]🌟 Add `MLP_PLR <https://arxiv.org/abs/2203.05556>`_ (NeurIPS 2022).

- [2024-07]🌟 Add `RealMLP <https://arxiv.org/abs/2407.04491>`_.

- [2024-07]🌟 Add `ProtoGate <https://arxiv.org/abs/2306.12330>`_ (ICML 2024).

- [2024-07]🌟 Add `BiSHop <https://arxiv.org/abs/2404.03830>`_ (ICML 2024).

- [2024-06]🌟 Check out our new baseline `ModernNCA <https://arxiv.org/abs/2407.03257>`_, inspired by traditional **Neighbor Component Analysis**, which outperforms both tree-based and other deep tabular models, while also reducing training time and model size!

- [2024-06]🌟 Check out our `benchmark paper <https://arxiv.org/abs/2407.00956>`_ about tabular data, which provides comprehensive evaluations of classical and deep tabular methods based on our toolbox in a fair manner!

.. note::
   If you want to view **benchmark results**, please visit:
   `https://6sy666.github.io/TALENT-Results/ <https://6sy666.github.io/TALENT-Results/>`_

   To explore **default hyperparameters and search spaces** of methods in this toolbox, check:
   `https://6sy666.github.io/TALENT-Configs/ <https://6sy666.github.io/TALENT-Configs/>`_

==========================
Contents
==========================

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Methods

   methods
   
.. toctree::
   :maxdepth: 2
   :caption: Dependencies
   
   dependencies

.. toctree::
   :maxdepth: 2
   :caption: Benchmark_Datasets   

   benchmark_datasets

.. toctree::
   :maxdepth: 2
   :caption: Experimental_Results

   experimental_results

.. toctree::
   :maxdepth: 1
   :caption: API Docs

   api/core
   api/classical_methods
   api/deep_learning
   api/lib

.. toctree::
   :maxdepth: 2
   :caption: Acknowledgements

   acknowledgements

