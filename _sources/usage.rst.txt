Set up
======

Installation
--------------
To use PENGUIN, first clone the GitHUb repo.

To clone this repository to your local machine, use the following command:

.. code-block:: bash

    git clone https://github.com/your_username/PENGUIN.git

Environment set up
-------------------

You can create the environment installing the packages or using the ymal file.

To manually create and install packages use:

.. code-block:: bash

    conda create --name penguin
    conda activate penguin
    conda install matplotlib pandas panel numpy opencv scikit-image ipywidgets jupyter ipykernel plotly
    pip install apeer-ometiff-library --no-deps

Alternatively, you can create the environemtn using the yml file:

.. code-block:: bash

    conda env create --file penguin_env.yml
    conda activate penguin
    pip install apeer-ometiff-library --no-deps


After you created the environment, and if you want to use the Jupyter notebooks

add the environment kernel to jupyter

.. code-block:: bash

    python -m ipykernel install --user --name=penguin

launch the jupyter and be sure that you are running with the penguin kernel.

