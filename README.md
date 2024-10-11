**Installation**

To use carla_visual as a package, from the folder CARLA_VISUAL run:
```
pip install -r requirements_carla_visual.txt
pip install -e setup.py
```


**Additional data**

For faster launch of the implemented PIECE and CounteRGAN, some weights and 
latent encodings were moved to Google Drive and can be found by the following link:
https://drive.google.com/drive/folders/16ufl5il7iHglWuqolf_GJXGjVLTgQGgl?usp=sharing


**Description**

To run experiments with the implemented models, navigate 
to the `experiments` folder under the data modality of interest,
either `images` or `tabular`.

The notebook `run_countergan_debug.ipynb` contains the implementation 
of CounteRGAN adapting subclassing of the tensorflow.keras.Model. 
The training works but has not been run on GPU yet.

The notebook `run_countergan.ipynb` contains the loading of the previously trained 
CounteRGAN with the original implementation.    
