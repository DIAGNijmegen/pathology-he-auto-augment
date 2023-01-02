*Updated*

This repository includes the implementation of a method proposed in **"Tailoring automated data augmentation to H&E-stained histopathology"** accepted to MIDL 2021.

**Abstract**

Convolutional neural networks (CNN) are sensitive to domain shifts, which can result in poor generalization.  In medical imaging, data acquisition conditions differ among institutions, which leads to variations in image properties and thus domain shift.  Stain variation in histopathological slides is a prominent example.  Data augmentation is one way to make CNNs robust to varying forms of domain shift but requires extensive hyperparameter tuning.   Due  to  the  large  search  space,  this  is  cumbersome  and  often  leads  to  suboptimal generalization  performance.   In  this  work,  we  focus  on  automated  and  computationally efficient data augmentation policy selection for histopathological slides.  Building upon the RandAugment framework, we introduce several domain-specific modifications relevant to histopathological images, increasing generalizability.  We test these modifications on H&E-stained histopathology slides from Camelyon17 dataset. Our proposed framework outper-forms the state-of-the-art manually engineered data augmentation strategy, achieving an area under the ROC curve of 0.964 compared to 0.958, respectively.

<div align="center">
    <img src="/he-randaugment/augmentations_new.png" width="900px"</img> 
</div>

**Methodology**

Check the [paper](https://2021.midl.io/proceedings/faryna21.pdf) for methodology.


**Code instructions**

To run the code:
```
launch_experiment.py dataset_dir output_dir experiment_tag trial_tag n_epochs batch_size lr randaugment rand_m rand_n ra_type v1_type v2_type t1_type t2_type
```
To use the modified version of randaugment standalone you can apply the distort_image_with_randaugment function to each image in the training batch, providing following arguments magnitude (m), number of sequentially applied transforms (n or num_layers), type of a transform set (ra_type= 'Default') for transforms used in this paper. 
```
distort_image_with_randaugment(image, m, n, ra_type)
``` 

**Practical matters & clarifications**


We explored following ranges of magnitudes, while tuning the RandAugment to H&E stained histopathology, (also Table 1 in the paper):

| transform type | magnitude range | transform type | magnitude range |
| ------------- | ------------- | ------------- | ------------- |
|‘identity’ | - | ‘shear x’ | [-0.9, 0.9] |
|‘contrast’ | [0.0, 5.5] | ‘shear y’ | [-0.9, 0.9] |
|‘brightness’ | [0.0, 5.5] | ‘HED shift’ | [-0.9, 0.9] |
|‘sharpness’ | [0.0, 5.5] | ‘HSV shift’ | [-0.9, 0.9] |
|‘rotation’ | [-90.0, 90.0] | ‘autocontrast’ | - |
|‘translate x’ | [-30.0, 30.0] | ‘color’ | [0.0, 5.5] |
|‘translate y’ | [-30.0, 30.0] | ‘equalize’ | - |

Discretized into 30 points as suggested in RandAugment code. 
For the **final** search of optimal magnitude we used the following ranges:

| transform type | magnitude range | transform type | magnitude range |
| ------------- | ------------- | ------------- | ------------- |
|‘identity’ | - | ‘shear x’ | [-0.45, 0.45] |
|‘contrast’ | [0.28, 2.8] | ‘shear y’ | [-0.45, 0.45] |
|‘brightness’ | [0.28, 2.8] | ‘HED shift’ | [-0.45, 0.45] |
|‘sharpness’ | [0.28, 2.8] | ‘HSV shift’ | [-0.45, 0.45] |
|‘rotation’ | [-45.0, 45.0] | ‘autocontrast’ | - |
|‘translate x’ | [-15.0, 15.0] | ‘color’ | [0.28, 2.8] |
|‘translate y’ | [-15.0, 15.0] | ‘equalize’ | - |

You can obtain these ranges py plugging values of magnitudes m from 1 to 15 into *distort_image_with_randaugment* function. We noticed that ranges with values of m above 15 can result in strong distortion of the histopatholological image for some transforms (i.e. *brightness* with m>15 results in structures in the image dissapearing), in final search we only used m={1:15}. For Camelyon17 we selected m=5, n=3.

*Bear in mind that these ranges only make sense if you use same transform functions as used in this code (hsv and hed from our library and PIL for the rest).*


**Notes**

In case of any questions, please contact: khrystyna.faryna@gmail.com 


**Publication**


If you find this work useful, cite as:

```
@InProceedings{faryna21,
  title = 	 {Tailoring automated data augmentation to H&amp;E-stained histopathology},
  author =       {Faryna, Khrystyna and van der Laak, Jeroen and Litjens, Geert},
  booktitle = 	 {Proceedings of the Fourth Conference on Medical Imaging with Deep Learning},
  pages = 	 {168--178},
  year = 	 {2021},
  editor = 	 {Heinrich, Mattias and Dou, Qi and de Bruijne, Marleen and Lellmann, Jan and Schläfer, Alexander and Ernst, Floris},
  volume = 	 {143},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {07--09 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v143/faryna21a/faryna21a.pdf},
  url = 	 {https://proceedings.mlr.press/v143/faryna21a.html},

}
```
