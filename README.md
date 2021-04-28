*Updated. Repository under construction.*

This repository includes the implementation of a method proposed in **"Tailoring automated data augmentation to H&E-stained histopathology"** accepted to MIDL 2021.

**Abstract**

Convolutional neural networks (CNN) are sensitive to domain shifts, which can result inpoor generalization.  In medical imaging, data acquisition conditions differ among institu-tions, which leads to variations in image properties and thus domain shift.  Stain variationin histopathological slides is a prominent example.  Data augmentation is one way to makeCNNs robust to varying forms of domain shift but requires extensive hyper-parameter tun-ing.   Due  to  the  large  search  space,  this  is  cumbersome  and  often  leads  to  sub-optimalgeneralization  performance.   In  this  work,  we  focus  on  automated  and  computationallyefficient data augmentation policy selection for histopathological slides.  Building upon theRandAugment framework, we introduce several domain-specific modifications relevant tohistopathological images, increasing generalizability.  We test these modifications on H&E-stained histopathology slides from Camelyon17 dataset. Our proposed framework outper-forms the state-of-the-art manually engineered data augmentation strategy, achieving anarea under the ROC curve of 0.964 compared to 0.958, respectively.

<div align="center">
    <img src="/he-randaugment/augmentations_new.png" width="900px"</img> 
</div>
**Methodology**
**Code instructions**
**Notes**
In case of any questions, please contact: khrystyna.faryna@gmail.com
