# UV Volumes for Real-time Rendering of Editable Free-view Human Performance<br><sub>Official PyTorch implementation</sub>

![Teaser image](./teaser.png)

**UV Volumes for Real-time Rendering of Editable Free-view Human Performance**<br>
Yue Chen, Xuan Wang, Qi Zhang, Xiaoyu Li, Xingyu Chen, Yu Guo, Jue Wang and Fei Wang
<br>
#### [Project Page](https://fanegg.github.io/UV-Volumes/)<br>

Abstract: *Neural volume rendering has been proven to be a promising method for efficient and photo-realistic rendering of a human performer in free-view, a critical task in many immersive VR/AR applications. However, existing approaches are severely limited by their high computational cost in the rendering process. To solve this problem, we propose the UV Volumes, an approach that can render an editable free-view video of a human performer in real-time. It is achieved by removing the high-frequency (i.e., non-smooth) human textures from the 3D volume and encoding them into a 2D neural texture stack (NTS). The smooth UV volume allows us to employ a much smaller and shallower structure for 3D CNN and MLP, to obtain the density and texture coordinates without losing image details. Meanwhile, the NTS only needs to be queried once for each pixel in the UV image to retrieve its RGB value. For editability, the 3D CNN and MLP decoder can easily fit the function that maps the input structured-and-posed latent codes to the relatively smooth densities and texture coordinates. It gives our model a better generalization ability to handle novel poses and shapes. Furthermore, the use of NST enables new applications, e.g., retexturing. Extensive experiments on CMU Panoptic, ZJU Mocap, and H36M datasets show that our model can render 900 × 500 images in 40 fps on average with comparable photorealism to state-of-the-art methods.*

## Citation

```
@article{chen2022uvvolumes,
  title={UV Volumes for Real-time Rendering of Editable Free-view Human Performance},
  author={Chen, Yue and Wang, Xuan and Zhang, Qi and Li, Xiaoyu and Chen, Xingyu and Guo, Yu and Wang, Jue and Wang, Fei},
  journal={arXiv preprint arXiv:2203.14402},
  year={2022}
}
```
