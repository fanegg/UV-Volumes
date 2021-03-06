# UV Volumes for Real-time Rendering of Editable Free-view Human Performance
### [Project Page](https://fanegg.github.io/UV-Volumes) | [Paper](https://arxiv.org/pdf/2203.14402) | [Supplementary](https://fanegg.github.io/UV-Volumes/files/UV_Volumes_Supplementary_Material.pdf)
<br/>

> UV Volumes for Real-time Rendering of Editable Free-view Human Performance  
> [Yue Chen](https://fanegg.github.io/), [Xuan Wang](https://xuanwangvc.github.io/), [Xingyu Chen](http://rover-xingyu.github.io/), [Qi Zhang](https://qzhang-cv.github.io/), [Xiaoyu Li](https://xiaoyu258.github.io/), [Yu Guo](https://yuguo-xjtu.github.io/), [Jue Wang](https://juewang725.github.io/), [Fei Wang](http://www.aiar.xjtu.edu.cn/info/1046/1242.htm)  
> arXiv preprint arXiv:2203.14402

![Teaser image](assets/teaser.jpg)

Abstract: *Neural volume rendering enables photo-realistic renderings of a human performer in free-view, a critical task in immersive VR/AR applications. But the practice is severely limited by high computational costs in the rendering process. To solve this problem, we propose the UV Volumes, a new approach that can render an editable free-view video of a human performer in realtime. It separates the high-frequency (i.e., non-smooth) human appearance from the 3D volume, and encodes them into 2D neural texture stacks (NTS). The smooth UV volumes allow much smaller and shallower neural networks to obtain densities and texture coordinates in 3D while capturing detailed appearance in 2D NTS. For editability, the mapping between the parameterized human model and the smooth texture coordinates allows us a better generalization on novel poses and shapes. Furthermore, the use of NTS enables interesting applications, e.g., retexturing. Extensive experiments on CMU Panoptic, ZJU Mocap, and H36M datasets show that our model can render 960 × 540 images in 30FPS on average with comparable photo-realism to state-of-the-art methods.*

<!-- ![demo_vid](assets/demo.gif) -->

[![Editable Free-view Human Performance](https://res.cloudinary.com/marcomontalbano/image/upload/v1655406367/video_to_markdown/images/youtube--5ODxXfB34CM-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=5ODxXfB34CM "Editable Free-view Human Performance")

## More editable free-view human performance
### Free view rendering showcase
![iuv](assets/iuv.jpg)
### We decompose the dynamic human into 3D _UV volumes_ and 2D _Neural Texture Stacks(NTS)_.
![dy360](assets/dy360.jpg)
### Reshaping showcase
![reshape](assets/reshape.jpg)
### Retexturing showcase
![retexture](assets/retexture.jpg)
![retexture2](assets/retexture2.jpg)

## Brewing🍺, code coming soon.
## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{chen2022uvvolumes,
  title={UV Volumes for Real-time Rendering of Editable Free-view Human Performance},
  author={Chen, Yue and Wang, Xuan and Chen, Xingyu and Zhang, Qi and Li, Xiaoyu and Guo, Yu and Wang, Jue and Wang, Fei},
  journal={arXiv preprint arXiv:2203.14402},
  year={2022}
}
```

