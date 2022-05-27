# UV Volumes for Real-time Rendering of Editable Free-view Human Performance
### [Project Page](https://fanegg.github.io/UV-Volumes) | [Paper](https://arxiv.org/pdf/2203.14402)
<br/>

> UV Volumes for Real-time Rendering of Editable Free-view Human Performance  
> [Yue Chen](https://github.com/fanegg/), [Xuan Wang](https://scholar.google.com/citations?user=h-3xd3EAAAAJ&hl=en), [Xingyu Chen](https://github.com/rover-xingyu/), [Qi Zhang](https://qzhang-cv.github.io/), [Xiaoyu Li](https://xiaoyu258.github.io/), [Yu Guo](https://gr.xjtu.edu.cn/web/yu.guo/home), [Jue Wang](https://juewang725.github.io/), [Fei Wang](http://www.aiar.xjtu.edu.cn/info/1046/1242.htm)  
> arXiv preprint arXiv:2203.14402

![demo_vid](assets/dynamic_human.mp4)

![Teaser image](assets/teaser.jpg)

Abstract: *Neural volume rendering enables photo-realistic renderings of a human performer in free-view, a critical task in immersive VR/AR applications. But the practice is severely limited by high computational costs in the rendering process. To solve this problem, we propose the UV Volumes, a new approach that can render an editable free-view video of a human performer in realtime. It separates the high-frequency (i.e., non-smooth) human appearance from the 3D volume, and encodes them into 2D neural texture stacks (NTS). The smooth UV volumes allow much smaller and shallower neural networks to obtain densities and texture coordinates in 3D while capturing detailed appearance in 2D NTS. For editability, the mapping between the parameterized human model and the smooth texture coordinates allows us a better generalization on novel poses and shapes. Furthermore, the use of NTS enables interesting applications, e.g., retexturing. Extensive experiments on CMU Panoptic, ZJU Mocap, and H36M datasets show that our model can render 960 × 540 images in 30FPS on average with comparable photo-realism to state-of-the-art methods.*

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

