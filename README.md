# LQ-Adapter


## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@inproceedings{inproceedings,
author = {Madan, Chetan and Gupta, Mayuna and Basu, Soumen and Gupta, Pankaj and Arora, Chetan},
year = {2025},
month = {02},
pages = {557-567},
title = {LQ-Adapter: ViT-Adapter with Learnable Queries for Gallbladder Cancer Detection from Ultrasound Images},
doi = {10.1109/WACV61041.2025.00064}
}
```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE.md) file.




sh dist_test.sh configs/htc++/htc++_beit_adapter_large_fpn_3x_coco.py /path/to/checkpoint_file 8 --eval bbox segm
