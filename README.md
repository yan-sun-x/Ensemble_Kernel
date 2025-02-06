# Unsupervised Multiple Kernel Learning for Graphs via Ordinality Preservation

> Published as a conference [paper](https://openreview.net/pdf?id=6nb2J90XJD) at ICLR 2025.

## Dependencies

- Python 3.8+
- PyTorch 2.2+
- Numpy 1.24+

The folder structure is:
```
|-- .Ensemble_Kernel
    |-- README.md
    |-- cache
    |-- experiment
    |-- models
    |-- utils
    |-- main.py

```

## Training and Evaluation

If you want to run the model `UMKL-G` with the loss function `power KL divergence` in power 2, run the command as below.

```{bash}
python main.py --model UMKL-G --dataset MUTAG --loss_fun PKL --power 2
```

## Results

Check the folder `experiemnt`, under which results on each dataset are display with 

- `kernel_list` configuration in `para.yml`
- For each set of parameters, the numbered folder (e.g. 1, 2, ...) contains comparison results of `UMKL-G` and baselines `UMKL`, `sparse-UMKL`.
  - Evaluation metrics include `Accuracy`, `Normalized Mutual Information`, and `Adjusted Rand Index` after clustering on the learned kernel matrix.
  - Losses and weights are recorded as well.



## Citation
```
@inproceedings{sun2023mmd,
  title={MMD Graph Kernel: Effective Metric Learning for Graphs via Maximum Mean Discrepancy},
  author={Sun, Yan and Fan, Jicong},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```