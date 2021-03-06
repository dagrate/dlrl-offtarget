# dlrl-offtarget

Deep Learning and Reinforcement Learning for off-target predictions <br>

Data for Experiments: <br>
- *data/* -> zip data to reproduce the experiments

Saved Models: <br>
- *saved_model_4x23/* -> saved deep learning models for the predictions with 4x23 encoding. RF has a fixed random seed for the reproducibility of the results.
- *saved_model_crispr_8x23/* -> saved deep learning models for the predictions with 8x23 encoding on CRISPOR data set. RF has a fixed random seed for the reproducibility of the results.
- *saved_model_guideseq_8x23/* -> saved deep learning models for the predictions with 8x23 encoding on GUIDE-seq data set with transfer learning. RF has a fixed random seed for the reproducibility of the results.

Notebooks: <br>
- *notebooks/* -> notebook to used to run the experiments with "4x23" encoding and "8x23" encoding

Python Files: <br>
- *src/cnns.py* -> cnns implementation
- *src/fnns.py* -> ffns implementation
- *src/mltrees.py* -> random forest implementation
- *src/utilities.py* -> set of python functions to preprocess the data and postprocess the results of the experiments

## Reference Papers

```bibtex
@article{peng2018recognition,
  title={Recognition of CRISPR/Cas9 off-target sites through ensemble learning of uneven mismatch distributions},
  author={Peng, Hui and Zheng, Yi and Zhao, Zhixun and Liu, Tao and Li, Jinyan},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i757--i765},
  year={2018},
  publisher={Oxford University Press}
}
```

```bibtex
@article{lin2018off,
  title={Off-target predictions in CRISPR-Cas9 gene editing using deep learning},
  author={Lin, Jiecong and Wong, Ka-Chun},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i656--i663},
  year={2018},
  publisher={Oxford University Press}
}
```
