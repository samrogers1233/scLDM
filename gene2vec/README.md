# gene2vec

Place pretrained gene2vec embeddings in this directory.

## Expected file

`gene2vec_dim_200_iter_9_w2v.txt` (word2vec text format):

```
<num_genes> <dim>
GENE_SYMBOL_1 v1 v2 ... vD
GENE_SYMBOL_2 v1 v2 ... vD
...
```

- First line: two integers — number of genes and embedding dimension.
- Each subsequent line: gene symbol followed by `dim` float values, space-separated.
- Loaded by `scLDM.perturbation.diffusion.gene_perturbation_datasets.load_gene2vec_txt_simple`.

## How it's used

Passed via the `--gene2vec_path` CLI flag to
[gene_sample.py](../script/training_diffusion/py_scripts/gene_sample.py) and as the
`gene2vec_path` kwarg to the gene-perturbation dataset loader. Training with
`--use_gene_cond True` also expects this file to exist.

## Source

Pretrained gene2vec embeddings are available from
[jingcheng-du/Gene2vec](https://github.com/jingcheng-du/Gene2vec).
Drop the `gene2vec_dim_200_iter_9_w2v.txt` file directly into this folder.
