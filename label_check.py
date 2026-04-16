
#cell_label:['Enterocyte.Progenitor', 'Stem', 'TA.Early', 'TA', 'Tuft', 'Enterocyte', 'Goblet', 'Endocrine']
#condition:['Control', 'Hpoly.Day10']





import scanpy as sc

# 读取 h5ad 文件
adata = sc.read_h5ad("/home/wuboyang/scduo-new/dataset/medicine_perturb_data/train_hpoly.h5ad")


# 查看 obs 的列名
print("obs 列名：", adata.obs.columns.tolist())
print("n_cells:", adata.n_obs, "n_genes:", adata.n_vars)

# 查看前几行
print("\nobs 前几行：")
print(adata.obs.head())

print(adata.obs["cell_label"].unique())

print(adata.obs["condition"].unique())

for cond in adata.obs["cell_type"].unique():
    adata_subset = adata[adata.obs["cell_type"] == cond]
    print(f"{cond}: {adata_subset.n_obs} cells")

# print(adata.obs["perturbation"].unique())
# print(adata.obs["perturbation"].value_counts())

# print(adata.obs["celltype"].unique())
# print(adata.obs["celltype"].value_counts())

# print(adata.X.shape)
# print(adata.layers.keys())

# print(adata.obs["pathway"].unique())
# print(adata.obs["pathway"].value_counts())

# print(adata.obs["cp_type"].unique())
# print(adata.obs["cp_type"].value_counts())

# # 打印扰动类型中样本数量大于200的扰动类型
# vc = adata.obs["perturbation"].value_counts()
# large = vc[vc > 200]
# print("\nPerturbations with >200 cells:")
# for pert, cnt in large.sort_values(ascending=False).items():
#     print(f"{pert}: {cnt}")




# # 在原始计数矩阵 X 上统计
# sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# adata.obs["n_genes"]  = adata.obs["n_genes_by_counts"].astype(int)
# adata.obs["n_counts"] = adata.obs["total_counts"].astype(int)



# print(adata.obs["perturbation"].unique())
# print(adata.obs["perturbation"].value_counts())

# # 计数（每个 perturbation 有多少细胞）
# print(adata.obs["cell_type"].unique())
# print(adata.obs["cell_type"].value_counts())


# print(adata.obs["dose_val"].unique())
# print(adata.obs["dose_val"].value_counts())


# print(adata.obs["control"].unique())
# print(adata.obs["control"].value_counts())


# # print("细胞类型：")
# # print(adata.obs["celltype"].value_counts())

# # # 查看主要扰动类型有哪些，以及数量
# print("\n批次类型：")
# print(adata.obs["batch"].unique)
# print(adata.obs["batch"].value_counts())









# import numpy as np

# # 跳过第一行，只读向量（去掉基因名）
# embeddings = np.loadtxt(
#     "/home/wuboyang/scduo-new/dataset/pre_trained_emb/gene2vec_dim_200_iter_9.txt",
#     skiprows=1, usecols=range(1, 201)   # 从第2列到第201列（共200维）
# )
# print("embedding shape:", embeddings.shape)
# # 输出前5个基因的嵌入向量
# for i in range(5):
#     print(f"Gene {i+1} embedding:", embeddings[i])





# import scanpy as sc
# import numpy as np

# # 读取数据
# adata = sc.read_h5ad("/home/wuboyang/scduo-new/dataset/processed_data/train_pbmc.h5ad")

# # 1) 检查 raw
# print("adata.raw 是否存在:", adata.raw is not None)

# # 2) 每个细胞的总表达量
# cell_sums = np.array(adata.X.sum(axis=1)).flatten()
# print("每个细胞表达量总和统计:")
# print("均值:", cell_sums.mean())
# print("中位数:", np.median(cell_sums))
# print("范围:", cell_sums.min(), "~", cell_sums.max())

# # 3) 检查数值范围
# vals = adata.X.A.flatten() if hasattr(adata.X, "A") else np.array(adata.X).flatten()
# print("数值范围:", vals.min(), "~", vals.max())





# import numpy as np

# path = "/home/wuboyang/scduo-new/gene2vec/pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt"

# with open(path, "r") as f:
#     # 读第一行：基因数 和 维度
#     num_genes, dim = map(int, f.readline().strip().split())
    
#     genes = []
#     embeddings = np.zeros((num_genes, dim), dtype=np.float32)

#     for i, line in enumerate(f):
#         parts = line.strip().split()
#         gene = parts[0]
#         vec = np.array(parts[1:], dtype=np.float32)
#         genes.append(gene)
#         embeddings[i] = vec

# print("总基因数:", len(genes))
# print("向量维度:", embeddings.shape[1])
# print("前5个基因:", genes[:5])
# print("示例向量:", embeddings[0][:10])






# import anndata as ad
# import pandas as pd

# path = "/home/wuboyang/scduo-new/dataset/processed_data/train_covid.h5ad"
# adata = ad.read_h5ad(path)

# print(f"=== File: {path}")
# print("X shape (n_cells, n_genes):", adata.shape)

# # 简便函数：兼容分类/字符串两种类型，返回有序列表
# def uniques(series):
#     if pd.api.types.is_categorical_dtype(series):
#         return list(series.cat.categories)
#     vals = pd.Series(series.dropna().unique())
#     try:
#         vals = vals.sort_values()
#     except Exception:
#         pass
#     return list(vals.astype(str))

# # 确认需要的列存在
# required_cols = ["condition", "celltype", "batch"]
# print("\nobs 列名：", list(adata.obs.columns))
# missing = [c for c in required_cols if c not in adata.obs.columns]
# if missing:
#     raise KeyError(f"缺少必要的 obs 列: {missing}")

# # 列出取值
# print("\n[Unique values]")
# print("celltype:", uniques(adata.obs["celltype"]))
# print("batch   :", uniques(adata.obs["batch"]))
# print("condition:", uniques(adata.obs["condition"]))

# # 统计各 condition 数量
# print("\n[condition value_counts]")
# print(adata.obs["condition"].value_counts(dropna=False))

# # 统计特定标签
# targets = ["control", "severe COVID-19"]
# print("\n[Counts for targets]")
# for t in targets:
#     cnt = (adata.obs["condition"] == t).sum()
#     print(f"{t}: {cnt}")





# import scanpy as sc
# import anndata as ad
# import pandas as pd

# path = "/home/wuboyang/scduo-new/dataset/processed_data/train_covid.h5ad"
# adata = ad.read_h5ad(path)

# # 确认有 celltype 列
# if "celltype" not in adata.obs.columns:
#     raise KeyError("obs 里没有 'celltype' 列，请检查列名或先生成该列。")

# # 若没有 UMAP，则计算（用最基础流程；你也可以换成 bbknn 流程）
# if "X_umap" not in adata.obsm:
#     print("未检测到 UMAP，开始计算 PCA→neighbors→UMAP ...")
#     # 可按需先挑 HVGs、标准化等；这里直接简单示例
#     if "X_pca" not in adata.obsm:
#         sc.pp.pca(adata, n_comps=50, svd_solver="arpack")
#     sc.pp.neighbors(adata, use_rep="X_pca" if "X_pca" in adata.obsm else "X")
#     sc.tl.umap(adata, min_dist=0.4, spread=1.0, random_state=0)

# # 绘制并保存（Scanpy 默认保存到工作目录下 figures/）
# sc.pl.umap(
#     adata,
#     color="celltype",
#     title="UMAP colored by celltype",
#     legend_loc="on data",   # 或 "right margin"
#     legend_fontsize=6,
#     legend_fontoutline=1,
#     s=8,                    # 点大小
#     save="_celltype.png",
#     show=False,
# )





# import scanpy as sc
# import numpy as np
# import pandas as pd

# # ========= 配置 =========
# in_path        = "/home/wuboyang/scduo-new/dataset/hpoly_merged.h5ad"   # 输入文件
# out_path       = "/home/wuboyang/scduo-new/dataset/processed_data/train_hpoly.h5ad"  # 输出文件
# condition_key  = "condition"
# control_value  = "Control"
# pert_value     = "Hpoly.Day10"
# celltype_key   = "cell_label"           # 用于对齐的类别
# random_state   = 42                    # 保证可复现
# # ========================

# # 1. 读取
# adata = sc.read_h5ad(in_path)

# # 2. 分组
# ctrl  = adata[adata.obs[condition_key] == control_value].copy()
# pert  = adata[adata.obs[condition_key] == pert_value   ].copy()

# # 3. 统计每个 cell_type 的数量
# counts_ctrl = ctrl.obs[celltype_key].value_counts()
# counts_pert = pert.obs[celltype_key].value_counts()

# # 4. 找到共同 cell_type，并对齐数量
# common_types = np.intersect1d(counts_ctrl.index, counts_pert.index)

# ctrl_list, pert_list = [], []

# rng = np.random.default_rng(random_state)
# for ct in common_types:
#     n_ctrl = counts_ctrl[ct]
#     n_pert = counts_pert[ct]
#     n_keep = min(n_ctrl, n_pert)            # 取较小值
    
#     # 在各自组内随机抽取 n_keep
#     idx_ctrl_ct = ctrl.obs[celltype_key] == ct
#     idx_pert_ct = pert.obs[celltype_key] == ct
    
#     choose_ctrl = rng.choice(np.where(idx_ctrl_ct)[0], size=n_keep, replace=False)
#     choose_pert = rng.choice(np.where(idx_pert_ct)[0], size=n_keep, replace=False)
    
#     ctrl_list.append(ctrl[choose_ctrl])
#     pert_list.append(pert[choose_pert])

# # 5. 按要求拼接：先全部 control，再全部 perturbed
# ctrl_ordered = ctrl_list[0].concatenate(*ctrl_list[1:], join="outer", batch_key=None)
# pert_ordered = pert_list[0].concatenate(*pert_list[1:], join="outer", batch_key=None)
# ordered      = ctrl_ordered.concatenate(pert_ordered, join="outer", batch_key=None)

# # 6. 保存
# ordered.write_h5ad(out_path)
# print(f"✔ 保存完成：{out_path}")
# print(f"最终形状：cells={ordered.n_obs}, genes/features={ordered.n_vars}")








# import scanpy as sc
# import sys

# out_path       = "/home/wuboyang/scduo-new/dataset/processed_data/train_hpoly.h5ad"  # 输出文件
# condition_key  = "condition"
# control_value  = "Control"
# pert_value     = "Hpoly.Day10"
# celltype_key   = "cell_label"

# #-------------------------------------------------
# print("🔹 读取 AnnData ...")
# adata = sc.read_h5ad(out_path)
# n = adata.n_obs
# if n % 2 != 0:
#     sys.exit(f"❌ 细胞总数 {n} 不是偶数，无法一一对应！")

# half = n // 2
# cond_first  = adata.obs[condition_key].iloc[:half].unique()
# cond_second = adata.obs[condition_key].iloc[half:].unique()

# # 1️⃣ 检查分段条件
# if set(cond_first)  != {control_value}:
#     sys.exit(f"❌ 前半段包含非 {control_value} 条件: {cond_first}")
# if set(cond_second) != {pert_value}:
#     sys.exit(f"❌ 后半段包含非 {pert_value} 条件: {cond_second}")

# print("✔ 顺序检查通过：前半 all control，后半 all perturbed")

# # 2️⃣ 并行逐行对齐检查
# print("🔹 开始并行细胞类型比对 ...")
# mismatches = []
# for i in range(half):
#     ct_ctrl = adata.obs[celltype_key].iloc[i]
#     ct_pert = adata.obs[celltype_key].iloc[i + half]
#     if ct_ctrl != ct_pert:
#         mismatches.append((i, ct_ctrl, ct_pert))
#         # 也可以选择立即退出：sys.exit(...)

# if not mismatches:
#     print(f"✅ 逐行细胞类型完全一致！（{half} 对）")
# else:
#     print(f"❌ 发现 {len(mismatches)} 处不匹配，示例：")
#     for idx, c1, c2 in mismatches[:10]:
#         print(f"  行 {idx}   control:{c1}   perturbed:{c2}")
#     # 若要强制退出：
#     # sys.exit("请检查采样/拼接逻辑！")
