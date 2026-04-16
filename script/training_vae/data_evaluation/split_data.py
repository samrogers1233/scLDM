import scanpy as sc
import anndata as ad

# 1. 假设你已经有两个 AnnData 对象
# adata_ctrl: 控制组数据
# adata_pert: 扰动组数据


dataset_path = '/home/wuboyang/scduo-main/dataset/processed_data/train_pbmc.h5ad'
adata_all = sc.read(dataset_path)

# 2. 分离 CD4T 和 非 CD4T
adata_cd4t = adata_all[adata_all.obs['cell_type'] == 'NK'].copy()
adata_wo_cd4t = adata_all[adata_all.obs['cell_type'] != 'NK'].copy()

# 3. 保存为 H5AD 文件
adata_cd4t.write('/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_NK.h5ad')
adata_wo_cd4t.write('/home/wuboyang/scduo-main/dataset/processed_data/ood_test/pbmc_oodNK.h5ad')
print(adata_all.shape,adata_cd4t.shape,adata_wo_cd4t.shape)
print("✅ 已成功将 NK 和非 NK 分离并保存！")