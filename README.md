## FG-LAED是什么?
FG-LAED (Feature Generator based on Local Atomic Environment Descriptor) is a Python-based toolkit designed to calculate atomic center symmetric function values by reading CIF structure files and automatically extracting inherent properties of coordinating elements. This toolkit facilitates customized feature generation for various types of catalysts, enabling automated catalyst design across a wide range of materials while effectively reducing costs.
## Requirements
```
ase (atomic simulation environment)
openpyxl (library for reading and writing Excel files)
shutil (library for file operations)
numpy (library for numerical computing)
pandas (library for data manipulation and analysis)
re (regular expressions library)
sys (system-specific parameters and functions)
utils.cif2input (custom utility for converting CIF files)
utils.ACSF (custom utility for Atomic Cluster Symmetry Function)
```

## Example
使用`input`中的数据进行特征提取。如下示例所示：
```python
    form run import run
    configpath = r'template/config1.data'
    inputpath = r'input/'
    data_2d, data_2d_path = DV_LAE(functionpath, reffunctionpath=None, intervelnum=10, mode='tsne', inputpath=inputpath,
                                   savename=None, refnum=1, num=-1, distancemode=0, interval=interval)

```
	
    After the execution is complete, an out folder will be automatically created within the temp directory. This out folder contains feature table files, and the Sheet3 in the table corresponds to the extracted features for each catalyst. 
	
## 参数介绍

```
inputpath：输入文件目录
configpath：对称函数配置文件目录
reffunctionpath：参考结构对称函数目录，默认为 None，使用 functionpath
intervelnum：使用直方图统计的区间个数
mode：降维方式，默认为 tsne，可选 pca、tsne
inputpath：势函数训练结构源文件
distancemode：不同的直方图统计模式，默认为 0，可选 0、1、2。选择模式 0 时，统计两个直方图区间有不同则将差异向量取 1，反之取 0；选择模式 1 时，统计两个直方图区间使用二者距离作为差异向量；选择模式 2 时，统计两个直方图区间有不同则将差异向量则加 1，反之加 0
savename：样本多样性可视化保存文件名，默认为 None，保存在 functionpath 下，命名格式为 [日期]_[源文件名]_[intervelnum]_[mode]_[distancemode].html
interval：使用降维后分布进行筛选时，取的网格大小，默认为 0.05
max_points_per_grid：使用降维后分布进行筛选时，每个网格最多保留样本的数量，默认为 1
output：自定义精简后数据文件名，默认为 None，保存在 functionpath 下，命名格式为 output_[日期]_[源文件名]_[interval]_[max_points_per_grid].data
```
