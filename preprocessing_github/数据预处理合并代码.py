import pandas as pd
from rdkit import Chem
import numpy as np
import warnings
import pathlib as path
from pathlib import Path
from typing import List
from rdkit.Chem.SaltRemover import SaltRemover
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import AllChem

def _InitialiseNeutralisationReactions():
    patts= (
        # Imidazoles
        ('[n+;H]','n'),
        # Amines
        ('[N+;!H0]','N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]','O'),
        # Thiols
        ('[S-;X1]','S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]','N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]','N'),
        # Tetrazoles
        ('[n-]','[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]','S'),
        # Amides
        ('[$([N-]C=O)]','N'),
        )
    return [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]

def NeutraliseCharges(mol, reactions=None):
    if reactions is None:
        _reactions=_InitialiseNeutralisationReactions()
        reactions=_reactions

    replaced = False
    for i,(reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    return mol

class DataProcessing:
    def __init__(self, filepath,smiles_column=None ):
        self.filepath = filepath
        if smiles_column is not None:
            self.data = pd.read_csv(self.filepath)
            self.smiles_column = smiles_column
            self.original_smiles = self.data[self.smiles_column]

    def process_smiles(self, output_filename, use_remove_dot=False,isomericSmiles=True,removeSalt=True,
                       use_smartsTosmi=True, use_function3=False):
        self.data = pd.read_csv(self.filepath)
        processed_smiles = []
        for smi in self.original_smiles:
            new_smi = ""

            if use_smartsTosmi:
                try:
                    remover = SaltRemover()
                    if ':' in smi: #   判断是否是smarts表示  
                        smi = self.smartsTosmi(smi)
                    m = Chem.MolFromSmiles(smi)
                    assert m is not None
                    if removeSalt:
                        assert('.' not in smi)
                    else:
                        m = remover.StripMol(m,dontRemoveEverything=True)  
                    m = NeutraliseCharges(m) #中和电荷
                    new_smi = Chem.MolToSmiles(m,canonical=True,isomericSmiles=isomericSmiles) #得到canonical smiles
                    new_smi = self.remove_dot(new_smi)
                except:
                    print(f"Error: Unable to process SMILES {smi}")
                    new_smi =smi
                    # return None 
                             
            if use_remove_dot:
                new_smi = self.remove_dot(smi)

            if use_function3:
                smi = self.function3(smi)
            processed_smiles.append(new_smi)
        
        # 将处理后的 smiles 添加到数据框中并保存到新的 CSV 文件
        self.data.insert(0, "deIsomericSmiles", processed_smiles)  # 将新增列插入到第0列
        self.data.to_csv(output_filename, index=None)


    def remove_dot(self, smi):
        if '.' in smi:
            smi_ls = smi.split('.')
            if len(smi_ls) != 0:
                smi_ls.sort(key=lambda x: len(x))
                smi = smi_ls[-1]
        return smi

    def smartsTosmi(self,smi):
        try:
            mol = Chem.MolFromSmarts(smi)
            for a in mol.GetAtoms(): a.SetAtomMapNum(0) #去除每个原子的smarts标签
            smi = Chem.MolToSmiles(mol,canonical=True)
            return smi
        except:
            print(f"Error: Unable to process SMILES {smi}")
            return None


    def function3(self, smi):
        # 对 smi 进行处理的代码
        pass
        


    def process_original_file(self, output_filename):
        data_original = pd.read_csv(self.filepath)
        data_original.insert(data_original.columns.get_loc('Quantitative value') + 1, 'min_value', None)
        data_original.insert(data_original.columns.get_loc('Quantitative value') + 2, 'max_value', None)
        # data_original['min_value'] = None
        # data_original['max_value'] = None

        for index, row in data_original.iterrows():
            qualitative_value = row['Qualitative value']
            quantitative_value = row['Quantitative value']
            min_value, max_value = None, None
            if qualitative_value == '=':
                min_value = max_value = quantitative_value
            elif qualitative_value == '<':
                min_value = 0
                max_value = quantitative_value
            elif qualitative_value == '>':
                min_value = quantitative_value
                max_value = data_original['Quantitative value'].max()

            data_original.at[index, 'min_value'] = min_value
            data_original.at[index, 'max_value'] = max_value


        data_original.to_csv(output_filename, index=None)
        
        
    def binning_data(self, bin_min: int, bin_max: int, columns_to_keep: List[str]):
        P = path.WindowsPath(self.filepath)
        df = pd.read_csv(P)
        data1 = df.copy()
        data1["active"] = np.zeros(len(data1))
        for index, row in data1.iterrows():
            if float(row["min_value"]) >= bin_max:
                data1["active"][index] = "low"
            elif float(row["max_value"]) <= bin_min:
                data1["active"][index] = "high"
            else:
                data1["active"][index] = "mid"
        data1.to_csv(P.parent.joinpath(P.stem + "{}".format(bin_min) + "_" + "{}".format(bin_max) + "分箱.csv"), index=False)
        data1 = data1.groupby("SMILES")
        conflict_data = []
        residue_data = []
        for name, gruop in data1:
            l = gruop["active"].tolist()
            s = set(l)
            if len(s) > 1:
                conflict_data.append(gruop)
            if len(s) == 1:
                residue_data.append(gruop)
        df_conflict_data = pd.concat(conflict_data)
        df_conflict_data.to_csv(P.parents[0].joinpath(P.stem + "conflict_data" + "{}".format(bin_min) + "_" + "{}".format(bin_max) + ".csv"), index=False)
        df_residue_data = pd.concat(residue_data)
        df_residue_data.drop_duplicates(subset=["SMILES"], keep="first", inplace=True)
        df_residue_data = df_residue_data.loc[:, columns_to_keep]
        df_residue_data.to_csv(P.parents[0].joinpath(P.stem + "residue_data" + "{}".format(bin_min) + "_" + "{}".format(bin_max) + ".csv"), index=False)

# 使用示例
# 文件路径
if __name__ == '__main__':
    # 第一步：处理原始文件，预处理包括统一单位，这里是把Quantitative value列后新增min_value和max_value两列
    # filepath_1 = Path("Z:/wenku/geixuedi/01数据预处理/过程文件/合并表.csv")
    # output_filepath = filepath_1 .parent / "合并表01.csv"

    # dp1 = DataProcessing(filepath_1 )
    # dp1.process_original_file(output_filepath)

    # 第二步：以原始smiles列为蓝本，新加deIsomericSmiles列
    filepath_2 = Path("Z:/wenku/geixuedi/01数据预处理/过程文件/标准化.csv")
    output_deIsomericSmiles = filepath_2.parent / "deIsomericSmiles.csv"

    dp2 = DataProcessing(filepath_2,"desmiles")
    dp2.process_smiles(output_deIsomericSmiles
                       , use_remove_dot=True,isomericSmiles=True,removeSalt=True,
                       use_smartsTosmi=True)

# remove_dot(smi)：去除 SMILES 字符串中的非离子部分。如果有多个部分，它将保留最长的部分,默认为关，想看看有哪些可以开。
# isomericSmiles=True:默认不去除分子手性
# removeSalt=True：默认去除所有.的分子，即复合形式的分子
# smartsTosmi(smi)：主函数，我没分出来，将 SMILES 字符串从 SMARTS 格式转换为 SMILES 格式，并返回 canonical SMILES。它还去除原子的 SMARTS 标签。
# remove_dot(smi)：去除 SMILES 字符串中的非离子部分。如果有多个部分，它将保留最长的部分。
# _InitialiseNeutralisationReactions()：初始化一组用于中和电荷的反应模式，该模式用于将化合物中的带正负电荷的原子转换为中性原子。
# NeutraliseCharges(mol, reactions=None)：使用 _InitialiseNeutralisationReactions() 中定义的反应模式来中和分子中的电荷。

    # 第三步：分箱

    # bin_min = 100
    # bin_max = 500
    # columns_to_keep = ['SMILES', 'active',"Links"]
    # # 我是设定的bin_max = 100，bin_min = 500，这样基于代码逻辑，负样本能得到更多的保留，具体逻辑理不清看代码，也可以按常规来

    # # 直接使用
    # filepath_3 = r"Z:\wenku\geixuedi\01数据预处理\过程文件\筛选总表02.csv"
    # dp3 = DataProcessing(filepath_3)
    # dp3.binning_data(bin_min, bin_max, columns_to_keep)

    # 更新文件路径以使用新生成的文件
    # filepath_3 = output_deIsomericSmiles
    # dp2.filepath = filepath_3
    # dp2.binning_data(bin_min, bin_max, output_conflict, output_residue_binning)
