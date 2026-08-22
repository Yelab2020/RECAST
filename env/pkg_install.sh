#conda create -n RECAST python==3.8.20
#conda activate RECAST
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install pyg -c pyg -y
conda install -c dglteam/label/th22_cu118 dgl -y
conda install pydantic -y
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rdkit
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple dgllife
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy==1.10.1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.24.3
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scanpy==1.9.8
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple imbalanced-learn==0.12.4
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple esda==2.5.1
conda install pytorch-cluster -c pyg
