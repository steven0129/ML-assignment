# Assignment 1

## Installation

```
pip install -r requirements.txt
```

## 預設參數

預設參數為k=1, distance_type='ssd'

```
python main.py train
```

## 調整參數

```
python main.py train --k=3 --distance-type=sad
```

## 超參數優化(Hyper-parameter optimization)

```
python main.py hyperparameter
```

## 結論

根據2012年的論文Random search for hyper-parameter optimization，針對超參數取樣230次，各個參數，包含k值與distance_type等參數對於準確率的影響如下圖

![](https://i.imgur.com/Z83p9vR.png)

![](https://i.imgur.com/gG69vnu.png)