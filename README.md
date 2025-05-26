# Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge

Official PyTorch Implementation of **SCINet**, from the following paper:

> Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge. IEEE Transactions on Multimedia 2025. (Under Review)


**Model**

<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./Figures/Model.png" align="center" ></td>
  </tr>
</table>
</p>


## 📦 Installation & Data Preparation


## 🚀 Train and Test Examples
`torch==1.11.0+cu113 torchvision==0.12.0+cu113`
conda env create -n SCINet -f environment.yml

conda activate SCINet

other:
```bash
pip install timm ftfy regex pycocotool
```

### train
```bash
python train.py --data voc2007
```

### test
```bash
python train.py --data voc2007 -t -r 1
```

## 💘 Acknowledgements


## 📎 Citation

If you find this project useful in your research, please consider citing our paper:
```
@article{WU2025,  
title = {Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge},  
year = {2025},  
author = {Wu, Xin and Teng, Fei and Feng, Yue and Shi, Kaibo and Lin, Zhuosheng and Zhang, Ji and Wang, James}
}
```

## 📬 Contact
In case of any questions, bugs, suggestions or improvements, please feel free to drop me at xinwu5386@gmail.com or open an issue.
