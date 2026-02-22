# Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge

Official PyTorch Implementation of **SCINet**, from the following paper:

> Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge. IEEE Transactions on Multimedia 2026.


**Model**

<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./Figures/Model.png" align="center" ></td>
  </tr>
</table>
</p>

## 📦 Installation & Data Preparation

### Installation
```bash
pip install -r requirement.txt
```

The `requirements.txt` file can be used to install the necessary packages into a virtual environment.

**Other:**
```bash
conda env create -n SCINet -f environment.yml
conda activate SCINet
```

### Data Preparation

Donwload [data](https://www.autodl.com/console/public-data?page_size=10&page_index=1), and unzip.

## 🚀 Train and Test Examples

### train 
```bash
python train.py --data voc2007
```

### test
```bash
python train.py --data voc2007 -t -r 1
```

## 📎 Citation

If you find this project useful in your research, please consider citing our paper:
```
@article{WU2026,  
  title = {Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge},
  author = {Wu, Xin and Teng, Fei and Feng, Yue and Shi, Kaibo and Lin, Zhuosheng and Zhang, Ji and Wang, James},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  publisher={IEEE}
}
```

## 💘 Acknowledgements
Thank you for making your open-source code in this field available.

## 📬 Contact
In case of any questions, bugs, suggestions or improvements, please feel free to drop me at xinwu5386@gmail.com or open an issue.
