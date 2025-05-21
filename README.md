# Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge

Official PyTorch Implementation of **SCINet**, from the following paper:

> Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge. IEEE Transactions on Multimedia 2025. (Under Review)


**Abstract**

Partial multi-label learning aims to extract knowledge from incompletely annotated data, which includes known correct labels, known incorrect labels, and unknown labels. The core challenge lies in accurately identifying the ambiguous relationships between labels and instances. In this paper, we emphasize that matching co-occurrence patterns between labels and instances is key to addressing this challenge. To this end, we propose Semantic Co-occurrence Insight Network (SCINet), a novel and effective framework for partial multi-label learning. Specifically, SCINet introduces a bi-dominant prompter module, which leverages an off-the-shelf multimodal model to capture text-image correlations and enhance semantic alignment. To reinforce instance-label interdependencies, we develop a cross-modality fusion module that jointly models inter-label correlations, inter-instance relationships, and co-occurrence patterns across instance-label assignments. Moreover, we propose an intrinsic semantic augmentation strategy that enhances the model’s understanding of intrinsic data semantics by applying diverse image transformations, thereby fostering a synergistic relationship between label confidence and sample difficulty. Extensive experiments on four widely-used benchmark datasets demonstrate that SCINet surpasses state-of-the-art methods.

<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./Figures/Model.png" align="center" ></td>
  </tr>
</table>
</p>



## Performance

