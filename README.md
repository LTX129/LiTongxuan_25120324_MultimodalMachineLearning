# 本地 AI 智能文献与图像管理助手（Local Multimodal AI Agent）

一个**纯本地运行**的多模态知识库助手，用于管理 PDF 文献与图片素材。  
相比“按文件名搜索”，本项目通过**语义向量检索**实现对论文内容与图片语义的快速定位，并支持自动归档与索引维护。
- 姓名： 李同轩
- 学号： 25120324
---

## 功能概览

### 1) 智能文献管理（PDF）
- **语义搜索（paper-level 聚合）**：输入自然语言问题，返回最相关的论文，并在每篇论文下展示多个相关片段（避免单篇论文占满候选）。
- **自动分类整理**
  - **单文件**：添加论文时按 `--topics` 自动归类到 `library/<topic>/`
  - **批量**：对文件夹内 PDF 递归整理并建立索引
- **索引维护**
  - **stats**：查看当前索引状态（论文 chunk 数、图片数、目录与数据库路径）
  - **rebuild_index**：从 `library/` 与 `datasets/images/` 全量重建索引，保证一致性
  - **remove_paper**：从索引中移除指定论文

### 2) 智能图像管理（Images）
- **以文搜图（text-to-image）**：输入文本描述（如 “sunset by the sea”）检索本地图片库，返回最匹配的图片路径。

---

## 项目结构
> 我用的俩模型太大了，传不上来，huggingface上下载下来放在文件夹下（我的做法）或者缓存都行，在config.py中修改模型路径即可（含 json 的那个文件夹）。

```text
.
├── datasets/
│   ├── papers/          # 原始 PDF 输入
│   └── images/          # 原始图片输入
├── library/             # 归档后的论文目录（按 topic 分文件夹）
├── storage/
│   ├── chroma_papers/   # 论文向量库（ChromaDB）
│   └── chroma_images/   # 图片向量库（ChromaDB）
├── main.py              # CLI 统一入口
├── config.py            # 路径与超参配置
└── ...
```

---

## 环境与依赖

- **Miniforge Python 3.10**
- **本地模型（仓库内本地加载, 更换电脑在config.py需要中更改模型地址）**：
  - **文本嵌入**：all-MiniLM-L6-v2
  - **图文匹配**：CLIP ViT-B-32
---

## 安装依赖：

`pip install -r requirements.txt`



---

## 快速开始（命令示例）

- **由于模型能力的限制，使用汉语进行搜索的时候可能会出现驴唇不对马嘴的情况，这不是代码 bug！当前用的是 OpenAI 原版 CLIP ViT-B/32，主要在英文文本上训练，中文描述的语义对齐较弱，所以例如检索图像时使用“树”，但“campus tree”被排到最后一位。代码里检索用的是标准的归一化 CLIP 特征，逻辑是正确的。强烈建议使用英文进行搜索，以保证项目的可复现！！！**

- **所有功能通过 main.py 调用：**


### 查看帮助
`python main.py --help`

### 1) 添加并按主题分类单篇论文（复制到 library/<topic>/ 并建立索引）
`python main.py add_paper datasets/papers/BERT.pdf --topics "CV,NLP,RL"`

### 2) 批量整理/索引整个文件夹（递归处理 PDF）
`python main.py organize datasets/papers --topics "CV,NLP,RL"`

### 3) 语义搜索论文（索引为空时会自动从 library/ 建索引）
`python main.py search_paper "Use cases of Transformer." --top_k 7`

### 4) 以文搜图（索引为空时会自动索引 datasets/images）
`python main.py search_image "sunset" --top_k 3`

### 5) 查看索引状态（论文 chunk 数、图片数、路径等）
`python main.py stats`

### 6) 全量重建索引（当更换模型、chunk_size、或移动/删除论文后建议执行）
`python main.py rebuild_index`


---



## 关键设计说明（Why it works）

### 1) PDF 分块与检索单元

- 使用 `pypdf` 提取文本  
- 按 `config.PDF_CHUNK_SIZE` 分块（chunk）  
- 每个 chunk 建立向量并写入 ChromaDB  
- 检索时以 chunk 为基础召回候选，再做 paper-level 聚合输出（一篇论文最多展示 N 个片段）

> 这样做既能命中具体内容，又能避免“同一篇论文占满 top_k”的问题。

### 2) 文本与图片向量

- **文本**：SentenceTransformers（`all-MiniLM-L6-v2`），归一化向量  
- **图片**：CLIP image encoder；检索时用 CLIP text encoder 生成查询向量，与图片向量对齐

### 3) 索引一致性（`stats` / `rebuild_index`）

- `stats` 提供可观察性，便于调试与演示  
- `rebuild_index` 用于模型升级、参数变更、或库内容变动后的全量重建，避免“新旧 embedding 混用”导致检索异常

---

## 运行结果（节选）

###  1) 批量整理与分类

```bash
> python main.py organize datasets/papers --topics "CV,NLP,RL"
2025-12-17 16:38:45,056 [INFO] Loading text model from /Users/tonglion/PycharmProjects/Experiment2/all-MiniLM-L6-v2
2025-12-17 16:38:45,057 [INFO] Load pretrained SentenceTransformer: /Users/tonglion/PycharmProjects/Experiment2/all-MiniLM-L6-v2
2025-12-17 16:38:45,143 [INFO] Loading CLIP model from /Users/tonglion/PycharmProjects/Experiment2/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.'
2025-12-17 16:38:45,773 [INFO] Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
2025-12-17 16:38:45,896 [INFO] Connected to Chroma collection=papers at /Users/tonglion/PycharmProjects/Experiment2/storage/chroma_papers
2025-12-17 16:38:45,897 [INFO] Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
2025-12-17 16:38:45,905 [INFO] Connected to Chroma collection=images at /Users/tonglion/PycharmProjects/Experiment2/storage/chroma_images
Organizing papers:   0%|                                                                                                                    | 0/10 [00:00<?, ?it/s]2025-12-17 16:38:45,923 [INFO] Copied into library /Users/tonglion/PycharmProjects/Experiment2/datasets/papers/π  3  - Permutation-Equivariant Visual Geometry Learning.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/π  3  - Permutation-Equivariant Visual Geometry Learning.pdf
2025-12-17 16:38:47,157 [INFO] Classified /Users/tonglion/PycharmProjects/Experiment2/library/π  3  - Permutation-Equivariant Visual Geometry Learning.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/CV/π  3  - Permutation-Equivariant Visual Geometry Learning.pdf (score=0.342)
Organizing papers:  10%|██████████▊                                                                                                 | 1/10 [00:01<00:15,  1.71s/it]2025-12-17 16:38:47,624 [INFO] Copied into library /Users/tonglion/PycharmProjects/Experiment2/datasets/papers/Proximal Policy Optimization Algorithms.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/Proximal Policy Optimization Algorithms.pdf
2025-12-17 16:38:51,122 [INFO] Classified /Users/tonglion/PycharmProjects/Experiment2/library/Proximal Policy Optimization Algorithms.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/RL/Proximal Policy Optimization Algorithms.pdf (score=0.353)
Organizing papers:  20%|█████████████████████▌                                                                                      | 2/10 [00:05<00:23,  2.90s/it]2025-12-17 16:38:51,361 [INFO] Copied into library /Users/tonglion/PycharmProjects/Experiment2/datasets/papers/Soft Adaptive Policy Optimization.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/Soft Adaptive Policy Optimization.pdf
2025-12-17 16:38:52,069 [INFO] Classified /Users/tonglion/PycharmProjects/Experiment2/library/Soft Adaptive Policy Optimization.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/RL/Soft Adaptive Policy Optimization.pdf (score=0.317)
Organizing papers:  30%|████████████████████████████████▍                                                                           | 3/10 [00:06<00:14,  2.01s/it]2025-12-17 16:38:52,319 [INFO] Copied into library /Users/tonglion/PycharmProjects/Experiment2/datasets/papers/Gemini- A Family of Highly Capable  Multimodal Models.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/Gemini- A Family of Highly Capable  Multimodal Models.pdf
2025-12-17 16:38:54,391 [INFO] Classified /Users/tonglion/PycharmProjects/Experiment2/library/Gemini- A Family of Highly Capable  Multimodal Models.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/CV/Gemini- A Family of Highly Capable  Multimodal Models.pdf (score=0.358)
Organizing papers:  40%|███████████████████████████████████████████▏                                                                | 4/10 [00:10<00:15,  2.66s/it]2025-12-17 16:38:55,978 [INFO] Copied into library /Users/tonglion/PycharmProjects/Experiment2/datasets/papers/Mamba- Linear-Time Sequence Modeling with Selective State Spaces.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/Mamba- Linear-Time Sequence Modeling with Selective State Spaces.pdf
2025-12-17 16:38:57,097 [INFO] Classified /Users/tonglion/PycharmProjects/Experiment2/library/Mamba- Linear-Time Sequence Modeling with Selective State Spaces.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/NLP/Mamba- Linear-Time Sequence Modeling with Selective State Spaces.pdf (score=0.222)
Organizing papers:  50%|██████████████████████████████████████████████████████                                                      | 5/10 [00:12<00:12,  2.48s/it]2025-12-17 16:38:58,136 [INFO] Copied into library /Users/tonglion/PycharmProjects/Experiment2/datasets/papers/BERT.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/BERT.pdf
2025-12-17 16:38:58,914 [INFO] Classified /Users/tonglion/PycharmProjects/Experiment2/library/BERT.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/NLP/BERT.pdf (score=0.471)
Organizing papers:  60%|████████████████████████████████████████████████████████████████▊                                           | 6/10 [00:13<00:08,  2.08s/it]2025-12-17 16:38:59,430 [INFO] Copied into library /Users/tonglion/PycharmProjects/Experiment2/datasets/papers/DeepSeekMath- Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/DeepSeekMath- Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf
2025-12-17 16:39:00,037 [INFO] Classified /Users/tonglion/PycharmProjects/Experiment2/library/DeepSeekMath- Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/NLP/DeepSeekMath- Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf (score=0.348)
Organizing papers:  70%|███████████████████████████████████████████████████████████████████████████▌                                | 7/10 [00:14<00:05,  1.80s/it]2025-12-17 16:39:00,659 [INFO] Copied into library /Users/tonglion/PycharmProjects/Experiment2/datasets/papers/GPT-4o System Card.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/GPT-4o System Card.pdf
2025-12-17 16:39:01,186 [INFO] Classified /Users/tonglion/PycharmProjects/Experiment2/library/GPT-4o System Card.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/CV/GPT-4o System Card.pdf (score=0.263)
Organizing papers:  80%|██████████████████████████████████████████████████████████████████████████████████████▍                     | 8/10 [00:15<00:03,  1.58s/it]2025-12-17 16:39:01,767 [INFO] Copied into library /Users/tonglion/PycharmProjects/Experiment2/datasets/papers/VGGT- Visual Geometry Grounded Transformer.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/VGGT- Visual Geometry Grounded Transformer.pdf
2025-12-17 16:39:02,591 [INFO] Classified /Users/tonglion/PycharmProjects/Experiment2/library/VGGT- Visual Geometry Grounded Transformer.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/CV/VGGT- Visual Geometry Grounded Transformer.pdf (score=0.357)
Organizing papers:  90%|█████████████████████████████████████████████████████████████████████████████████████████████████▏          | 9/10 [00:17<00:01,  1.57s/it]2025-12-17 16:39:03,310 [INFO] Copied into library /Users/tonglion/PycharmProjects/Experiment2/datasets/papers/Attention Is All You Need.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/Attention Is All You Need.pdf
2025-12-17 16:39:04,013 [INFO] Classified /Users/tonglion/PycharmProjects/Experiment2/library/Attention Is All You Need.pdf -> /Users/tonglion/PycharmProjects/Experiment2/library/NLP/Attention Is All You Need.pdf (score=0.288)
Organizing papers: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:18<00:00,  1.84s/it]
2025-12-17 16:39:04,354 [INFO] Organized 10 papers

```
###  2) 语义搜索（paper-level 聚合结果）

```bash
> python main.py search_paper "Use cases of Transformer." --top_k 7
[1] /Users/tonglion/PycharmProjects/Experiment2/library/NLP/Mamba- Linear-Time Sequence Modeling with Selective State Spaces.pdf (topic=NLP, score=-0.369)
  (1) [chunk 43] score=-0.369: Donald Metzler. “Long Range Arena: A Benchmark for Efficient Transformers”. In: International Conference on Learning Representations (ICLR) . 2021. [104] Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. “Efficie...
  (2) [chunk 29] score=-0.427: model, Mamba can achieve 5×higher throughput than Transformers. 4.6 Model Ablations We perform a series of detailed ablations on components of our model, focusing on the setting of language modeling with size ≈350M model...
------------------------------------------------------------
[2] /Users/tonglion/PycharmProjects/Experiment2/library/NLP/Attention Is All You Need.pdf (topic=NLP, score=-0.377)
  (1) [chunk 11] score=-0.377: 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost. Model BLEU Training Cost (FLOPs) ...
  (2) [chunk 6] score=-0.528: h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full d...
------------------------------------------------------------
[3] /Users/tonglion/PycharmProjects/Experiment2/library/NLP/BERT.pdf (topic=NLP, score=-0.436)
  (1) [chunk 5] score=-0.436: and ﬁne-tuning. Dur- ing pre-training, the model is trained on unlabeled data over different pre-training tasks. For ﬁne- tuning, the BERT model is ﬁrst initialized with the pre-trained parameters, and all of the param- ...
  (2) [chunk 0] score=-0.671: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova Google AI Language {jacobdevlin,mingweichang,kentonl,kristout}@google.com Abstrac...
------------------------------------------------------------
[4] /Users/tonglion/PycharmProjects/Experiment2/library/CV/VGGT- Visual Geometry Grounded Transformer.pdf (topic=CV, score=-0.509)
  (1) [chunk 46] score=-0.509: . Vasilakos, and Thippa Reddy Gadekallu. Generative pre-trained trans- former: A comprehensive review on enabling technologies, potential applications, emerging challenges, and future di- rections. arXiv.cs, abs/2305.104...
  (2) [chunk 35] score=-0.553: Vision . Cambridge University Press, ISBN: 0521540518, 2004. 13 [47] Xingyi He, Jiaming Sun, Yifan Wang, Sida Peng, Qixing Huang, Hujun Bao, and Xiaowei Zhou. Detector-free struc- ture from motion. In arxiv, 2023. 12 [48...
------------------------------------------------------------
[5] /Users/tonglion/PycharmProjects/Experiment2/library/CV/Gemini- A Family of Highly Capable  Multimodal Models.pdf (topic=CV, score=-0.653)
  (1) [chunk 80] score=-0.653: of Highly Capable Multimodal Models 9. Contributions and Acknowledgments Gemini Leads Rohan Anil,Co-Lead, Text Sebastian Borgeaud,Co-Lead, Text Jean-Baptiste Alayrac,Co-Lead, MM Vision Jiahui Yu,Co-Lead, MM Vision Radu S...
  (2) [chunk 58] score=-0.705: feedback across safety and other domain areas through the user interface, and where possible, in-depth interviews. Focus areas included safety and persona, functionality, coding and instruction capabilities, and factuali...
------------------------------------------------------------
[6] /Users/tonglion/PycharmProjects/Experiment2/library/CV/π  3  - Permutation-Equivariant Visual Geometry Learning.pdf (topic=CV, score=-0.716)
  (1) [chunk 26] score=-0.716: this design can introduce noticeable grid-like artifacts, particularly in regions with high reconstruction uncertainty. 14...
  (2) [chunk 23] score=-0.757: whereas VGGT uses 48. The decoders for camera poses, local point maps, and confidence scores share the same architecture but do not share weights. This architecture is a lightweight, 5-layer transformer that applies self...
------------------------------------------------------------
[7] /Users/tonglion/PycharmProjects/Experiment2/library/CV/GPT-4o System Card.pdf (topic=CV, score=-0.765)
  (1) [chunk 28] score=-0.765: Zhong, Mia Glaese, Nick Turley, Noah Deutsch, Noel Bundick, Ola Okelola, Olivier Godement, Owen Campbell-Moore, Peter Bak, Peter Bakkum, Raul Puri, Rowan Zellers, Saachi Jain, Shantanu Jain, Shirong Wu, Spencer Papay, Ta...
------------------------------------------------------------

> python main.py search_paper "visual" --top_k 7
[1] /Users/tonglion/PycharmProjects/Experiment2/library/CV/VGGT- Visual Geometry Grounded Transformer.pdf (topic=CV, score=-0.135)
  (1) [chunk 0] score=-0.135: VGGT: Visual Geometry Grounded Transformer Jianyuan Wang1,2 Minghao Chen1,2 Nikita Karaev1,2 Andrea Vedaldi1,2 Christian Rupprecht1 David Novotny2 1Visual Geometry Group, University of Oxford 2Meta AI … Figure 1. VGGT is...
  (2) [chunk 19] score=-0.185: Input Images Ground Truth Prediction Figure 6. Qualitative Examples of Novel View Synthesis. The top row shows the input images, the middle row displays the ground truth images from target viewpoints, and the bottom row ...
------------------------------------------------------------
[2] /Users/tonglion/PycharmProjects/Experiment2/library/CV/Gemini- A Family of Highly Capable  Multimodal Models.pdf (topic=CV, score=-0.150)
  (1) [chunk 70] score=-0.150: Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai- Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts.arXiv preprint ar...
  (2) [chunk 98] score=-0.196: This section shows sample qualitative examples from prompting the Gemini Ultra model. Some illustrative examples of multimodal reasoning for image understanding tasks over charts, natural images and memes are shown in Fi...
------------------------------------------------------------
[3] /Users/tonglion/PycharmProjects/Experiment2/library/CV/GPT-4o System Card.pdf (topic=CV, score=-0.195)
  (1) [chunk 1] score=-0.195: text and vision capabilities of GPT-4o, depending on the risk assessed. This is indicated accordingly throughout the System Card. 1 arXiv:2410.21276v1 [cs.CL] 25 Oct 2024 • Proprietary data from data partnerships.We form...
  (2) [chunk 27] score=-0.290: Cunninghman, Thomas Dimson, Thomas Raoux, Tianhao Zheng, Christina Kim, Todd Underwood, Tristan Heywood, Valerie Qi, Vinnie Monaco, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wojciech Zaremba, Yash Patil, Yilei, Qian, Yongji...
------------------------------------------------------------
[4] /Users/tonglion/PycharmProjects/Experiment2/library/CV/π  3  - Permutation-Equivariant Visual Geometry Learning.pdf (topic=CV, score=-0.223)
  (1) [chunk 0] score=-0.223: π3: Permutation-Equivariant Visual Geometry Learning Yifan Wang1∗ Jianjun Zhou123∗ Haoyi Zhu1 Wenzheng Chang1 Yang Zhou1 Zizun Li1 Junyi Chen1 Jiangmiao Pang1 Chunhua Shen2 Tong He13† 1Shanghai AI Lab 2ZJU 3SII ∗Equal Co...
  (2) [chunk 21] score=-0.294: Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer.arXiv preprint arXiv:2503.11651, 2025. [35] Kaixuan Wang and Shaojie Shen. Fl...
------------------------------------------------------------
[5] /Users/tonglion/PycharmProjects/Experiment2/library/NLP/Attention Is All You Need.pdf (topic=NLP, score=-0.391)
  (1) [chunk 4] score=-0.391: of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent pos...
------------------------------------------------------------
[6] /Users/tonglion/PycharmProjects/Experiment2/library/NLP/BERT.pdf (topic=NLP, score=-0.406)
  (1) [chunk 25] score=-0.406: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144. Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. 2014. How transferable are features in deep neural networks? In Advances...
------------------------------------------------------------

> python main.py search_paper "Use cases of reinforcement learning." --top_k 7
[1] /Users/tonglion/PycharmProjects/Experiment2/library/NLP/DeepSeekMath- Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf (topic=NLP, score=-0.121)
  (1) [chunk 29] score=-0.121: techniques (Kwon et al., 2023; Leviathan et al., 2023; Xia et al., 2023, 2024), which determines 21 the exploration efficiency of policy models, also play an exceedingly important role. Algorithms Algorithms process the ...
  (2) [chunk 19] score=-0.123: of each reasoning step. Formally, given the question 𝑞and 𝐺 sampled outputs {𝑜1, 𝑜2, ··· , 𝑜𝐺}, a process reward model is used to score each step of the outputs, yielding corresponding rewards: R = {{𝑟𝑖𝑛𝑑𝑒𝑥(1) 1 , ··· , ...
------------------------------------------------------------
[2] /Users/tonglion/PycharmProjects/Experiment2/library/RL/Proximal Policy Optimization Algorithms.pdf (topic=RL, score=-0.167)
  (1) [chunk 11] score=-0.167: J. Schulman, J. Tang, and W. Zaremba. “OpenAI Gym”. In: arXiv preprint arXiv:1606.01540 (2016). [Dua+16] Y. Duan, X. Chen, R. Houthooft, J. Schulman, and P. Abbeel. “Benchmarking Deep Reinforcement Learning for Continuou...
  (2) [chunk 0] score=-0.217: Proximal Policy Optimization Algorithms John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov OpenAI {joschu, filip, prafulla, alec, oleg}@openai.com Abstract We propose a new family of policy gradien...
------------------------------------------------------------
[3] /Users/tonglion/PycharmProjects/Experiment2/library/NLP/Mamba- Linear-Time Sequence Modeling with Selective State Spaces.pdf (topic=NLP, score=-0.242)
  (1) [chunk 16] score=-0.242: (and general LTI models). On the other hand, selective models can simply reset their state at any time to remove extraneous history, and thus their performance in principle improves monotonicly with context length (e.g. ...
  (2) [chunk 37] score=-0.270: Models with Generalized Basis Projections”. In: The International Conference on Learning Representations (ICLR) . 2023. [42] Ankit Gupta, Albert Gu, and Jonathan Berant. “Diagonal State Spaces are as Effective as Structu...
------------------------------------------------------------
[4] /Users/tonglion/PycharmProjects/Experiment2/library/CV/Gemini- A Family of Highly Capable  Multimodal Models.pdf (topic=CV, score=-0.333)
  (1) [chunk 33] score=-0.333: measures human preference in domains such as travel planning and video discovery. We find models equipped with tools are preferred on this set 78% of the time over models without tools (excluding ties). Gemini API models...
  (2) [chunk 30] score=-0.336: to our models provides further gains over SFT alone. Our approach creates an iterative process in which RL continually pushes the boundaries of the RM, while the RM is continuously improved through evaluation and data co...
------------------------------------------------------------
[5] /Users/tonglion/PycharmProjects/Experiment2/library/RL/Soft Adaptive Policy Optimization.pdf (topic=RL, score=-0.371)
  (1) [chunk 6] score=-0.371: ) bAi,t ∂zv = ∂πθ(yi,t |q,y i,<t ) ∂zv · bAi,t πθ(yi,t |q,y i,<t ) = 1(v=y i,t )exp(z yi,t ) ∑v′∈V exp(zv′ )−exp(z yi,t )exp(z v) (∑v′∈V exp(zv′ ))2 · bAi,t πθ(yi,t |q,y i,<t ) = ( 1−π θ(yi,t |q,y i,<t )  · bAi,t ifv=y...
  (2) [chunk 14] score=-0.394: on Artificial Intelligence, volume 37, pages 7078–7086, 2023. DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.arXiv preprint arXiv:2501.12948, 2025. HMMT. Hmmt 2025.https:/...
------------------------------------------------------------
[6] /Users/tonglion/PycharmProjects/Experiment2/library/CV/GPT-4o System Card.pdf (topic=CV, score=-0.402)
  (1) [chunk 37] score=-0.402: “Paradigm: Improving patient access to clinical trials.”https://openai.com/index/paradigm/, 2024. Accessed: 2024-08-07. [49] M. Hutson, “How ai is being used to accelerate clinical trials,”Nature, vol. 627, pp. S2–S5, 20...
  (2) [chunk 18] score=-0.433: scheming. Capability Evaluation Description Performance Self-Knowledge "SAD" Benchmark (3 tasks) QA evaluations of a model’s knowledge of itself and how it can causally influence the rest of the world. ••◦ Explicit Theor...
------------------------------------------------------------
```
###  3) 索引状态与重建

```bash
> python main.py stats
Papers indexed chunks: 418
Images indexed: 0
Library dir: /Users/tonglion/PycharmProjects/Experiment2/library
Paper DB: /Users/tonglion/PycharmProjects/Experiment2/storage/chroma_papers
Image DB: /Users/tonglion/PycharmProjects/Experiment2/storage/chroma_images

> python main.py rebuild_index
2025-12-17 16:43:39,413 [INFO] Loading text model from /Users/tonglion/PycharmProjects/Experiment2/all-MiniLM-L6-v2
2025-12-17 16:43:39,414 [INFO] Load pretrained SentenceTransformer: /Users/tonglion/PycharmProjects/Experiment2/all-MiniLM-L6-v2
2025-12-17 16:43:39,498 [INFO] Loading CLIP model from /Users/tonglion/PycharmProjects/Experiment2/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.'
2025-12-17 16:43:40,088 [INFO] Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
2025-12-17 16:43:40,173 [INFO] Connected to Chroma collection=papers at /Users/tonglion/PycharmProjects/Experiment2/storage/chroma_papers
2025-12-17 16:43:40,174 [INFO] Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
2025-12-17 16:43:40,177 [INFO] Connected to Chroma collection=images at /Users/tonglion/PycharmProjects/Experiment2/storage/chroma_images
2025-12-17 16:43:40,177 [INFO] Rebuilding paper index from library /Users/tonglion/PycharmProjects/Experiment2/library
Indexing papers: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:16<00:00,  1.69s/it]
2025-12-17 16:43:57,172 [INFO] Rebuilding image index from /Users/tonglion/PycharmProjects/Experiment2/datasets/images
2025-12-17 16:43:57,465 [INFO] Indexed 4 images from /Users/tonglion/PycharmProjects/Experiment2/datasets/images
2025-12-17 16:43:57,465 [INFO] Done.

> python main.py stats
2025-12-17 16:44:07,745 [INFO] Loading text model from /Users/tonglion/PycharmProjects/Experiment2/all-MiniLM-L6-v2
2025-12-17 16:44:07,746 [INFO] Load pretrained SentenceTransformer: /Users/tonglion/PycharmProjects/Experiment2/all-MiniLM-L6-v2
2025-12-17 16:44:07,795 [INFO] Loading CLIP model from /Users/tonglion/PycharmProjects/Experiment2/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.'
2025-12-17 16:44:08,356 [INFO] Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
2025-12-17 16:44:08,466 [INFO] Connected to Chroma collection=papers at /Users/tonglion/PycharmProjects/Experiment2/storage/chroma_papers
2025-12-17 16:44:08,466 [INFO] Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
2025-12-17 16:44:08,469 [INFO] Connected to Chroma collection=images at /Users/tonglion/PycharmProjects/Experiment2/storage/chroma_images
Papers indexed chunks: 418
Images indexed: 4
Library dir: /Users/tonglion/PycharmProjects/Experiment2/library
Paper DB: /Users/tonglion/PycharmProjects/Experiment2/storage/chroma_papers
Image DB: /Users/tonglion/PycharmProjects/Experiment2/storage/chroma_images
```
###  4) 以文搜图

```bash
>  python main.py search_image "sunset" --top_k 3
[1] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/sunset by the sea.png (sunset by the sea.png)
[2] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/village field.png (village field.png)
[3] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/campus tree.png (campus tree.png)

>  python main.py search_image "perple" --top_k 3
[1] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/campus tree.png (campus tree.png)
[2] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/village field.png (village field.png)
[3] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/sunset by the sea.png (sunset by the sea.png)
  
                                                                                                                                                      
> python main.py search_image "people" --top_k 3
[1] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/street walker.png (street walker.png)
[2] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/sunset by the sea.png (sunset by the sea.png)
[3] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/village field.png (village field.png)

> python main.py search_image "ground" --top_k 3
[1] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/campus tree.png (campus tree.png)
[2] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/village field.png (village field.png)
[3] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/sunset by the sea.png (sunset by the sea.png)

> python main.py search_image "plants" --top_k 3
[1] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/campus tree.png (campus tree.png)
[2] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/village field.png (village field.png)
[3] /Users/tonglion/PycharmProjects/Experiment2/datasets/images/sunset by the sea.png (sunset by the sea.png)
```
###  5) 文献分类效果示例（CV，NLP，RL）
- 分类效果偏向于论文语言描述，比如 deepseek 的 DeepSeekMath- Pushing the Limits of Mathematical Reasoning in Open Language Models，我觉得它属于 RL 和 NLP 都行，模型把它分到了 NLP。
  
<img width="97" height="100" alt="截屏2025-12-17 下午7 09 26" src="https://github.com/user-attachments/assets/18b7c4b2-5bb0-40b6-a01d-c2a12a441fca" />
<img width="514" height="312" alt="截屏2025-12-17 下午7 13 29" src="https://github.com/user-attachments/assets/aa21fbad-8a73-47df-9ac7-92d984c1e083" />

## 可选优化方向（后续可扩展）
- 参考文献/致谢噪声过滤：减少引用段落对检索的干扰
- 返回页码/段落定位：在元数据中存储 page/chunk → 页码映射
- 支持更多主题与多标签分类：不仅选 1 个 topic，可输出 top-2/top-3
- GUI/REST API：FastAPI + 简易前端，提升演示效果

