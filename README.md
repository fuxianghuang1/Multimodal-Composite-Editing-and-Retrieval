# <p align=center> Multimodal Composite Editing and Retrieval </p> # 

:fire::fire: This is a collection of awesome articles about multimodal composite editing and retrieval:fire::fire:

[NEWS.20240909] **The related survey [paper](http://arxiv.org/abs/2409.05405) has been released.**

If you find this repository is useful for you, please cite our paper:
```
@misc{li2024survey,
      title={A Survey of Multimodal Composite Editing and Retrieval}, 
      author={Suyan Li, Fuxiang Huang, and Lei Zhang},
      year={2024},
      eprint={2409.05405},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


- [Papers and related codes](#papers-and-related-codes)
  - [Image-text composite editing](#image-text-composite-editing)
  - [Image-text composite retrieval](#image-text-composite-retrieval)
  - [Other mutimodal composite retrieval](#other-mutimodal-composite-retrieval)
  

- [Datasets](#datasets)
  - [Datasets for image-text composite editing](#datasets-for-image-text-composite-editing)
  - [Datasets for image-text composite retrieval](#datasets-for-image-text-composite-retrieval)
  - [Other mutimodal composite retrieval](#datasets-for-other-mutimodal-composite-retrieval)

- [Experimental Results](#experimental-results)
  
  
# Papers and related codes
## Image-text composite editing

### 2024
**[ACM SIGIR, 2024] Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval**  
*Haokun Wen, Xuemeng Song, Xiaolin Chen, Yinwei Wei, Liqiang Nie, Tat-Seng Chua*  
[[Paper](http://dx.doi.org/10.1145/3626772.3657727)]

**[IEEE TIP, 2024] Multimodal Composition Example Mining for Composed Query Image Retrieval**  
*Gangjian Zhang, Shikun Li, Shikui Wei, Shiming Ge, Na Cai, Yao Zhao*  
[[Paper](http://dx.doi.org/10.1109/TIP.2024.3359062)]

**[IEEE TMM, 2024] Align and Retrieve: Composition and Decomposition Learning in Image Retrieval With Text Feedback**  
*Yahui Xu, Yi Bin, Jiwei Wei, Yang Yang, Guoqing Wang, Heng Tao Shen*  
[[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Anwaar_Compositional_Learning_of_Image-Text_Query_for_Image_Retrieval_WACV_2021_paper.pdf)]

**[WACV, 2024] Text-to-Image Editing by Image Information Removal**  
*Zhongping Zhang, Jian Zheng, Zhiyuan Fang, Bryan A. Plummer*  
[[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Text-to-Image_Editing_by_Image_Information_Removal_WACV_2024_paper.pdf)]

**[WACV, 2024] Shape-Guided Diffusion with Inside-Outside Attention**  
*Dong Huk Park, Grace Luo, Clayton Toste, Samaneh Azadi, Xihui Liu, Maka Karalashvili, Anna Rohrbach, Trevor Darrell*  
[[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Park_Shape-Guided_Diffusion_With_Inside-Outside_Attention_WACV_2024_paper.pdf)]

### 2023

**[IEEE Access, 2023] Text-Guided Image Manipulation via Generative Adversarial Network With Referring Image Segmentation-Based Guidance**  
*Yuto Watanabe, Ren Togo, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10107599)] 

**[arXiv, 2023] InstructEdit: Improving Automatic Masks for Diffusion-Based Image Editing with User Instructions**  
*Qian Wang, Biao Zhang, Michael Birsak, Peter Wonka*  
[[Paper](https://arxiv.org/pdf/2305.18047)] [[GitHub](https://github.com/QianWangX/InstructEdit)]

**[ICLR, 2023] DiffEdit: Diffusion-Based Semantic Image Editing with Mask Guidance**  
*Guillaume Couairon, Jakob Verbeek, Holger Schwenk, Matthieu Cord*  
[[Paper](https://arxiv.org/pdf/2210.11427)] [[GitHub](https://github.com/huggingface/diffusers/issues/2800)]

**[CVPR, 2023] SINE: Single Image Editing with Text-to-Image Diffusion Models**  
*Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N Metaxas, Jian Ren*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_SINE_SINgle_Image_Editing_With_Text-to-Image_Diffusion_Models_CVPR_2023_paper.pdf)] [[GitHub](https://github.com/zhang-zx/SINE)]

**[CVPR, 2023] Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation**  
*Narek Tumanyan, Michal Geyer, Shai Bagon, Tali Dekel*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tumanyan_Plug-and-Play_Diffusion_Features_for_Text-Driven_Image-to-Image_Translation_CVPR_2023_paper.pdf)] [[GitHub](https://github.com/MichalGeyer/plug-and-play)]

**[arXiv, 2023] PRedItOR: Text Guided Image Editing with Diffusion Prior**  
*Hareesh Ravi, Sachin Kelkar, Midhun Harikumar, Ajinkya Kale*  
[[Paper](https://arxiv.org/pdf/2302.07979)]

**[TOG, 2023] Unitune: Text-Driven Image Editing by Fine Tuning a Diffusion Model on a Single Image**  
*Dani Valevski, Matan Kalman, Eyal Molad, Eyal Segalis, Yossi Matias, Yaniv Leviathan*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3592451)] [[GitHub](https://github.com/xuduo35/UniTune)]

**[arXiv, 2023] Custom-Edit: Text-Guided Image Editing with Customized Diffusion Models**  
*Jooyoung Choi, Yunjey Choi, Yunji Kim, Junho Kim, Sungroh Yoon*  
[[Paper](https://arxiv.org/pdf/2305.15779)] [[GitHub](https://github.com/taki0112/taki0112)]

**[CVPR, 2023] Imagic: Text-Based Real Image Editing with Diffusion Models**  
*Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, Michal Irani*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kawar_Imagic_Text-Based_Real_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.pdf)] [[GitHub](https://github.com/huggingface/diffusers/issues/895)]

**[ICLR, 2023] Diffusion-Based Image Translation Using Disentangled Style and Content Representation**  
*Gihyun Kwon, Jong Chul Ye*  
[[Paper](https://arxiv.org/pdf/2209.15264)] [[GitHub](https://github.com/cyclomon/DiffuseIT)]

**[arXiv, 2023] MDP: A Generalized Framework for Text-Guided Image Editing by Manipulating the Diffusion Path**  
*Qian Wang, Biao Zhang, Michael Birsak, Peter Wonka*  
[[Paper](https://arxiv.org/pdf/2303.16765)] [[GitHub](https://github.com/QianWangX/MDP-Diffusion)]

**[CVPR, 2023] InstructPix2Pix: Learning to Follow Image Editing Instructions**  
*Tim Brooks, Aleksander Holynski, Alexei A. Efros*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.pdf)] [[GitHub](https://github.com/timothybrooks/instruct-pix2pix)]

**[ICCV, 2023] Prompt Tuning Inversion for Text-Driven Image Editing Using Diffusion Models**  
*Wenkai Dong, Song Xue, Xiaoyue Duan, Shumin Han*  
[[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Dong_Prompt_Tuning_Inversion_for_Text-driven_Image_Editing_Using_Diffusion_Models_ICCV_2023_paper.pdf)] 

**[arXiv, 2023] DeltaSpace: A Semantic-Aligned Feature Space for Flexible Text-Guided Image Editing**  
*Yueming Lyu, Kang Zhao, Bo Peng, Yue Jiang, Yingya Zhang, Jing Dong*  
[[Paper](https://arxiv.org/pdf/2310.08785)]

**[AAAI, 2023] DE-Net: Dynamic Text-Guided Image Editing Adversarial Networks**  
*Ming Tao, Bing-Kun Bao, Hao Tang, Fei Wu, Longhui Wei, Qi Tian*  
[[Paper](https://arxiv.org/pdf/2206.01160)] [[GitHub](https://github.com/tobran/DE-Net)]

### 2022

**[ACM MM, 2022] LS-GAN: Iterative Language-Based Image Manipulation via Long and Short Term Consistency Reasoning**  
*Gaoxiang Cong, Liang Li, Zhenhuan Liu, Yunbin Tu, Weijun Qin, Shenyuan Zhang, Chengang Yan, Wenyu Wang, Bin Jiang*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3503161.3548206)]

**[arXiv, 2022] FEAT: Face Editing with Attention**  
*Xianxu Hou, Linlin Shen, Or Patashnik, Daniel Cohen-Or, Hui Huang*  
[[Paper](https://arxiv.org/pdf/2202.02713)]

**[ECCV, 2022] VQGAN-CLIP: Open Domain Image Generation and Editing with Natural Language Guidance**  
*Katherine Crowson, Stella Biderman, Daniel Kornis, Dashiell Stander, Eric Hallahan, Louis Castricato, Edward Raff*  
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136970088.pdf)] [[GitHub](https://github.com/EleutherAI/vqgan-clip)]

**[ICML, 2022] GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**  
*Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, Mark Chen*  
[[Paper](https://proceedings.mlr.press/v162/nichol22a/nichol22a.pdf)] [[GitHub](https://github.com/openai/glide-text2im)]

**[WACV, 2022] StyleMC: Multi-Channel Based Fast Text-Guided Image Generation and Manipulation**  
*Umut Kocasari, Alara Dirik, Mert Tiftikci, Pinar Yanardag*  
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Kocasari_StyleMC_Multi-Channel_Based_Fast_Text-Guided_Image_Generation_and_Manipulation_WACV_2022_paper.pdf)] [[GitHub](https://github.com/catlab-team/stylemc)] [[website](https://catlab-team.github.io/stylemc/)]

**[CVPR, 2022] HairCLIP: Design Your Hair by Text and Reference Image**  
*Tianyi Wei, Dongdong Chen, Wenbo Zhou, Jing Liao, Zhentao Tan, Lu Yuan, Weiming Zhang, Nenghai Yu*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_HairCLIP_Design_Your_Hair_by_Text_and_Reference_Image_CVPR_2022_paper.pdf)] [[GitHub](https://github.com/wty-ustc/HairCLIP)]

**[NeurIPS, 2022] One Model to Edit Them All: Free-Form Text-Driven Image Manipulation with Semantic Modulations**  
*Yiming Zhu, Hongyu Liu, Yibing Song, Ziyang Yuan, Xintong Han, Chun Yuan, Qifeng Chen, Jue Wang*  
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a0a53fefef4c2ad72d5ab79703ba70cb-Paper-Conference.pdf)] [[GitHub](https://github.com/KumapowerLIU/FFCLIP)]

**[CVPR, 2022] Predict, Prevent, and Evaluate: Disentangled Text-Driven Image Manipulation Empowered by Pre-Trained Vision-Language Model**  
*Zipeng Xu, Tianwei Lin, Hao Tang, Fu Li, Dongliang He, Nicu Sebe, Radu Timofte, Luc Van Gool, Errui Ding*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Predict_Prevent_and_Evaluate_Disentangled_Text-Driven_Image_Manipulation_Empowered_by_CVPR_2022_paper.pdf)] [[GitHub](https://github.com/zipengxuc/PPE)]

**[CVPR, 2022] Blended Diffusion for Text-Driven Editing of Natural Images**  
*Omri Avrahami, Dani Lischinski, Ohad Fried*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Avrahami_Blended_Diffusion_for_Text-Driven_Editing_of_Natural_Images_CVPR_2022_paper.pdf)] [[GitHub](https://github.com/omriav/blended-diffusion)]

**[CVPR, 2022] DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation**  
*Gwanghyun Kim, Taesung Kwon, Jong Chul Ye*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.pdf)] [[GitHub](https://github.com/gwang-kim/DiffusionCLIP)]

**[ICLR, 2022] SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations**  
*Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon*  
[[Paper](https://openreview.net/pdf?id=aBsCjcPu_tE)] [[GitHub](https://github.com/ermongroup/SDEdit)] [[Website](https://sde-image-editing.github.io/)]

### 2021

**[CVPR, 2021] TediGAN: Text-guided diverse face image generation and manipulation**  
*Weihao Xia, Yujiu Yang, Jing-Hao Xue, Baoyuan Wu*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xia_TediGAN_Text-Guided_Diverse_Face_Image_Generation_and_Manipulation_CVPR_2021_paper.pdf)] [[GitHub](https://github.com/IIGROUP/TediGAN)]

**[ICIP, 2021] Segmentation-Aware Text-Guided Image Manipulation**  
*Tomoki Haruyama, Ren Togo, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9506601)] [[GitHub](#)]

**[IJPR, 2021] FocusGAN: Preserving Background in Text-Guided Image Editing**  
*Liuqing Zhao, Linyan Li, Fuyuan Hu, Zhenping Xia, Rui Yao*  
[[Paper](https://arxiv.org/pdf/2405.19708)]

**[ICCV, 2021] StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery**  
*Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, Dani Lischinski*  
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Patashnik_StyleCLIP_Text-Driven_Manipulation_of_StyleGAN_Imagery_ICCV_2021_paper.pdf)] [[GitHub](https://github.com/orpatashnik/StyleCLIP)]

**[MM, 2021] Text as Neural Operator: Image Manipulation by Text Instruction**  
*Tianhao Zhang, Hung-Yu Tseng, Lu Jiang, Weilong Yang, Honglak Lee, Irfan Essa*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3474085.3475343)] [[GitHub](https://github.com/google/tim-gan)]

**[CVPR, 2021] TediGAN: Text-guided diverse face image generation and manipulation**  
*Weihao Xia, Yujiu Yang, Jing-Hao Xue, Baoyuan Wu*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xia_TediGAN_Text-Guided_Diverse_Face_Image_Generation_and_Manipulation_CVPR_2021_paper.pdf)] [[GitHub](https://github.com/IIGROUP/TediGAN)]

**[arXiv, 2021] Paint by Word**  
*Alex Andonian, Sabrina Osmany, Audrey Cui, YeonHwan Park, Ali Jahanian, Antonio Torralba, David Bau*  
[[Paper](https://arxiv.org/pdf/2103.10951)] [[GitHub](https://github.com/alexandonian/paint-by-word)] [[Website](http://paintbyword.csail.mit.edu/)]

**[CVPR, 2021] Learning by Planning: Language-Guided Global Image Editing**  
*Jing Shi, Ning Xu, Yihang Xu, Trung Bui, Franck Dernoncourt, Chenliang Xu*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shi_Learning_by_Planning_Language-Guided_Global_Image_Editing_CVPR_2021_paper.pdf)] [[GitHub](https://jshi31.github.io/T2ONet/)]

### 2020

**[ACM MM, 2020] IR-GAN: Image Manipulation with Linguistic Instruction by Increment Reasoning**  
*Zhenhuan Liu, Jincan Deng, Liang Li, Shaofei Cai, Qianqian Xu, Shuhui Wang, Qingming Huang*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413777)] [[GitHub](https://github.com/Victarry/IR-GAN-Code)]

**[CVPR, 2020] ManiGAN: Text-Guided Image Manipulation**  
*Bowen Li, Xiaojuan Qi, Thomas Lukasiewicz, Philip HS Torr*  
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_ManiGAN_Text-Guided_Image_Manipulation_CVPR_2020_paper.pdf)] [[GitHub](https://github.com/mrlibw/ManiGAN)]

**[NeurIPS, 2020] Lightweight Generative Adversarial Networks for Text-Guided Image Manipulation**  
*Bowen Li, Xiaojuan Qi, Philip Torr, Thomas Lukasiewicz*  
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/fae0b27c451c728867a567e8c1bb4e53-Paper.pdf)] [[GitHub](https://github.com/mrlibw/Lightweight-Manipulation)]

**[LNCS, 2020] CAFE-GAN: Arbitrary Face Attribute Editing with Complementary Attention Feature**  
*Jeong-gi Kwak, David K. Han, Hanseok Ko*  
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590511.pdf)] [[GitHub](https://github.com/zuael/CAFE-GAN-Pytorch)]

**[ECCV, 2020] Open-Edit: Open-Domain Image Manipulation with Open-Vocabulary Instructions**  
*Xihui Liu, Zhe Lin, Jianming Zhang, Handong Zhao, Quan Tran, Xiaogang Wang, Hongsheng Li*  
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560086.pdf)] [[GitHub](https://github.com/xh-liu/Open-Edit)]

### 2019

**[ICASSP, 2019] Bilinear Representation for Language-based Image Editing Using Conditional Generative Adversarial Networks**  
*Xiaofeng Mao, Yuefeng Chen, Yuhong Li, Tao Xiong, Yuan He, Hui Xue*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683008)] [[GitHub](https://github.com/vtddggg/BilinearGAN_for_LBIE)]

### 2018

**[NeurIPS, 2018] Text-Adaptive Generative Adversarial Networks: Manipulating Images with Natural Language**  
*Seonghyeon Nam, Yunji Kim, Seon Joo Kim*  
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/d645920e395fedad7bbbed0eca3fe2e0-Paper.pdf)] [[GitHub](https://github.com/woozzu/tagan)]

**[CVPR, 2018] Language-based image editing with recurrent attentive models**  
*Jianbo Chen, Yelong Shen, Jianfeng Gao, Jingjing Liu, Xiaodong Liu*  
[[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Language-Based_Image_Editing_CVPR_2018_paper.pdf)] 

**[arXiv, 2018] Interactive Image Manipulation with Natural Language Instruction Commands**  
*Seitaro Shinagawa, Koichiro Yoshino, Sakriani Sakti, Yu Suzuki, Satoshi Nakamura*  
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Morita_Interactive_Image_Manipulation_With_Complex_Text_Instructions_WACV_2023_paper.pdf)]

**[CVPR, 2018] Language-based image editing with recurrent attentive models**  
*Jianbo Chen, Yelong Shen, Jianfeng Gao, Jingjing Liu, Xiaodong Liu*  
[[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Language-Based_Image_Editing_CVPR_2018_paper.pdf)]

### 2017

**[ICCV, 2017] Semantic image synthesis via adversarial learning**  
*Hao Dong, Simiao Yu, Chao Wu, Yike Guo*  
[[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dong_Semantic_Image_Synthesis_ICCV_2017_paper.pdf)] [[GitHub](https://github.com/woozzu/dong_iccv_2017)]




## Image-text composite retrieval

### 2024
**[AAAI, 2024] Dynamic weighted combiner for mixed-modal image retrieval** \
*Fuxiang Huang, Lei Zhang, Xiaowei Fu, Suqi Song* \
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28004/28023)] [[GitHub](https://github.com/fuxianghuang1/DWC)]

**[ICMR, 2024] Enhancing Interactive Image Retrieval With Query Rewriting Using Large Language Models and Vision Language Models**  
*Hongyi Zhu, Jia-Hong Huang, Stevan Rudinac, Evangelos Kanoulas*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3652583.3658032)] [[GitHub](https://github.com/s04240051/Multimodal-Conversational-Retrieval-System)]

**[ACM MM, 2024] Improving Composed Image Retrieval via Contrastive Learning with Scaling Positives and Negatives**  
*Zhangchi Feng, Richong Zhang, Zhijie Nie*  
[[Paper](https://arxiv.org/abs/2404.11317)] [[GitHub](https://github.com/BUAADreamer/SPN4CIR)]

**[CVPR, 2024] Language-only Training of Zero-shot Composed Image Retrieval**  
*Geonmo Gu, Sanghyuk Chun, Wonjae Kim, Yoohoon Kang, Sangdoo Yun*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Gu_Language-only_Training_of_Zero-shot_Composed_Image_Retrieval_CVPR_2024_paper.pdf)] [[GitHub](https://github.com/navervision/lincir)]

**[AAAI, 2024] Context-I2W: Mapping Images to Context-dependent Words for Accurate Zero-Shot Composed Image Retrieval**  
*Yuanmin Tang, Jing Yu, Keke Gai, Jiamin Zhuang, Gang Xiong, Yue Hu, Qi Wu*  
[[Paper](https://arxiv.org/pdf/2309.16137)] [[GitHub](https://github.com/Pter61/context-i2w)]

**[CVPR, 2024] Knowledge-enhanced dual-stream zero-shot composed image retrieval**  
*Yucheng Suo, Fan Ma, Linchao Zhu, Yi Yang*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Suo_Knowledge-Enhanced_Dual-stream_Zero-shot_Composed_Image_Retrieval_CVPR_2024_paper.pdf)]

### 2023

**[CVPR, 2023] Fame-vil: Multi-tasking vision-language model for heterogeneous fashion tasks**  
*Xiao Han, Xiatian Zhu, Licheng Yu, Li Zhang, Yi-Zhe Song, Tao Xiang*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Han_FAME-ViL_Multi-Tasking_Vision-Language_Model_for_Heterogeneous_Fashion_Tasks_CVPR_2023_paper.pdf)] [[GitHub](https://github.com/BrandonHanx/FAME-ViL)]

**[ICCV, 2023] FashionNTM: Multi-turn fashion image retrieval via cascaded memory**  
*Anwesan Pal, Sahil Wadhwa, Ayush Jaiswal, Xu Zhang, Yue Wu, Rakesh Chada, Pradeep Natarajan, Henrik I Christensen*  
[[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Pal_FashionNTM_Multi-turn_Fashion_Image_Retrieval_via_Cascaded_Memory_ICCV_2023_paper.pdf)]

**[CVPR, 2023] Pic2word: Mapping pictures to words for zero-shot composed image retrieval**  
*Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, Tomas Pfister*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Saito_Pic2Word_Mapping_Pictures_to_Words_for_Zero-Shot_Composed_Image_Retrieval_CVPR_2023_paper.pdf)] [[GitHub](https://github.com/google-research/composed_image_retrieval)]

**[arXiv, 2023] Pretrain like you inference: Masked tuning improves zero-shot composed image retrieval**  
*Junyang Chen, Hanjiang Lai*  
[[Paper](https://arxiv.org/abs/2311.07622)] [[GitHub](#)]

**[ICCV, 2023] Zero-shot composed image retrieval with textual inversion**  
*Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, Alberto Del Bimbo*  
[[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Baldrati_Zero-Shot_Composed_Image_Retrieval_with_Textual_Inversion_ICCV_2023_paper.pdf)] [[GitHub](https://github.com/miccunifi/SEARLE)]

**[ACM, 2023] Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features**  
*Alberto Baldrati, Marco Bertini, Tiberio Uricchio, Alberto Del Bimbo*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3617597)] [[GitHub](https://github.com/ABaldrati/CLIP4Cir)]


### 2022

**[IEEE, 2022] Adversarial and isotropic gradient augmentation for image retrieval with text feedback**  
*Fuxiang Huang, Lei Zhang, Yuhang Zhou, Xinbo Gao*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9953564)]

**[TOMM, 2022] Tell, imagine, and search: End-to-end learning for composing text and image to image retrieval**  
*Feifei Zhang, Mingliang Xu, Changsheng Xu*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3478642)]

**[arXiv, 2022] Image Search with Text Feedback by Additive Attention Compositional Learning**  
*Yuxin Tian, Shawn Newsam, Kofi Boakye*  
[[Paper](https://arxiv.org/pdf/2203.03809)]

**[IEEE, 2022] Heterogeneous feature alignment and fusion in cross-modal augmented space for composed image retrieval**  
*Huaxin Pang, Shikui Wei, Gangjian Zhang, Shiyin Zhang, Shuang Qiu, Yao Zhao*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3474085.3475659)]


**[ICLR, 2022] ARTEMIS: Attention-based Retrieval with Text-Explicit Matching and Implicit Similarity**  
*Ginger Delmas, Rafael S. Rezende, Gabriela Csurka, Diane Larlus*  
[[Paper](https://openreview.net/pdf?id=CVfLvQq9gLo)] [[GitHub](https://github.com/naver/artemis)]

**[WACV, 2022] SAC: Semantic attention composition for text-conditioned image retrieval**  
*Surgan Jandial, Pinkesh Badjatiya, Pranit Chawla, Ayush Chopra, Mausoom Sarkar, Balaji Krishnamurthy*  
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Jandial_SAC_Semantic_Attention_Composition_for_Text-Conditioned_Image_Retrieval_WACV_2022_paper.pdf)]

**[ACM TOMCCAP, 2022] AMC: Adaptive Multi-expert Collaborative Network for Text-guided Image Retrieval**  
*Hongguang Zhu,  Yunchao Wei, Yao Zhao, Chunjie Zhang, Shujuan Huang*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3584703)][[GitHub](https://github.com/KevinLight831/AMC)]

### 2021

**[SIGIR, 2021] Comprehensive linguistic-visual composition network for image retrieval**  
*Haokun Wen, Xuemeng Song, Xin Yang, Yibing Zhan, Liqiang Nie*  
[[Paper](https://liqiangnie.github.io/paper/Comprehensive%20Linguistic-Visual%20Composition%20Network%20for%20Image%20Retrieval.pdf)]

**[AAAI, 2021] Dual compositional learning in interactive image retrieval**  
*Jongseok Kim, Youngjae Yu, Hoeseong Kim, Gunhee Kim*  
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16271/16078)] [[GitHub](https://github.com/ozmig77/dcnet)]

**[CVPRW, 2021] Leveraging Style and Content features for Text Conditioned Image Retrieval**  
*Pranit Chawla, Surgan Jandial, Pinkesh Badjatiya, Ayush Chopra, Mausoom Sarkar, Balaji Krishnamurthy*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2021W/CVFAD/papers/Chawla_Leveraging_Style_and_Content_Features_for_Text_Conditioned_Image_Retrieval_CVPRW_2021_paper.pdf)]

**[ICCV, 2021] Image retrieval on real-life images with pre-trained vision-and-language models**  
*Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, Stephen Gould*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9710082)] [[GitHub](https://github.com/Cuberick-Orion/CIRPLANT)]

**[SIGIR, 2021] Conversational fashion image retrieval via multiturn natural language feedback**  
*Yifei Yuan, Wai Lam*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3462881)] [[GitHub](https://github.com/yfyuan01/MultiturnFashionRetrieval)]

**[WACV, 2021] Compositional learning of image-text query for image retrieval**  
*Muhammad Umer Anwaar, Egor Labintcev, Martin Kleinsteuber*  
[[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Anwaar_Compositional_Learning_of_Image-Text_Query_for_Image_Retrieval_WACV_2021_paper.pdf)] [[GitHub](https://github.com/ecom-research/ComposeAE)]



### 2020

**[ECCV, 2020] Learning joint visual semantic matching embeddings for language-guided retrieval**  
*Yanbei Chen, Loris Bazzani*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Banani_Learning_Visual_Representations_via_Language-Guided_Sampling_CVPR_2023_paper.pdf)]

**[arXiv, 2020] CurlingNet: Compositional Learning between Images and Text for Fashion IQ Data**  
*Youngjae Yu, Seunghwan Lee, Yuncheol Choi, Gunhee Kim*  
[[Paper](https://arxiv.org/pdf/2003.12299)]

**[CVPR, 2020] Image search with text feedback by visiolinguistic attention learning**  
*Yanbei Chen, Shaogang Gong, Loris Bazzani*  
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Image_Search_With_Text_Feedback_by_Visiolinguistic_Attention_Learning_CVPR_2020_paper.html)] [[GitHub](https://github.com/yanbeic/VAL?tab=readme-ov-file)]

**[arXiv, 2020] Modality-Agnostic Attention Fusion for visual search with text feedback**  
*Eric Dodds, Jack Culpepper, Simao Herdade, Yang Zhang, Kofi Boakye*  
[[Paper](https://arxiv.org/pdf/2007.00145)] [[GitHub](https://github.com/yahoo/maaf)]

### 2018

**[CVPR, 2018] Language-based image editing with recurrent attentive models**  
*Jianbo Chen, Yelong Shen, Jianfeng Gao, Jingjing Liu, Xiaodong Liu*  
[[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Language-Based_Image_Editing_CVPR_2018_paper.pdf)]

**[NeurIPS, 2018] Dialog-based interactive image retrieval**  
*Xiaoxiao Guo, Hui Wu, Yu Cheng, Steven Rennie, Gerald Tesauro, Rogerio Feris*  
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/a01a0380ca3c61428c26a231f0e49a09-Paper.pdf)] [[GitHub](https://github.com/XiaoxiaoGuo/fashion-retrieval)]

### 2017

**[ICCV, 2017] Automatic spatially-aware fashion concept discovery**  
*Xintong Han, Zuxuan Wu, Phoenix X Huang, Xiao Zhang, Menglong Zhu, Yuan Li, Yang Zhao, Larry S Davis*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8237425)]

**[ICCV, 2017] Be your own prada: Fashion synthesis with structural coherence**  
*Shizhan Zhu, Raquel Urtasun, Sanja Fidler, Dahua Lin, Chen Change Loy*  
[[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Be_Your_Own_ICCV_2017_paper.pdf)] [[GitHub](https://github.com/zhusz/ICCV17-fashionGAN)]

## Other mutimodal composite retrieval

### 2024

**[CVPR, 2024] Tri-modal motion retrieval by learning a joint embedding space**  
*Kangning Yin, Shihao Zou, Yuxuan Ge, Zheng Tian*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_Tri-Modal_Motion_Retrieval_by_Learning_a_Joint_Embedding_Space_CVPR_2024_paper.pdf)]

**[WACV, 2024] Modality-Aware Representation Learning for Zero-shot Sketch-based Image Retrieval**  
*Eunyi Lyou, Doyeon Lee, Jooeun Kim, Joonseok Lee*  
[[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Lyou_Modality-Aware_Representation_Learning_for_Zero-Shot_Sketch-Based_Image_Retrieval_WACV_2024_paper.pdf)] [[GitHub](https://github.com/eunyi-lyou/MA-ZS-SBIR)]

**[CVPR, 2024] Pros: Prompting-to-simulate generalized knowledge for universal cross-domain retrieval**  
*Kaipeng Fang, Jingkuan Song, Lianli Gao, Pengpeng Zeng, Zhi-Qi Cheng, Xiyao Li, Heng Tao Shen*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Fang_ProS_Prompting-to-simulate_Generalized_knowledge_for_Universal_Cross-Domain_Retrieval_CVPR_2024_paper.pdf)] [[GitHub](https://github.com/fangkaipeng/ProS)]

**[CVPR, 2024] You'll Never Walk Alone: A Sketch and Text Duet for Fine-Grained Image Retrieval**  
*Subhadeep Koley, Ayan Kumar Bhunia, Aneeshan Sain, Pinaki Nath Chowdhury, Tao Xiang, Yi-Zhe Song*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Koley_Youll_Never_Walk_Alone_A_Sketch_and_Text_Duet_for_CVPR_2024_paper.pdf)]

**[AAAI, 2024] T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models**  
*Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan*  
[[Paper](https://arxiv.org/pdf/2302.08453)] [[GitHub](https://github.com/TencentARC/T2I-Adapter)]

**[IEEE/CVF, 2024] TriCoLo: Trimodal contrastive loss for text to shape retrieval**  
*Yue Ruan, Han-Hung Lee, Yiming Zhang, Ke Zhang, Angel X Chang*  
[[Paper](https://openaccess.thecvf.com/content/WACV2024/html/Ruan_TriCoLo_Trimodal_Contrastive_Loss_for_Text_to_Shape_Retrieval_WACV_2024_paper.html)] [[GitHub](https://github.com/3dlg-hcvc/tricolo)]

### 2023

**[CVPR, 2023] SceneTrilogy: On Human Scene-Sketch and its Complementarity with Photo and Text**  
*Pinaki Nath Chowdhury, Ayan Kumar Bhunia, Aneeshan Sain, Subhadeep Koley, Tao Xiang, Yi-Zhe Song*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chowdhury_SceneTrilogy_On_Human_Scene-Sketch_and_Its_Complementarity_With_Photo_and_CVPR_2023_paper.pdf)]

### 2022

**[ECCV, 2022] A sketch is worth a thousand words: Image retrieval with text and sketch**  
*Patsorn Sangkloy, Wittawat Jitkrittum, Diyi Yang, James Hays*  
[[Paper](https://dl.acm.org/doi/abs/10.1007/978-3-031-19839-7_15)]

**[ECCV, 2022] Motionclip: Exposing human motion generation to clip space**  
*Guy Tevet, Brian Gordon, Amir Hertz, Amit H Bermano, Daniel Cohen-Or*  
[[Paper](https://arxiv.org/pdf/2203.08063)] [[GitHub](https://github.com/GuyTevet/MotionCLIP)]

**[IEEE, 2022] Multimodal Fusion Remote Sensing Image–Audio Retrieval**  
*Rui Yang, Shuang Wang, Yingzhi Sun, Huan Zhang, Yu Liao, Yu Gu, Biao Hou, Licheng Jiao*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9847567)]

### 2021

**[CVPR, 2021] Connecting what to say with where to look by modeling human attention traces**  
*Zihang Meng, Licheng Yu, Ning Zhang, Tamara L Berg, Babak Damavandi, Vikas Singh, Amy Bearman*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Meng_Connecting_What_To_Say_With_Where_To_Look_by_Modeling_CVPR_2021_paper.pdf)] [[GitHub](https://github.com/facebookresearch/connect-caption-and-trace)]

**[ICCV, 2021] Telling the what while pointing to the where: Multimodal queries for image retrieval**  
*Soravit Changpinyo, Jordi Pont-Tuset, Vittorio Ferrari, Radu Soricut*  
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Changpinyo_Telling_the_What_While_Pointing_to_the_Where_Multimodal_Queries_ICCV_2021_paper.pdf)]

### 2020

**[arXiv, 2020] A Feature Analysis for Multimodal News Retrieval**  
*Golsa Tahmasebzadeh, Sherzod Hakimov, Eric Müller-Budack, Ralph Ewerth*  
[[Paper](https://arxiv.org/abs/2007.06390)] [[GitHub](https://github.com/cleopatra-itn/multimodal-news-retrieval)]

### 2019

**[Multimedia Tools and Applications, 2019] Efficient and interactive spatial-semantic image retrieval**  
*Ryosuke Furuta, Naoto Inoue, Toshihiko Yamasaki*  
[[Paper](https://link.springer.com/content/pdf/10.1007/s11042-018-7148-1.pdf)]

**[arXiv, 2019] Query by Semantic Sketch**  
*Luca Rossetto, Ralph Gasser, Heiko Schuldt*  
[[Paper](https://arxiv.org/abs/1909.01477)]

### 2017

**[IJCNLP, 2017] Draw and tell: Multimodal descriptions outperform verbal-or sketch-only descriptions in an image retrieval task**  
*Ting Han, David Schlangen*  
[[Paper](https://tingh.github.io/files/papers/sketch_ijcnlp_short.pdf)]

**[CVPR, 2017] Spatial-Semantic Image Search by Visual Feature Synthesis**  
*Long Mai, Hailin Jin, Zhe Lin, Chen Fang, Jonathan Brandt, Feng Liu*  
[[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mai_Spatial-Semantic_Image_Search_CVPR_2017_paper.pdf)]

**[ACM Multimedia, 2017] Region-based image retrieval revisited**  
*Ryota Hinami, Yusuke Matsui, Shin'ichi Satoh*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3123266.3123312)]

### 2014

**[Cancer Informatics, 2014] Medical image retrieval: a multimodal approach**  
*Yu Cao, Shawn Steffey, Jianbiao He, Degui Xiao, Cui Tao, Ping Chen, Henning Müller*  
[[Paper](https://journals.sagepub.com/doi/pdf/10.4137/CIN.S14053)]

### 2013

**[10th Conference on Open Research Areas in Information Retrieval, 2013] NovaMedSearch: a multimodal search engine for medical case-based retrieval**  
*André Mourão, Flávio Martins*  
[[Paper](https://dl.acm.org/doi/pdf/10.5555/2491748.2491798)]

**[12th International Conference on Document Analysis and Recognition, 2013] Multi-modal Information Integration for Document Retrieval**  
*Ehtesham Hassan, Santanu Chaudhury, M. Gopal*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6628804)]

### 2003

**[EURASIP, 2003] Semantic indexing of multimedia content using visual, audio, and text cues**  
*WH Adams, Giridharan Iyengar, Ching-Yung Lin, Milind Ramesh Naphade, Chalapathy Neti, Harriet J Nock, John R Smith*  
[[Paper](https://asp-eurasipjournals.springeropen.com/counter/pdf/10.1155/S1110865703211173.pdf)]


# Datasets
## Datasets for image-text composite editing

| **Dataset**                     | **Modalities**                    | **Scale**                                | **Link**                                                                                           |
|---------------------------------|-----------------------------------|------------------------------------------|----------------------------------------------------------------------------------------------------|
| Caltech-UCSD Birds(CUB)         | Images, Captions                  | 11K images, 11K attributes               | [Link](https://www.vision.caltech.edu/datasets/cub_200_2011/)                                      |
| Oxford-102 flower               | Images, Captions                  | 8K images, 8K attributes                 | [Link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)                                         |
| CelebFaces Attributes (CelebA)  | Images, Captions                  | 202K images, 8M attributes               | [Link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)                                          |
| DeepFashion (Fashion Synthesis) | Images, Captions                  | 78K images, -                            | [Link](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html)                    |
| MIT-Adobe 5k                    | Images, Captions                  | 5K images, 20K texts                     | [Link](https://data.csail.mit.edu/graphics/fivek/)                                                 |
| MS-COCO                         | Image, Caption                    | 164K images, 616K texts                  | [Link](https://cocodataset.org/)                                                                   |
| ReferIt                         | Image, Caption                    | 19K images, 130K text                    | [Link](https://github.com/lichengunc/refer)                                                        |
| CLEVR                           | 3D images, Questions              | 100K images, 865K questions              | [Link](https://cs.stanford.edu/people/jcjohns/clevr/)                                              |
| i-CLEVR                         | 3D image, Instruction             | 10K sequences, 50K instructions          | [Link](https://github.com/topics/i-clevr)                                                          |
| CSS                             | 3D images, 2D images, Instructions| 34K images, -                            | [Link](https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view?pli=1)               |
| CoDraw                          | images, text instructions         | 9K images, -                             | [Link](https://github.com/facebookresearch/CoDraw)                                                 |
| Cityscapes                      | images, Captions                  | 25K images, -                            | [Link](https://www.cityscapes-dataset.com/)                                                        |
| Zap-Seq                         | image sequences, Captions         | 8K images, 18K texts                     | -                                                                                                  |
| DeepFashion-Seq                 | image sequences, Captions         | 4K images, 12K texts                     | -                                                                                                  |
| FFHQ                            | Images                            | 70K images                               | [Link](https://github.com/NVlabs/ffhq-dataset)                                                     |
| LSUN                            | Images                            | 1M images                                | [Link](https://github.com/fyu/lsun/blob/master/README.md)                                          |
| Animal FacesHQ (AFHQ)           | Images                            | 15K images                               | [Link](https://www.kaggle.com/datasets/andrewmvd/animal-faces/code)                                 |
| CelebA-HQ                       | Images                            | 30K images                               | [Link](https://github.com/tkarras/progressive_growing_of_gans)                                     |
| Animal faces                    | Images                            | 16K images                               | [Link](https://www.kaggle.com/datasets/andrewmvd/animal-faces)                                     |
| Landscapes                      | Images                            | 4K images                                | [Link](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)                                |

## Datasets for image-text composite retrieval
| **Dataset**                     | **Modalities**                    | **Scale**                                | **Link**                                                                                           |
|---------------------------------|-----------------------------------|------------------------------------------|----------------------------------------------------------------------------------------------------|
| Fashion200k                     | Image, Captions                   | 200K images, 200K text                   | [Link](https://github.com/xthan/fashion-200k)                                                      |
| MIT-States                      | Image, Captions                   | 53K images, 53K text                     | [Link](https://web.mit.edu/phillipi/Public/states_and_transformations/index.html)                  |
| Fashion IQ                      | Image, Captions                   | 77K images, -                            | [Link](https://github.com/XiaoxiaoGuo/fashion-iq)                                                  |
| CIRR                            | Image, Captions                   | 21K images, -                            | [Link](https://github.com/Cuberick-Orion/CIRR)                                                     |
| CSS                             | 3D images, 2D images, Instructions| 34K images, -                            | [Link](https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view?pli=1)               |
| Shoes                           | Images                            | 14K images                               | [Link](https://www.kaggle.com/datasets/noobyogi0100/shoe-dataset)                                  |
| Birds-to-Words                  | Images, Captions                  | -                                        | [Link](https://github.com/google-research-datasets/birds-to-words)                                 |
| SketchyCOCO                     | Images, Sketches                  | 14K sketches, 14K photos                 | [Link](https://github.com/sysu-imsl/SketchyCOCO)                                                   |
| FSCOCO                          | Images, Sketches                  | 10K sketches                             | [Link](https://www.pinakinathc.me/fscoco/)                                                         |

## Datasets for other mutimodal composite retrieval
| **Dataset**                     | **Modalities**                    | **Scale**                                | **Link**                                                                                           |
|---------------------------------|-----------------------------------|------------------------------------------|----------------------------------------------------------------------------------------------------|
| HumanML3D                       | Motions, Captions                 | 14K motion sequences, 44K text           | [Link](https://github.com/EricGuo5513/HumanML3D)                                                   |
| KIT-ML                          | Motions, Captions                 | 3K motion sequences, 6K text             | [Link](https://h2t.iar.kit.edu/english/1445.php)                                                   |
| Text2Shape                      | Shapes, Captions                  | 6K chairs, 8K tables, 70K text           | [Link](https://github.com/kchen92/text2shape)                                                      |
| Flickr30k LocNar                | Images, Captions                  | 31K images, 155K texts                   | [Link](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)                            |
| Conceptual Captions             | Images, Captions                  | 3.3M images, 33M texts                   | [Link](https://github.com/google-research-datasets/conceptual-captions)                            |
| Sydney_IV                       | RS Images, Audio Captions         | 613 images, 3K audio descriptions        | [Link](https://github.com/201528014227051/RSICD_optimal)                                           |
| UCM_IV                          | Images, Audio Captions            | 2K images, 10K audio descriptions        | [Link](https://github.com/201528014227051/RSICD_optimal)                                           |
| RSICD_IV                        | Image, Audio Captions             | 11K images, 55K audio descriptions       | [Link](https://github.com/201528014227051/RSICD_optimal)                                           |


## Datasets for other mutimodal composite retrieval
| **Dataset**                     | **Modalities**                    | **Scale**                                | **Link**                                                                                           |
|---------------------------------|-----------------------------------|------------------------------------------|----------------------------------------------------------------------------------------------------|
| HumanML3D                       | Motions, Captions                 | 14K motion sequences, 44K text           | [Link](https://github.com/EricGuo5513/HumanML3D)                                                   |
| KIT-ML                          | Motions, Captions                 | 3K motion sequences, 6K text             | [Link](https://h2t.iar.kit.edu/english/1445.php)                                                   |
| Text2Shape                      | Shapes, Captions                  | 6K chairs, 8K tables, 70K text           | [Link](https://github.com/kchen92/text2shape)                                                      |
| Flickr30k LocNar                | Images, Captions                  | 31K images, 155K texts                   | [Link](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)                            |
| Conceptual Captions             | Images, Captions                  | 3.3M images, 33M texts                   | [Link](https://github.com/google-research-datasets/conceptual-captions)                            |
| Sydney_IV                       | RS Images, Audio Captions         | 613 images, 3K audio descriptions        | [Link](https://github.com/201528014227051/RSICD_optimal)                                           |
| UCM_IV                          | Images, Audio Captions            | 2K images, 10K audio descriptions        | [Link](https://github.com/201528014227051/RSICD_optimal)                                           |
| RSICD_IV                        | Image, Audio Captions             | 11K images, 55K audio descriptions       | [Link](https://github.com/201528014227051/RSICD_optimal)                                           |


# Experimental Results
## Performance comparison on the Fashion-IQ datase((VAL split)
| **Methods**                    | **Image Encoder** | **Dress R@10** | **Dress R@50** | **Shirt R@10** | **Shirt R@50** | **Toptee R@10** | **Toptee R@50** | **Average R@10** | **Average R@50** | **Avg.** |
|--------------------------------|--------------------|----------------|----------------|----------------|----------------|-----------------|-----------------|------------------|------------------|----------|
| ARTEMIS+LSTM <!--\cite{delmas2022ARTEMIS}-->  | ResNet-18         | 25.23          | 48.64          | 20.35          | 43.67          | 23.36           | 46.97           | 22.98            | 46.43            | 34.70    |
| ARTEMIS+BiGRU <!--\cite{delmas2022ARTEMIS}-->  | ResNet-18         | 24.84          | 49.00          | 20.40          | 43.22          | 23.63           | 47.39           | 22.95            | 46.54            | 34.75    |
| JPM(VAL,MSE) <!--\cite{JPM}-->                 | ResNet-18         | 21.27          | 43.12          | 21.88          | 43.30          | 25.81           | 50.27           | 22.98            | 45.59            | 34.29    |
| JPM(VAL,Tri) <!--\cite{JPM}-->                 | ResNet-18         | 21.38          | 45.15          | 22.81          | 45.18          | 27.78           | 51.70           | 23.99            | 47.34            | 35.67    |
| EER <!--\cite{EER}-->                         | ResNet-50         | 30.02          | 55.44          | 25.32          | 49.87          | 33.20           | 60.34           | 29.51            | 55.22            | 42.36    |
| Ranking-aware <!--\cite{chen2023ranking-aware}--> | ResNet-50         | 34.80          | 60.22          | 45.01          | 69.06          | 47.68           | 74.85           | 42.50            | 68.04            | 55.27    |
| CRN <!--\cite{2023-CRN}-->                    | ResNet-50         | 30.20          | 57.15          | 29.17          | 55.03          | 33.70           | 63.91           | 31.02            | 58.70            | 44.86    |
| DWC <!--\cite{huang2023-DWC}-->               | ResNet-50         | 32.67          | 57.96          | 35.53          | 60.11          | 40.13           | 66.09           | 36.11            | 61.39            | 48.75    |
| DATIR <!--\cite{zhao2022-PL4CIR_PLHMQ-twostage}--> | ResNet-50         | 21.90          | 43.80          | 21.90          | 43.70          | 27.20           | 51.60           | 23.70            | 46.40            | 35.05    |
| CoSMo <!--\cite{AMC}-->                      | ResNet-50         | 25.64          | 50.30          | 24.90          | 49.18          | 29.21           | 57.46           | 26.58            | 52.31            | 39.45    |
| FashionVLP <!--\cite{ComqueryFormer}-->       | ResNet-50         | 32.42          | 60.29          | 31.89          | 58.44          | 38.51           | 68.79           | 34.27            | 62.51            | 48.39    |
| CLVC-Net <!--\cite{wen2021-CLVC-NET}-->        | ResNet-50         | 29.85          | 56.47          | 28.75          | 54.76          | 33.50           | 64.00           | 30.70            | 58.41            | 44.56    |
| SAC w/BERT <!--\cite{jandial2021SAC}-->        | ResNet-50         | 26.52          | 51.01          | 28.02          | 51.86          | 32.70           | 61.23           | 29.08            | 54.70            | 41.89    |
| SAC w/ Random Emb. <!--\cite{jandial2021SAC}--> | ResNet-50         | 26.13          | 52.10          | 26.20          | 50.93          | 31.16           | 59.05           | 27.83            | 54.03            | 40.93    |
| DCNet <!--\cite{kim2021-DCNet}-->             | ResNet-50         | 28.95          | 56.07          | 23.95          | 47.30          | 30.44           | 58.29           | 27.78            | 53.89            | 40.83    |
| AMC <!--\cite{AMC}-->                        | ResNet-50         | 31.73          | 59.25          | 30.67          | 59.08          | 36.21           | 66.60           | 32.87            | 61.64            | 47.25    |
| VAL(Lvv) <!--\cite{Chen2020VAL}-->            | ResNet-50         | 21.12          | 42.19          | 21.03          | 43.44          | 25.64           | 49.49           | 22.60            | 45.04            | 33.82    |
| ARTEMIS+LSTM <!--\cite{delmas2022ARTEMIS}-->  | ResNet-50         | 27.34          | 51.71          | 21.05          | 44.18          | 24.91           | 49.87           | 24.43            | 48.59            | 36.51    |
| ARTEMIS+BiGRU <!--\cite{delmas2022ARTEMIS}-->  | ResNet-50         | 27.16          | 52.40          | 21.78          | 43.64          | 29.20           | 54.83           | 26.05            | 50.29            | 38.17    |
| VAL(Lvv + Lvs) <!--\cite{Chen2020VAL}-->      | ResNet-50         | 21.47          | 43.83          | 21.03          | 42.75          | 26.71           | 51.81           | 23.07            | 46.13            | 34.60    |
| VAL(GloVe) <!--\cite{Chen2020VAL}-->          | ResNet-50         | 22.53          | 44.00          | 22.38          | 44.15          | 27.53           | 51.68           | 24.15            | 46.61            | 35.38    |
| AlRet <!--\cite{xu2024-AlRet}-->             | ResNet-50         | 30.19          | 58.80          | 29.39          | 55.69          | 37.66           | 64.97           | 32.36            | 59.76            | 46.12    |
| RTIC <!--\cite{shin2021RTIC}-->              | ResNet-50         | 19.40          | 43.51          | 16.93          | 38.36          | 21.58           | 47.88           | 19.30            | 43.25            | 31.28    |
| RTIC-GCN <!--\cite{shin2021RTIC}-->          | ResNet-50         | 19.79          | 43.55          | 16.95          | 38.67          | 21.97           | 49.11           | 19.57            | 43.78            | 31.68    |
| Uncertainty (CLVC-Net) <!--\cite{chen2024uncertainty}--> | ResNet-50         | 30.60          | 57.46          | 31.54          | 58.29          | 37.37           | 68.41           | 33.17            | 61.39            | 47.28    |
| Uncertainty (CLIP4CIR) <!--\cite{chen2024uncertainty}--> | ResNet-50  | 32.61 | 61.34 | 33.23 | 62.55 | 41.40 | 72.51 | 35.75 | 65.47 | 50.61 |
| CRR <!--\cite{CRR}-->                      | ResNet-101 | 30.41 | 57.11 | 33.67 | 64.48 | 30.73 | 58.02 | 31.60 | 59.87 | 45.74 |
| CIRPLANT <!--\cite{liu2021CIRPLANT}-->    | ResNet-152 | 14.38 | 34.66 | 13.64 | 33.56 | 16.44 | 38.34 | 14.82 | 35.52 | 25.17 |
| CIRPLANT w/OSCAR <!--\cite{liu2021CIRPLANT}--> | ResNet-152 | 17.45 | 40.41 | 17.53 | 38.81 | 21.64 | 45.38 | 18.87 | 41.53 | 30.20 |
| ComqueryFormer <!--\cite{ComqueryFormer}--> | Swin       | 33.86 | 61.08 | 35.57 | 62.19 | 42.07 | 69.30 | 37.17 | 64.19 | 50.68 |
| CRN <!--\cite{2023-CRN}-->                 | Swin       | 30.34 | 57.61 | 29.83 | 55.54 | 33.91 | 64.04 | 31.36 | 59.06 | 45.21 |
| CRN <!--\cite{2023-CRN}-->                 | Swin-L     | 32.67 | 59.30 | 30.27 | 56.97 | 37.74 | 65.94 | 33.56 | 60.74 | 47.15 |
| BLIP4CIR1 <!--\cite{liu2023BLIP4CIR1}-->  | BLIP-B     | 43.78 | 67.38 | 45.04 | 67.47 | 49.62 | 72.62 | 46.15 | 69.15 | 57.65 |
| CASE <!--\cite{levy2023CASE}-->            | BLIP       | 47.44 | 69.36 | 48.48 | 70.23 | 50.18 | 72.24 | 48.79 | 70.68 | 59.74 |
| BLIP4CIR2 <!--\cite{liu2024-BLIP4CIR2}--> | BLIP       | 40.65 | 66.34 | 40.38 | 64.13 | 46.86 | 69.91 | 42.63 | 66.79 | 54.71 |
| BLIP4CIR2+Bi <!--\cite{liu2024-BLIP4CIR2}--> | BLIP       | 42.09 | 67.33 | 41.76 | 64.28 | 46.61 | 70.32 | 43.49 | 67.31 | 55.40 |
| CLIP4CIR3 <!--\cite{CLIP4CIR3}-->          | CLIP       | 39.46 | 64.55 | 44.41 | 65.26 | 47.48 | 70.98 | 43.78 | 66.93 | 55.36 |
| CLIP4CIR <!--\cite{CLIP4CIR2}-->           | CLIP       | 33.81 | 59.40 | 39.99 | 60.45 | 41.41 | 65.37 | 38.32 | 61.74 | 50.03 |
| AlRet <!--\cite{xu2024-AlRet}-->           | CLIP-RN50  | 40.23 | 65.89 | 47.15 | 70.88 | 51.05 | 75.78 | 46.10 | 70.80 | 58.50 |
| Combiner <!--\cite{baldrati2022combiner}--> | CLIP-RN50  | 31.63 | 56.67 | 36.36 | 58.00 | 38.19 | 62.42 | 35.39 | 59.03 | 47.21 |
| DQU-CIR <!--\cite{Wen_2024-DQU-CIR}-->     | CLIP-H     | 57.63 | 78.56 | 62.14 | 80.38 | 66.15 | 85.73 | 61.97 | 81.56 | 71.77 |
| PL4CIR <!--\cite{zhao2022-PL4CIR_PLHMQ-twostage}--> | CLIP-L     | 38.18 | 64.50 | 48.63 | 71.54 | 52.32 | 76.90 | 46.37 | 70.98 | 58.68 |
| TG-CIR <!--\cite{Wen_2023-TG-CIR}-->       | CLIP-B     | 45.22 | 69.66 | 52.60 | 72.52 | 56.14 | 77.10 | 51.32 | 73.09 | 62.21 |
| PL4CIR <!--\cite{zhao2022-PL4CIR_PLHMQ-twostage}--> | CLIP-B     | 33.22 | 59.99 | 46.17 | 68.79 | 46.46 | 73.84 | 41.98 | 67.54 | 54.76 |


## Performance comparison on the Fashion-IQ dataset(original split)
| **Methods**                    | **Image Encoder** | **Dress R@10** | **Dress R@50** | **Shirt R@10** | **Shirt R@50** | **Toptee R@10** | **Toptee R@50** | **Average R@10** | **Average R@50** | **Avg.** |
|--------------------------------|--------------------|----------------|----------------|----------------|----------------|-----------------|-----------------|------------------|------------------|----------|
| ComposeAE <!--\cite{anwaar2021composeAE}--> | ResNet-18  | 10.77 | 28.29 | 9.96  | 25.14 | 12.74 | 30.79 | -     | -     | -     |
| TIRG <!--\cite{vo2018TIRG}-->               | ResNet-18  | 14.87 | 34.66 | 18.26 | 37.89 | 19.08 | 39.62 | 17.40 | 37.39 | 27.40 |
| MAAF <!--\cite{AMC}-->                      | ResNet-50  | 23.80 | 48.60 | 21.30 | 44.20 | 27.90 | 53.60 | 24.30 | 48.80 | 36.60 |
| Leveraging <!--\cite{Leveraging}-->     | ResNet-50   | 19.33 | 43.52 | 14.47 | 35.47 | 19.73 | 44.56 | 17.84 | 41.18 | 29.51 |
| MCR <!--\cite{pang2022MCR}-->           | ResNet-50   | 26.20 | 51.20 | 22.40 | 46.01 | 29.70 | 56.40 | 26.10 | 51.20 | 38.65 |
| MCEM (\(L_CE\)) <!--\cite{MCEM}-->     | ResNet-50   | 30.07 | 56.13 | 23.90 | 47.60 | 30.90 | 57.52 | 28.29 | 53.75 | 41.02 |
| MCEM (\(L_FCE\)) <!--\cite{MCEM}-->    | ResNet-50   | 31.50 | 58.41 | 25.01 | 49.73 | 32.77 | 61.02 | 29.76 | 56.39 | 43.07 |
| MCEM (\(L_AFCE\)) <!--\cite{MCEM}-->   | ResNet-50   | 33.23 | 59.16 | 26.15 | 50.87 | 33.83 | 61.40 | 31.07 | 57.14 | 44.11 |
| AlRet <!--\cite{xu2024-AlRet}-->        | ResNet-50   | 27.34 | 53.42 | 21.30 | 43.08 | 29.07 | 54.21 | 25.86 | 50.17 | 38.02 |
| MCEM (\(L_AFCE\) w/ BERT) <!--\cite{MCEM}--> | ResNet-50   | 32.11 | 59.21 | 27.28 | 52.01 | 33.96 | 62.30 | 31.12 | 57.84 | 44.48 |
| JVSM <!--\cite{chen2020JVSM}-->         | MobileNet-v1 | 10.70 | 25.90 | 12.00 | 27.10 | 13.00 | 26.90 | 11.90 | 26.63 | 19.27 |
| FashionIQ (Dialog Turn 1) <!--\cite{wu2020fashioniq}--> | EfficientNet-b | 12.45 | 35.21 | 11.05 | 28.99 | 11.24 | 30.45 | 11.58 | 31.55 | 21.57 |
| FashionIQ (Dialog Turn 5) <!--\cite{wu2020fashioniq}--> | EfficientNet-b | 41.35 | 73.63 | 33.91 | 63.42 | 33.52 | 63.85 | 36.26 | 66.97 | 51.61 |
| AACL <!--\cite{tian2022AACL}-->         | Swin        | 29.89 | 55.85 | 24.82 | 48.85 | 30.88 | 56.85 | 28.53 | 53.85 | 41.19 |
| ComqueryFormer <!--\cite{ComqueryFormer}--> | Swin      | 28.85 | 55.38 | 25.64 | 50.22 | 33.61 | 60.48 | 29.37 | 55.36 | 42.36 |
| AlRet <!--\cite{xu2024-AlRet}-->        | CLIP        | 35.75 | 60.56 | 37.02 | 60.55 | 42.25 | 67.52 | 38.30 | 62.82 | 50.56 |
| MCEM (\(L_AFCE\)) <!--\cite{MCEM}-->   | CLIP        | 33.98 | 59.96 | 40.15 | 62.76 | 43.75 | 67.70 | 39.29 | 63.47 | 51.38 |
| SPN (TG-CIR) <!--\cite{feng2024data_generation-SPN}--> | CLIP | 36.84 | 60.83 | 41.85 | 63.89 | 45.59 | 68.79 | 41.43 | 64.50 | 52.97 |
| SPN (CLIP4CIR) <!--\cite{feng2024data_generation-SPN}--> | CLIP | 38.82 | 62.92 | 45.83 | 66.44 | 48.80 | 71.29 | 44.48 | 66.88 | 55.68 |
| PL4CIR <!--\cite{zhao2022-PL4CIR_PLHMQ-twostage}--> | CLIP-B | 29.00 | 53.94 | 35.43 | 58.88 | 39.16 | 64.56 | 34.53 | 59.13 | 46.83 |
| FAME-ViL <!--\cite{han2023FAMEvil}-->          | CLIP-B | 42.19 | 67.38 | 47.64 | 68.79 | 50.69 | 73.07 | 46.84 | 69.75 | 58.30 |
| PALAVRA <!--\cite{cohen2022-PALAVRA}-->        | CLIP-B | 17.25 | 35.94 | 21.49 | 37.05 | 20.55 | 38.76 | 19.76 | 37.25 | 28.51 |
| MagicLens-B <!--\cite{zhang2024magiclens}-->  | CLIP-B | 21.50 | 41.30 | 27.30 | 48.80 | 30.20 | 52.30 | 26.30 | 47.40 | 36.85 |
| SEARLE <!--\cite{Baldrati2023SEARLE}-->        | CLIP-B | 18.54 | 39.51 | 24.44 | 41.61 | 25.70 | 46.46 | 22.89 | 42.53 | 32.71 |
| CIReVL <!--\cite{karthik2024-CIReVL}-->        | CLIP-B | 25.29 | 46.36 | 28.36 | 47.84 | 31.21 | 53.85 | 28.29 | 49.35 | 38.82 |
| SEARLE-OTI <!--\cite{Baldrati2023SEARLE}-->    | CLIP-B | 17.85 | 39.91 | 25.37 | 41.32 | 24.12 | 45.79 | 22.44 | 42.34 | 32.39 |
| PLI <!--\cite{chen2023-PLI}-->                | CLIP-B | 25.71 | 47.81 | 33.36 | 53.47 | 34.87 | 58.44 | 31.31 | 53.24 | 42.28 |
| PL4CIR <!--\cite{zhao2022-PL4CIR_PLHMQ-twostage}--> | CLIP-L | 33.60 | 58.90 | 39.45 | 61.78 | 43.96 | 68.33 | 39.02 | 63.00 | 51.01 |
| SEARLE-XL <!--\cite{Baldrati2023SEARLE}-->     | CLIP-L | 20.48 | 43.13 | 26.89 | 45.58 | 29.32 | 49.97 | 25.56 | 46.23 | 35.90 |
| SEARLE-XL-OTI <!--\cite{Baldrati2023SEARLE}--> | CLIP-L | 21.57 | 44.47 | 30.37 | 47.49 | 30.90 | 51.76 | 27.61 | 47.90 | 37.76 |
| Context-I2W <!--\cite{tang2023contexti2w}-->  | CLIP-L | 23.10 | 45.30 | 29.70 | 48.60 | 30.60 | 52.90 | 27.80 | 48.90 | 38.35 |
| CompoDiff (with SynthTriplets18M) <!--\cite{gu2024compodiff}--> | CLIP-L | 32.24 | 46.27 | 37.69 | 49.08 | 38.12 | 50.57 | 36.02 | 48.64 | 42.33 |
| CompoDiff (with SynthTriplets18M) <!--\cite{gu2024compodiff}--> | CLIP-L | 37.78 | 49.10 | 41.31 | 55.17 | 44.26 | 56.41 | 39.02 | 51.71 | 46.85 |
| Pic2Word <!--\cite{saito2023pic2word}-->       | CLIP-L | 20.00 | 40.20 | 26.20 | 43.60 | 27.90 | 47.40 | 24.70 | 43.70 | 34.20 |
| PLI <!--\cite{chen2023-PLI}-->                | CLIP-L | 28.11 | 51.12 | 38.63 | 58.51 | 39.42 | 62.68 | 35.39 | 57.44 | 46.42 |
| KEDs <!--\cite{suo2024KEDs}-->                | CLIP-L | 21.70 | 43.80 | 28.90 | 48.00 | 29.90 | 51.90 | 26.80 | 47.90 | 37.35 |
| CIReVL <!--\cite{karthik2024-CIReVL}-->        | CLIP-L | 24.79 | 44.76 | 29.49 | 47.40 | 31.36 | 53.65 | 28.55 | 48.57 | 38.56 |
| LinCIR <!--\cite{gu2024LinCIR}-->              | CLIP-L | 20.92 | 42.44 | 29.10 | 46.81 | 28.81 | 50.18 | 26.28 | 46.49 | 36.39 |
| MagicLens-L <!--\cite{zhang2024magiclens}-->  | CLIP-L | 25.50 | 46.10 | 32.70 | 53.80 | 34.00 | 57.70 | 30.70 | 52.50 | 41.60 |
| LinCIR <!--\cite{gu2024LinCIR}-->              | CLIP-H | 29.80 | 52.11 | 36.90 | 57.75 | 42.07 | 62.52 | 36.26 | 57.46 | 46.86 |
| DQU-CIR <!--\cite{Wen_2024-DQU-CIR}-->        | CLIP-H | 51.90 | 74.37 | 53.57 | 73.21 | 58.48 | 79.23 | 54.65 | 75.60 | 65.13 |
| LinCIR <!--\cite{gu2024LinCIR}-->             | CLIP-G | 38.08 | 60.88 | 46.76 | 65.11 | 50.48 | 71.09 | 45.11 | 65.69 | 55.40 |
| CIReVL <!--\cite{karthik2024-CIReVL}-->       | CLIP-G | 27.07 | 49.53 | 33.71 | 51.42 | 35.80 | 56.14 | 32.19 | 52.36 | 42.28 |
| MagicLens-B <!--\cite{zhang2024magiclens}--> | CoCa-B | 29.00 | 48.90 | 36.50 | 55.50 | 40.20 | 61.90 | 35.20 | 55.40 | 45.30 |
| MagicLens-L <!--\cite{zhang2024magiclens}--> | CoCa-L | 32.30 | 52.70 | 40.50 | 59.20 | 41.40 | 63.00 | 38.00 | 58.20 | 48.10 |
| SPN (BLIP4CIR1) <!--\cite{feng2024data_generation-SPN}--> | BLIP    | 44.52 | 67.13 | 45.68 | 67.96 | 50.74 | 73.79 | 46.98 | 69.63 | 58.30 |
| PLI <!--\cite{chen2023-PLI}-->                           | BLIP-B | 28.62 | 50.78 | 38.09 | 57.79 | 40.92 | 62.68 | 35.88 | 57.08 | 46.48 |
| SPN (SPRC) <!--\cite{feng2024data_generation-SPN}-->     | BLIP-2 | 50.57 | 74.12 | 57.70 | 75.27 | 60.84 | 79.96 | 56.37 | 76.45 | 66.41 |
| CurlingNet <!--\cite{yu2020Curlingnet}-->                | -      | 24.44 | 47.69 | 18.59 | 40.57 | 25.19 | 49.66 | 22.74 | 45.97 | 34.36 |

## Performance comparison on the Fashion200k dataset
| **Methods**                    | **Image Encoder** | **R@1** | **R@10** | **R@50** |
|--------------------------------|-------------------|---------|----------|----------|
| TIRG <!--\cite{vo2018TIRG}-->              | ResNet-18 | 14.10 | 42.50 | 63.80 |
| ComposeAE <!--\cite{anwaar2021composeAE}--> | ResNet-18 | 22.80 | 55.30 | 73.40 |
| HCL <!--\cite{HCL}-->                          | ResNet-18 | 23.48 | 54.03 | 73.71 |
| CoSMo <!--\cite{Lee2021CoSMo}-->              | ResNet-18 | 23.30 | 50.40 | 69.30 |
| JPM(TIRG,MSE) <!--\cite{JPM}-->                | ResNet-18 | 19.80 | 46.50 | 66.60 |
| JPM(TIRG,Tri) <!--\cite{JPM}-->                | ResNet-18 | 17.70 | 44.70 | 64.50 |
| ARTEMIS <!--\cite{delmas2022ARTEMIS}-->        | ResNet-18 | 21.50 | 51.10 | 70.50 |
| GA(TIRG-BERT) <!--\cite{huang2022-GA-data-augmentation}--> | ResNet-18 | 31.40 | 54.10 | 77.60 |
| LGLI <!--\cite{huang2023-LGLI}-->              | ResNet-18 | 26.50 | 58.60 | 75.60 |
| AlRet <!--\cite{xu2024-AlRet}-->               | ResNet-18 | 24.42 | 53.93 | 73.25 |
| FashionVLP <!--\cite{ComqueryFormer}-->        | ResNet-18 | -     | 49.90 | 70.50 |
| CLVC-Net <!--\cite{wen2021-CLVC-NET}-->        | ResNet-50 | 22.60 | 53.00 | 72.20 |
| Uncertainty <!--\cite{chen2024uncertainty}-->  | ResNet-50 | 21.80 | 52.10 | 70.20 |
| MCR <!--\cite{ComqueryFormer}-->               | ResNet-50 | 49.40 | 69.40 | 59.40 |
| CRN <!--\cite{2023-CRN}-->                      | ResNet-50 | -     | 53.10 | 73.00 |
| EER w/ Random Emb. <!--\cite{EER}-->           | ResNet-50 | -     | 51.09 | 70.23 |
| EER w/ GloVe <!--\cite{EER}-->                  | ResNet-50 | -     | 50.88 | 73.40 |
| DWC <!--\cite{huang2023-DWC}-->                | ResNet-50 | 36.49 | 63.58 | 79.02 |
| JGAN <!--\cite{JGAN}-->                         | ResNet-101 | 17.34 | 45.28 | 65.65 |
| CRR <!--\cite{CRR}-->                           | ResNet-101 | 24.85 | 56.41 | 73.56 |
| GSCMR <!--\cite{2022-GSCMR}-->                  | ResNet-101 | 21.57 | 52.84 | 70.12 |
| VAL(GloVe) <!--\cite{Chen2020VAL}-->            | MobileNet  | 22.90 | 50.80 | 73.30 |
| VAL(Lvv+Lvs) <!--\cite{Chen2020VAL}-->          | MobileNet  | 21.50 | 53.80 | 72.70 |
| DATIR <!--\cite{ComqueryFormer}-->              | MobileNet  | 21.50 | 48.80 | 71.60 |
| VAL(Lvv) <!--\cite{Chen2020VAL}-->              | MobileNet  | 21.20 | 49.00 | 68.80 |
| JVSM <!--\cite{chen2020JVSM}-->                 | MobileNet-v1 | 19.00 | 52.10 | 70.00 |
| TIS <!--\cite{TIS}-->                           | MobileNet-v1 | 17.76 | 47.54 | 68.02 |
| DCNet <!--\cite{kim2021-DCNet}-->               | MobileNet-v1 | -     | 46.89 | 67.56 |
| TIS <!--\cite{TIS}-->                           | Inception-v3 | 16.25 | 44.14 | 65.02 |
| LBF(big) <!--\cite{hosseinzadeh2020-locally-LBF}--> | Faster-RCNN | 17.78 | 48.35 | 68.50 |
| LBF(small) <!--\cite{hosseinzadeh2020-locally-LBF}--> | Faster-RCNN | 16.26 | 46.90 | 71.73 |
| ProVLA <!--\cite{Hu_2023_ICCV-ProVLA}-->        | Swin         | 21.70 | 53.70 | 74.60 |
| CRN <!--\cite{2023-CRN}-->                      | Swin         | -     | 53.30 | 73.30 |
| ComqueryFormer <!--\cite{ComqueryFormer}-->    | Swin         | -     | 52.20 | 72.20 |
| AACL <!--\cite{tian2022AACL}-->                 | Swin         | 19.64 | 58.85 | 78.86 |
| CRN <!--\cite{2023-CRN}-->                      | Swin-L       | -     | 53.50 | 74.50 |
| DQU-CIR <!--\cite{Wen_2024-DQU-CIR}-->          | CLIP-H       | 36.80 | 67.90 | 87.80 |

## Performance comparison on the MIT-States dataset
| **Methods**                    | **Image Encoder** | **R@1** | **R@10** | **R@50** | **Average** |
|--------------------------------|-------------------|---------|----------|----------|-------------|
| TIRG <!--\cite{vo2018TIRG}-->                       | ResNet-18 | 12.20 | 31.90 | 43.10 | 29.10 |
| ComposeAE <!--\cite{anwaar2021composeAE}-->       | ResNet-18 | 13.90 | 35.30 | 47.90 | 32.37 |
| HCL <!--\cite{HCL}-->                               | ResNet-18 | 15.22 | 35.95 | 46.71 | 32.63 |
| GA(TIRG) <!--\cite{huang2022-GA-data-augmentation}--> | ResNet-18 | 13.60 | 32.40 | 43.20 | 29.70 |
| GA(TIRG-BERT) <!--\cite{huang2022-GA-data-augmentation}--> | ResNet-18 | 15.40 | 36.30 | 47.70 | 33.20 |
| GA(ComposeAE) <!--\cite{huang2022-GA-data-augmentation}--> | ResNet-18 | 14.60 | 37.00 | 47.90 | 33.20 |
| LGLI <!--\cite{huang2023-LGLI}-->                   | ResNet-18 | 14.90 | 36.40 | 47.70 | 33.00 |
| MAAF <!--\cite{dodds2020MAAF}-->                     | ResNet-50 | 12.70 | 32.60 | 44.80 | -     |
| MCR <!--\cite{CRR}-->                                | ResNet-50 | 14.30 | 35.36 | 47.12 | 32.26 |
| CRR <!--\cite{CRR}-->                                | ResNet-101 | 17.71 | 37.16 | 47.83 | 34.23 |
| JGAN <!--\cite{JGAN}-->                              | ResNet-101 | 14.27 | 33.21 | 45.34 | 29.10 |
| GSCMR <!--\cite{2022-GSCMR}-->                       | ResNet-101 | 17.28 | -     | 36.45 | -     |
| TIS <!--\cite{TIS}-->                                | Inception-v3 | 13.13 | 31.94 | 43.32 | 29.46 |
| LBF(big) <!--\cite{hosseinzadeh2020-locally-LBF}--> | Faster-RCNN | 14.72 | 35.30 | 46.56 | 96.58 |
| LBF(small) <!--\cite{hosseinzadeh2020-locally-LBF}--> | Faster-RCNN | 14.29 | -     | 34.67 | 46.06 |

## Performance comparison on the CSS dataset
| **Methods**                    | **Image Encoder** | **R@1(3D-to-3D)** | **R@1(2D-to-3D** |
|--------------------------------|-------------------|-------------------|------------------|
| TIRG <!--\cite{JGAN}-->                               | ResNet-18 | 73.70 | 46.60 |
| HCL <!--\cite{HCL}-->                                 | ResNet-18 | 81.59 | 58.65 |
| GA(TIRG) <!--\cite{huang2022-GA-data-augmentation}--> | ResNet-18 | 91.20 | -     |
| TIRG+JPM(MSE) <!--\cite{JPM}-->                       | ResNet-18 | 83.80 | -     |
| TIRG+JPM(Tri) <!--\cite{JPM}-->                       | ResNet-18 | 83.20 | -     |
| LGLI <!--\cite{huang2023-LGLI}-->                     | ResNet-18 | 93.30 | -     |
| MAAF <!--\cite{dodds2020MAAF}-->                      | ResNet-50 | 87.80 | -     |
| CRR <!--\cite{CRR}-->                                 | ResNet-101 | 85.84 | -     |
| JGAN <!--\cite{JGAN}-->                               | ResNet-101 | 76.07 | 48.85 |
| GSCMR <!--\cite{2022-GSCMR}-->                        | ResNet-101 | 81.81 | 58.74 |
| TIS <!--\cite{TIS}-->                                 | Inception-v3 | 76.64 | 48.02 |
| LBF(big) <!--\cite{hosseinzadeh2020-locally-LBF}-->  | Faster-RCNN | 79.20 | 55.69 |
| LBF(small) <!--\cite{hosseinzadeh2020-locally-LBF}--> | Faster-RCNN | 67.26 | 50.31 |

## Performance comparison on the Shoes dataset
| **Methods**                    | **Image Encoder** | **R@1** | **R@10** | **R@50** | **Average** |
|--------------------------------|-------------------|---------|----------|----------|-------------|
| ComposeAE <!--\cite{anwaar2021composeAE}-->       | ResNet-18  | 31.25 | 60.30 | -   | -    | % shin2021RTIC |
| TIRG <!--\cite{vo2018TIRG}-->                       | ResNet-50  | 12.60 | 45.45 | 69.39 | 42.48 | % Lee2021CoSMo |
| VAL(Lvv) <!--\cite{Chen2020VAL}-->                  | ResNet-50  | 16.49 | 49.12 | 73.53 | 46.38 |
| VAL(Lvv + Lvs) <!--\cite{Chen2020VAL}-->            | ResNet-50  | 16.98 | 49.83 | 73.91 | 46.91 |
| VAL(GloVe) <!--\cite{Chen2020VAL}-->                | ResNet-50  | 17.18 | 51.52 | 75.83 | 48.18 |
| CoSMo <!--\cite{Lee2021CoSMo}-->                    | ResNet-50  | 16.72 | 48.36 | 75.64 | 46.91 |
| CLVC-Net <!--\cite{wen2021-CLVC-NET}-->             | ResNet-50  | 17.64 | 54.39 | 79.47 | 50.50 |
| DCNet <!--\cite{kim2021-DCNet}-->                   | ResNet-50  | -     | 53.82 | 79.33 | -     |
| SAC w/BERT <!--\cite{jandial2021SAC}-->             | ResNet-50  | 18.50 | 51.73 | 77.28 | 49.17 |
| SAC w/Random Emb. <!--\cite{jandial2021SAC}-->      | ResNet-50  | 18.11 | 52.41 | 75.42 | 48.64 |
| ARTEMIS+LSTM <!--\cite{delmas2022ARTEMIS}-->         | ResNet-50  | 17.60 | 51.05 | 76.85 | 48.50 |
| ARTEMIS+BiGRU <!--\cite{delmas2022ARTEMIS}-->        | ResNet-50  | 18.72 | 53.11 | 79.31 | 50.38 |
| AMC <!--\cite{AMC}-->                               | ResNet-50  | 19.99 | 56.89 | 79.27 | 52.05 |
| DATIR <!--\cite{zhao2022-PL4CIR_PLHMQ-twostage}-->  | ResNet-50  | 17.20 | 51.10 | 75.60 | 47.97 |
| MCR <!--\cite{CRR}-->                               | ResNet-50  | 17.85 | 50.95 | 77.24 | 48.68 |
| EER <!--\cite{EER}-->                               | ResNet-50  | 20.05 | 56.02 | 79.94 | 52.00 |
| CRN <!--\cite{2023-CRN}-->                           | ResNet-50  | 17.19 | 53.88 | 79.12 | 50.06 |
| Uncertainty <!--\cite{chen2024uncertainty}-->       | ResNet-50  | 18.41 | 53.63 | 79.84 | 50.63 |
| FashionVLP <!--\cite{Goenka_2022_FashionVLP}-->     | ResNet-50  | -     | 49.08 | 77.32 | -     |
| DWC <!--\cite{huang2023-DWC}-->                      | ResNet-50  | 18.94 | 55.55 | 80.19 | 51.56 |
| MCEM(\(L_CE\)) <!--\cite{MCEM}-->                   | ResNet-50  | 15.17 | 49.33 | 73.78 | 46.09 |
| MCEM(\(L_FCE\)) <!--\cite{MCEM}-->                  | ResNet-50  | 18.13 | 54.31 | 78.65 | 50.36 |
| MCEM(\(L_AFCE\)) <!--\cite{MCEM}-->                 | ResNet-50  | 19.10 | 55.37 | 79.57 | 51.35 |
| AlRet <!--\cite{xu2024-AlRet}-->                     | ResNet-50  | 18.13 | 53.98 | 78.81 | 50.31 |
| RTIC <!--\cite{shin2021RTIC}-->                      | ResNet-50  | 43.66 | 72.11 | -     | -     |
| RTIC-GCN <!--\cite{shin2021RTIC}-->                  | ResNet-50  | 43.38 | 72.09 | -     | -     |
| CRR <!--\cite{CRR}-->                               | ResNet-101 | 18.41 | 56.38 | 79.92 | 51.57 |
| CRN <!--\cite{2023-CRN}-->                           | Swin       | 17.32 | 54.15 | 79.34 | 50.27 |
| ProVLA <!--\cite{Hu_2023_ICCV-ProVLA}-->             | Swin       | 19.20 | 56.20 | 73.30 | 49.57 |
| CRN <!--\cite{2023-CRN}-->                           | Swin-L     | 18.92 | 54.55 | 80.04 | 51.17 |
| AlRet <!--\cite{xu2024-AlRet}-->                     | CLIP       | 21.02 | 55.72 | 80.77 | 52.50 |
| PL4CIR <!--\cite{zhao2022-PL4CIR_PLHMQ-twostage}-->  | CLIP-L     | 22.88 | 58.83 | 84.16 | 55.29 |
| PL4CIR <!--\cite{zhao2022-PL4CIR_PLHMQ-twostage}-->  | CLIP-B     | 19.53 | 55.65 | 80.58 | 51.92 |
| TG-CIR <!--\cite{Wen_2023-TG-CIR}-->                 | CLIP-B     | 25.89 | 63.20 | 85.07 | 58.05 |
| DQU-CIR <!--\cite{Wen_2024-DQU-CIR}-->                | CLIP-H     | 31.47 | 69.19 | 88.52 | 63.06 |

## Performance comparison on the CIRR dataset
| **Methods**                    | **Image Encoder** | **R@1** | **R@5** | **R@10** | **R@50** |
|--------------------------------|-------------------|---------|---------|----------|----------|
| ComposeAE <!--\cite{shin2021RTIC}-->                | ResNet-18  | -     | 29.60 | 59.82 | -     |
| MCEM(\(L_CE\)) <!--\cite{MCEM}-->                   | ResNet-18  | 14.26 | 40.46 | 55.61 | 85.66 |
| MCEM(\(L_FCE\)) <!--\cite{MCEM}-->                  | ResNet-18  | 16.12 | 43.92 | 58.87 | 86.85 |
| MCEM(\(L_AFCE\)) <!--\cite{MCEM}-->                 | ResNet-18  | 17.48 | 46.13 | 62.17 | 88.91 |
| Ranking-aware <!--\cite{chen2023ranking-aware}-->    | ResNet-50  | 32.24 | 66.63 | 79.23 | 96.43 |
| SAC w/BERT <!--\cite{jandial2021SAC}-->              | ResNet-50  | -     | 19.56 | 45.24 | -     |
| SAC w/Random Emb. <!--\cite{jandial2021SAC}-->       | ResNet-50  | -     | 20.34 | 44.94 | -     |
| ARTEMIS+BiGRU <!--\cite{delmas2022ARTEMIS}-->        | ResNet-152 | 16.96 | 46.10 | 61.31 | 87.73 |
| CIRPLANT <!--\cite{liu2021CIRPLANT}-->                     | ResNet-152 | 15.18 | 43.36 | 60.48 | 87.64 |
| CIRPLANT w/ OSCAR <!--\cite{liu2021CIRPLANT}-->             | ResNet-152 | 19.55 | 52.55 | 68.39 | 92.38 |
| CASE <!--\cite{levy2023CASE}-->                             | ViT        | 48.00 | 79.11 | 87.25 | 97.57 |
| ComqueryFormer <!--\cite{ComqueryFormer}-->                 | Swin       | 25.76 | 61.76 | 75.90 | 95.13 |
| CLIP4CIR <!--\cite{baldrati2022-CLIP4CIR}-->                | CLIP       | 38.53 | 69.98 | 81.86 | 95.93 |
| CLIP4CIR3 <!--\cite{CLIP4CIR3}-->                           | CLIP       | 44.82 | 77.04 | 86.65 | 97.90 |
| SPN(TG-CIR) <!--\cite{feng2024data_generation-SPN}-->      | CLIP       | 47.28 | 79.13 | 87.98 | 97.54 |
| SPN(CLIP4CIR) <!--\cite{feng2024data_generation-SPN}-->    | CLIP       | 45.33 | 78.07 | 87.61 | 98.17 |
| Combiner <!--\cite{baldrati2022combiner}-->                 | CLIP       | 33.59 | 65.35 | 77.35 | 95.21 |
| MCEM(\(L_AFCE\)) <!--\cite{MCEM}-->                        | CLIP       | 39.80 | 74.24 | 85.71 | 97.23 |
| TG-CIR <!--\cite{Wen_2023-TG-CIR}-->                        | CLIP-B     | 45.25 | 78.29 | 87.16 | 97.30 |
| CIReVL <!--\cite{karthik2024-CIReVL}-->                     | CLIP-B     | 23.94 | 52.51 | 66.00 | 86.95 |
| SEARLE-OTI <!--\cite{Baldrati2023SEARLE}-->                 | CLIP-B     | 24.27 | 53.25 | 66.10 | 88.84 |
| SEARLE <!--\cite{Baldrati2023SEARLE}-->                     | CLIP-B     | 24.00 | 53.42 | 66.82 | 89.78 |
| PLI <!--\cite{chen2023-PLI}-->                              | CLIP-B     | 18.80 | 46.07 | 60.75 | 86.41 |
| SEARLE-XL <!--\cite{Baldrati2023SEARLE}-->                  | CLIP-L     | 24.24 | 52.48 | 66.29 | 88.84 |
| SEARLE-XL-OTI <!--\cite{Baldrati2023SEARLE}-->              | CLIP-L     | 24.87 | 52.31 | 66.29 | 88.58 |
| CIReVL <!--\cite{karthik2024-CIReVL}-->                     | CLIP-L     | 24.55 | 52.31 | 64.92 | 86.34 |
| Context-I2W <!--\cite{tang2023contexti2w}-->                | CLIP-L     | 25.60 | 55.10 | 68.50 | 89.80 |
| Pic2Word <!--\cite{saito2023pic2word}-->                    | CLIP-L     | 23.90 | 51.70 | 65.30 | 87.80 |
| CompoDiff(with SynthTriplets18M) <!--\cite{gu2024compodiff}--> | CLIP-L | 18.24 | 53.14 | 70.82 | 90.25 |
| LinCIR <!--\cite{gu2024LinCIR}-->                          | CLIP-L    | 25.04 | 53.25 | 66.68 | -     |
| PLI <!--\cite{chen2023-PLI}-->                            | CLIP-L    | 25.52 | 54.58 | 67.59 | 88.70 |
| KEDs <!--\cite{suo2024KEDs}-->                            | CLIP-L    | 26.40 | 54.80 | 67.20 | 89.20 |
| CIReVL <!--\cite{karthik2024-CIReVL}-->                   | CLIP-G    | 34.65 | 64.29 | 75.06 | 91.66 |
| LinCIR <!--\cite{gu2024LinCIR}-->                          | CLIP-G    | 35.25 | 64.72 | 76.05 | -     |
| CompoDiff(with SynthTriplets18M) <!--\cite{gu2024compodiff}--> | CLIP-G | 26.71 | 55.14 | 74.52 | 92.01 |
| LinCIR <!--\cite{gu2024LinCIR}-->                          | CLIP-H    | 33.83 | 63.52 | 75.35 | -     |
| DQU-CIR <!--\cite{Wen_2024-DQU-CIR}-->                     | CLIP-H    | 46.22 | 78.17 | 87.64 | 97.81 |
| PLI <!--\cite{chen2023-PLI}-->                            | BLIP      | 27.23 | 58.87 | 71.40 | 91.25 |
| BLIP4CIR2 <!--\cite{liu2024-BLIP4CIR2}-->                    | BLIP      | 40.17 | 71.81 | 83.18 | 95.69 |
| BLIP4CIR2+Bi <!--\cite{liu2024-BLIP4CIR2}-->                | BLIP      | 40.15 | 73.08 | 83.88 | 96.27 |
| SPN(BLIP4CIR1) <!--\cite{feng2024data_generation-SPN}-->    | BLIP      | 46.43 | 77.64 | 87.01 | 97.06 |
| SPN(SPRC) <!--\cite{feng2024data_generation-SPN}-->          | BLIP-2    | 55.06 | 83.83 | 90.87 | 98.29 |
| BLIP4CIR1 <!--\cite{liu2023BLIP4CIR1}-->                     | BLIP-B    | 46.83 | 78.59 | 88.04 | 97.08 |



[NOTE] **If you have any questions, please don't hesitate to contact [us](mailto:fxhuang1995@gmail.com).** 
