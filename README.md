# <p align=center> Multimodal Composite Editing and Retrieval </p> # 

:fire::fire: This is a collection of awesome articles about multimodal composite editing and retrieval:fire::fire:

[NEWS.20240909] **The related survey [paper](http://arxiv.org/abs/2409.05405) has been released.**

[NOTE] **If you have any questions, please don't hesitate to [contact us](fxhuang1995@gmail.com).** 

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


# Papers and related codes
## Image-text composite editing


**[SIGIR, 2024] Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval**  
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



**[ICASSP, 2019] Bilinear Representation for Language-based Image Editing Using Conditional Generative Adversarial Networks**  
*Xiaofeng Mao, Yuefeng Chen, Yuhong Li, Tao Xiong, Yuan He, Hui Xue*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683008)] [[GitHub](https://github.com/vtddggg/BilinearGAN_for_LBIE)]

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

**[ICCV, 2017] Semantic image synthesis via adversarial learning**  
*Hao Dong, Simiao Yu, Chao Wu, Yike Guo*  
[[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dong_Semantic_Image_Synthesis_ICCV_2017_paper.pdf)] [[GitHub](https://github.com/woozzu/dong_iccv_2017)]





## Image-text composite retrieval

# 2024
**[AAAI, 2024] Dynamic weighted combiner for mixed-modal image retrieval** \
*Fuxiang Huang, Lei Zhang, Xiaowei Fu, Suqi Song* \
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28004/28023)] [[GitHub](https://github.com/fuxianghuang1/DWC)]

**[ICMR, 2024] Enhancing Interactive Image Retrieval With Query Rewriting Using Large Language Models and Vision Language Models**  
*Hongyi Zhu, Jia-Hong Huang, Stevan Rudinac, Evangelos Kanoulas*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3652583.3658032)] [[GitHub](https://github.com/s04240051/Multimodal-Conversational-Retrieval-System)]

**[arXiv, 2024] Improving Composed Image Retrieval via Contrastive Learning with Scaling Positives and Negatives**  
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

#2023

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


#2022

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

#2021

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



#2020

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

#2018

**[CVPR, 2018] Language-based image editing with recurrent attentive models**  
*Jianbo Chen, Yelong Shen, Jianfeng Gao, Jingjing Liu, Xiaodong Liu*  
[[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Language-Based_Image_Editing_CVPR_2018_paper.pdf)]

**[NeurIPS, 2018] Dialog-based interactive image retrieval**  
*Xiaoxiao Guo, Hui Wu, Yu Cheng, Steven Rennie, Gerald Tesauro, Rogerio Feris*  
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/a01a0380ca3c61428c26a231f0e49a09-Paper.pdf)] [[GitHub](https://github.com/XiaoxiaoGuo/fashion-retrieval)]

#2017

**[ICCV, 2017] Automatic spatially-aware fashion concept discovery**  
*Xintong Han, Zuxuan Wu, Phoenix X Huang, Xiao Zhang, Menglong Zhu, Yuan Li, Yang Zhao, Larry S Davis*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8237425)]

**[ICCV, 2017] Be your own prada: Fashion synthesis with structural coherence**  
*Shizhan Zhu, Raquel Urtasun, Sanja Fidler, Dahua Lin, Chen Change Loy*  
[[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Be_Your_Own_ICCV_2017_paper.pdf)] [[GitHub](https://github.com/zhusz/ICCV17-fashionGAN)]

## Other mutimodal composite retrieval

# 2024

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

#2023

**[CVPR, 2023] SceneTrilogy: On Human Scene-Sketch and its Complementarity with Photo and Text**  
*Pinaki Nath Chowdhury, Ayan Kumar Bhunia, Aneeshan Sain, Subhadeep Koley, Tao Xiang, Yi-Zhe Song*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chowdhury_SceneTrilogy_On_Human_Scene-Sketch_and_Its_Complementarity_With_Photo_and_CVPR_2023_paper.pdf)]

#2022

**[ECCV, 2022] A sketch is worth a thousand words: Image retrieval with text and sketch**  
*Patsorn Sangkloy, Wittawat Jitkrittum, Diyi Yang, James Hays*  
[[Paper](https://dl.acm.org/doi/abs/10.1007/978-3-031-19839-7_15)]

**[ECCV, 2022] Motionclip: Exposing human motion generation to clip space**  
*Guy Tevet, Brian Gordon, Amir Hertz, Amit H Bermano, Daniel Cohen-Or*  
[[Paper](https://arxiv.org/pdf/2203.08063)] [[GitHub](https://github.com/GuyTevet/MotionCLIP)]

**[IEEE, 2022] Multimodal Fusion Remote Sensing Image–Audio Retrieval**  
*Rui Yang, Shuang Wang, Yingzhi Sun, Huan Zhang, Yu Liao, Yu Gu, Biao Hou, Licheng Jiao*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9847567)]

#2021

**[CVPR, 2021] Connecting what to say with where to look by modeling human attention traces**  
*Zihang Meng, Licheng Yu, Ning Zhang, Tamara L Berg, Babak Damavandi, Vikas Singh, Amy Bearman*  
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Meng_Connecting_What_To_Say_With_Where_To_Look_by_Modeling_CVPR_2021_paper.pdf)] [[GitHub](https://github.com/facebookresearch/connect-caption-and-trace)]

**[ICCV, 2021] Telling the what while pointing to the where: Multimodal queries for image retrieval**  
*Soravit Changpinyo, Jordi Pont-Tuset, Vittorio Ferrari, Radu Soricut*  
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Changpinyo_Telling_the_What_While_Pointing_to_the_Where_Multimodal_Queries_ICCV_2021_paper.pdf)]

# 2020

**[arXiv, 2020] A Feature Analysis for Multimodal News Retrieval**  
*Golsa Tahmasebzadeh, Sherzod Hakimov, Eric Müller-Budack, Ralph Ewerth*  
[[Paper](https://arxiv.org/abs/2007.06390)] [[GitHub](https://github.com/cleopatra-itn/multimodal-news-retrieval)]

#2019

**[Multimedia Tools and Applications, 2019] Efficient and interactive spatial-semantic image retrieval**  
*Ryosuke Furuta, Naoto Inoue, Toshihiko Yamasaki*  
[[Paper](https://link.springer.com/content/pdf/10.1007/s11042-018-7148-1.pdf)]

**[arXiv, 2019] Query by Semantic Sketch**  
*Luca Rossetto, Ralph Gasser, Heiko Schuldt*  
[[Paper](https://arxiv.org/abs/1909.01477)]

#2017

**[IJCNLP, 2017] Draw and tell: Multimodal descriptions outperform verbal-or sketch-only descriptions in an image retrieval task**  
*Ting Han, David Schlangen*  
[[Paper](https://tingh.github.io/files/papers/sketch_ijcnlp_short.pdf)]

**[CVPR, 2017] Spatial-Semantic Image Search by Visual Feature Synthesis**  
*Long Mai, Hailin Jin, Zhe Lin, Chen Fang, Jonathan Brandt, Feng Liu*  
[[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mai_Spatial-Semantic_Image_Search_CVPR_2017_paper.pdf)]

**[ACM Multimedia, 2017] Region-based image retrieval revisited**  
*Ryota Hinami, Yusuke Matsui, Shin'ichi Satoh*  
[[Paper](https://dl.acm.org/doi/pdf/10.1145/3123266.3123312)]

#2014

**[Cancer Informatics, 2014] Medical image retrieval: a multimodal approach**  
*Yu Cao, Shawn Steffey, Jianbiao He, Degui Xiao, Cui Tao, Ping Chen, Henning Müller*  
[[Paper](https://journals.sagepub.com/doi/pdf/10.4137/CIN.S14053)]

#2013

**[10th Conference on Open Research Areas in Information Retrieval, 2013] NovaMedSearch: a multimodal search engine for medical case-based retrieval**  
*André Mourão, Flávio Martins*  
[[Paper](https://dl.acm.org/doi/pdf/10.5555/2491748.2491798)]

**[12th International Conference on Document Analysis and Recognition, 2013] Multi-modal Information Integration for Document Retrieval**  
*Ehtesham Hassan, Santanu Chaudhury, M. Gopal*  
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6628804)]

#2003

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
