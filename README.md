This repository contains the implementation of our full paper accepted for the IEEE EMBC 2024 Conference.

If you find our work useful, please consider citing it.

M. Y. Jabarulla, S. Oeltze-Jafra, P. Beerbaum and T. Uden, "MedDoc-Bot: A Chat Tool for Comparative Analysis of Large Language Models in the Context of the Pediatric Hypertension Guideline," 2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Orlando, FL, USA, 2024, pp. 1-4, doi: 10.1109/EMBC53108.2024.10781509.

You can access the full article on [arXiv](https://arxiv.org/abs/2405.03359). 

MedDoc-Bot: A Chat Tool for Comparative Analysis of Large Language Models in the Context of the Pediatric Hypertension Guideline<br/>
------------------------------------------------------------------------------
1. The MedDoc-Bot interface [CODE](https://github.com/yaseen28/MedDoc-Bot/blob/main/Main_MedDoc-Bot.py) allows users to choose from four quantized Language Model Models (LLMs) to chat with multiple PDF documents.The models used for our evaluations are downloaded from huggingface (Link provided below).
2. In our clinical use case, we assessed each model's performance by interpreting the hypertension in children and adolescents ESC guidelines PDF document. [Source](https://academic.oup.com/eurheartj/article/43/35/3290/6633855)<br/>
5. The original pediatric hypertension guidelines [Link](https://github.com/yaseen28/MedDoc-Bot/blob/main/Dataset/Original%20Pediatric_HTN_Guideline.pdf) contain text, tables, and figures on twelve pages. We carefully transformed figures and tables into textual representations to enhance interpretation and extraction. This involves providing detailed captions, extracting numerical data, and describing visual features in text [Transformed Document For Visual Element Analysis](https://github.com/yaseen28/MedDoc-Bot/blob/main/Dataset/Transformed_Pediatric_Guidelines%20.pdf). 
3. Evaluation involved using a benchmark dataset crafted by a pediatric specialist with four years of experience in pediatric cardiology manually generated twelve questions and corresponding responses by meticulously reviewing the pediatric hypertension guidelines.  [Dataset](https://github.com/yaseen28/MedDoc-Bot/tree/main/Dataset).
4. Evaluated models' accuracy, chrF, and METEOR score [Detailed Results](https://github.com/yaseen28/MedDoc-Bot/tree/main/Detailed%20Analysis).

# MedDoc-Bot Chat Tool 

A Streamlit-Powered Chat Tool for interpreting Multi-PDF Document using Four Large Language Models.

|![Image1](https://github.com/yaseen28/MedDoc-Bot/blob/main/UI_ScreenShot/Slide1.PNG?raw=true) | ![Image2](https://github.com/yaseen28/MedDoc-Bot/blob/main/UI_ScreenShot/Slide2.PNG?raw=true) |
|:---:|:---:|
|![Image1](https://github.com/yaseen28/MedDoc-Bot/blob/main/UI_ScreenShot/Slide3.PNG?raw=true) | ![Image2](https://github.com/yaseen28/MedDoc-Bot/blob/main/UI_ScreenShot/Slide4.PNG?raw=true) |


------------------------------------------------------------------------------
MedDoc-Bot:<br/> Manual Installation Guide Using Anaconda
------------------------------------------------------------------------------
#### 1. Install Conda

https://docs.conda.io/en/latest/miniconda.html

#### 2. Create a new conda environment

```
conda create -n MedDoc-Bot python=3.11
conda activate MedDoc-Bot
```
#### 3. Install Pytorch

| System | GPU | Command |
|--------|---------|---------|
| Windows | NVIDIA | `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| Windows | CPU only | `pip3 install torch torchvision torchaudio` |

The up-to-date commands can be found here: https://pytorch.org/get-started/locally/.

For NVIDIA, you also need to install the CUDA runtime libraries:

```
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-runtime
```

#### 4. Install the web UI

```
git clone https://github.com/yaseen28/MedDoc-Bot
cd MedDoc-Bot
pip install -r requirements.txt
```
#### 5. Download the Four Pre-Quantised Language Models to the Project Folder

   (i) Llama-2 {Version: llama-2-13b.Q5_K_S.gguf} [Link](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF)<br/>
   (ii) MedAlpaca {Version: medalpaca-13b.Q5_K_S.gguf} [Link](https://huggingface.co/TheBloke/medalpaca-13B-GGUF)<br/>
   (iii) Meditron {Version: meditron-7b.Q5_K_S.gguf} [Link](https://huggingface.co/TheBloke/meditron-7B-GGUF)<br/>
   (iv) Mistral {Version: mistral-7b-instruct-v0.2.Q5_K_M.gguf} [Link](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)<br/>

   NOTE!! Please ensure that you rename the model file to match the name listed in the 'Select Model' dropdown in the browser. 

### 6. Start the MedDoc-Bot

```
conda activate MedDoc-Bot
cd MedDoc-Bot
streamlit run Main_MedDoc-Bot.py
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501

### 7. Provide Default Username and Password
```
User
User@123
```

If you find our work useful, please consider citing it.

@INPROCEEDINGS{10781509,
  author={Jabarulla, Mohamed Yaseen and Oeltze-Jafra, Steffen and Beerbaum, Philipp and Uden, Theodor},
  booktitle={2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)}, 
  title={MedDoc-Bot: A Chat Tool for Comparative Analysis of Large Language Models in the Context of the Pediatric Hypertension Guideline}, 
  year={2024},
  volume={},
  number={},
  pages={1-4},
  keywords={Hypertension;Measurement;Analytical models;Visualization;Large language models;Biological system modeling;Portable document format;Meteors;Reliability;Guidelines;Medical language models;Streamlit;Medical Guidelines PDF Chatbot;Clinical care},
  doi={10.1109/EMBC53108.2024.10781509}}
}

