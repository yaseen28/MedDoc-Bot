MedDoc-Bot: A Chat Tool for Comparative Analysis of Large Language Models in the Context of Pediatric Hypertension Guidelines<br/> (This article has been submitted as a full contributed paper for EMBC 2024.)
------------------------------------------------------------------------------
1. The MedDoc-Bot interface allows users to choose from four quantized Language Model Models (LLMs) for document processing and chat with multiple PDF documents.The models used for our evaluations are downloaded from huggingface.
4. In a clinical use case, assessed each model's performance by interpreting the hypertension in children and adolescents ESC guidelines PDF document. [Source](https://academic.oup.com/eurheartj/article/43/35/3290/6633855)<br/>
5. The original pediatric hypertension guidelines [Link]([https://academic.oup.com/eurheartj/article/43/35/3290/6633855](https://github.com/yaseen28/MedDoc-Bot/blob/main/Dataset/Original%20Pediatric_HTN_Guideline.pdf) contain text, tables, and figures on twelve pages. We carefully transformed figures and tables into textual representations to enhance interpretation and extraction. This involves providing detailed captions, extracting numerical data, and describing visual features in text [Transfored Document For Visual Element Analysis](https://github.com/yaseen28/MedDoc-Bot/blob/main/Dataset/Transformed_Pediatric_Guidelines%20.pdf). 
3. Evaluation involved using a benchmark dataset crafted by an expert cardiologist [Dataset](https://github.com/yaseen28/MedDoc-Bot/tree/main/Dataset).
4. Evaluated models' accuracy, chrF, and METEOR score [Detailed Results].
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

#### 3. Install the web UI

```
git clone https://github.com/yaseen28/MedDoc-Bot
cd MedDoc-Bot
pip install -r requirements.txt
```
#### 4. Download the Four Pre-Quantised Language Models to the Porject Folder

```
   (i) Llama-2 [Link](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF)<br/>
   (ii) MedAlpaca [Link]([https://huggingface.co/TheBloke/](https://huggingface.co/TheBloke/medalpaca-13B-GGUF)<br/>
   (iii) Meditron [Link](https://huggingface.co/TheBloke/meditron-7B-GGUF)<br/>
   (iv) Mistral [Link](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)<br/>

```


### 5. Start the MedDoc-Bot

```
conda activate MedDoc-Bot
cd MedDoc-Bot
streamlit run Main_MedDoc-Bot.py
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501

### 6. Provide Default Username and Password
```
User
User@123
```

