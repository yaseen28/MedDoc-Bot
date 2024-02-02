MedDoc-Bot:<br/> A Chat Tool for Comparative Analysis of Large Language Models in the Context of Pediatric Hypertension Guidelines
------------------------------------------------------------------------------
1. The MedDoc-Bot interface allows users to choose from four quantized Language Model Models (LLMs) for document processing and chat with multiple PDF documents.The models used for our evaluations are downloaded from huggingface.
4. In a clinical use case, assessed each model's performance by interpreting the hypertension in children and adolescents ESC guidelines PDF document. [Link](https://academic.oup.com/eurheartj/article/43/35/3290/6633855)<br/>
5. The original pediatric hypertension guidelines [Link](https://academic.oup.com/eurheartj/article/43/35/3290/6633855) contain text, tables, and figures on twelve pages. We carefully transformed figures and tables into textual representations to enhance interpretation and extraction. This involves providing detailed captions, extracting numerical data, and describing visual features in text [Transfored Document For Visual Element Analysis]. 
3. Evaluation involved using a benchmark dataset crafted by an expert cardiologist [Dataset](https://github.com/yaseen28/MedDoc-Bot/tree/main/Dataset).
4. Evaluated models' accuracy, chrF, and METEOR score [Detailed Metrics].
