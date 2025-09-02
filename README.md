# LLM-Powered Prediction of Hyperglycemia and Discovery of Behavioral Treatment Pathways from Wearables and Diet
### _Published in the Sensors journal_

Download and read the full paper here: [https://www.mdpi.com/1424-8220/25/17/5372](https://www.mdpi.com/1424-8220/25/17/5372)

Presentation slides: [https://abdullah-mamun.com/publication/2025-08-abdullah-llm_powered_hyperglycemia/GlucoLens_Mamun.pdf](https://abdullah-mamun.com/publication/2025-08-abdullah-llm_powered_hyperglycemia/GlucoLens_Mamun.pdf)

This repository contains the code and resources for **GlucoLens**, an explainable machine learning framework designed to predict the postprandial area under the curve (AUC) and hyperglycemia from multimodal data, including dietary intake, physical activity, and glucose levels. GlucoLens is an LLM-powered hybrid multimodal machine learning model for AUC and hyperglycemia prediction. Advanced LLMs such as GPT 3.5 Turbo, GPT 4, Claude Opus 4, DeepSeek V3, Gemini 2.0 Flash, Grok 3, and Mistral Large were employed to empower the GlucoLens system.

The latest version of the code is now available in this repository.

## Overview  
Postprandial hyperglycemia, characterized by elevated blood glucose levels after meals, is a significant predictor of progression toward type 2 diabetes. Accurate prediction and understanding of AUC can empower individuals to make lifestyle adjustments to maintain healthy glucose levels.  

GlucoLens is a novel computational model that combines machine learning with explainable AI to:  
1. **Predict AUC** based on fasting glucose, recent glucose trends, activity levels, and macronutrient intake.
2. **Prediction Model**: Random Forest backbone achieving a normalized root mean squared error (NRMSE) of 0.123, outperforming baseline models by 16%. 
3. **Classify hyperglycemia** with an accuracy of 79% and an F1 score of 0.749.
4. **Dive deep into relevant features** that are likely to be responsible for hyperglycemia with SHAP.  
5. **Provide actionable recommendations** to avoid hyperglycemia through diverse counterfactual scenarios.  

## Features  
- **Data Inputs**: Multimodal data including fasting glucose, recent glucose trends, physical activity metrics, and macronutrient composition of meals.  
- **Explainability**: SHAP and counterfactual explanations that provide actionable insights for lifestyle adjustments.

## Citation 
If you use part of our code or dataset or mention this work in your paper, please cite the following two publications:

**_1. LLM-Powered Prediction of Hyperglycemia and Discovery of Behavioral Treatment Pathways from Wearables and Diet_**
````
@Article{s25175372,
AUTHOR = {Mamun, Abdullah and Arefeen, Asiful and Racette, Susan B. and Sears, Dorothy D. and Whisner, Corrie M. and Buman, Matthew P. and Ghasemzadeh, Hassan},
TITLE = {LLM-Powered Prediction of Hyperglycemia and Discovery of Behavioral Treatment Pathways from Wearables and Diet},
JOURNAL = {Sensors},
VOLUME = {25},
YEAR = {2025},
NUMBER = {17},
ARTICLE-NUMBER = {5372},
URL = {https://www.mdpi.com/1424-8220/25/17/5372},
ISSN = {1424-8220},
DOI = {10.3390/s25175372}
}
````

**_2. Effects of Increased Standing and Light-Intensity Physical Activity to Improve Postprandial Glucose in Sedentary Office Workers: Protocol for a Randomized Crossover Trial_**

````
@article{wilson2023effects,
  title={Effects of Increased Standing and Light-Intensity Physical Activity to Improve Postprandial Glucose in Sedentary Office Workers: Protocol for a Randomized Crossover Trial},
  author={Wilson, Shannon L and Crosley-Lyons, Rachel and Junk, Jordan and Hasanaj, Kristina and Larouche, Miranda L and Hollingshead, Kevin and Gu, Haiwei and Whisner, Corrie and Sears, Dorothy D and Buman, Matthew P},
  journal={JMIR Research Protocols},
  volume={12},
  number={1},
  pages={e45133},
  year={2023},
  publisher={JMIR Publications Inc., Toronto, Canada}
}
````

## Contact
For questions, suggestions, or bug reports: a.mamun@asu.edu

Read our other papers: https://abdullah-mamun.com

Follow on X for the latest news: https://x.com/AB9Mamun
