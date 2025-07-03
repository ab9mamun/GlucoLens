# LLM-Powered Prediction of Hyperglycemia and Discovery of Behavioral Treatment Pathways from Wearables and Diet

This repository contains the code and resources for **GlucoLens**, an explainable machine learning framework designed to predict the postprandial area under the curve (AUC) and hyperglycemia from multimodal data, including dietary intake, physical activity, and glucose levels. GlucoLens is an LLM-powered hybrid multimodal machine learning model for AUC and hyperglycemia prediction.

## Overview  
Postprandial hyperglycemia, characterized by elevated blood glucose levels after meals, is a significant predictor of progression toward type 2 diabetes. Accurate prediction and understanding of AUC can empower individuals to make lifestyle adjustments to maintain healthy glucose levels.  

GlucoLens is a novel computational model that combines machine learning with counterfactual explanations to:  
1. **Predict AUC** based on fasting glucose, recent glucose trends, activity levels, and macronutrient intake.
2. **Prediction Model**: Random Forest backbone achieving a normalized root mean squared error (NRMSE) of 0.123, outperforming baseline models by 16%. 
3. **Classify hyperglycemia** with an accuracy of 73.3% and an F1 score of 0.716.  
4. **Provide actionable recommendations** to avoid hyperglycemia through diverse counterfactual scenarios.  

## Features  
- **Data Inputs**: Multimodal data including fasting glucose, recent glucose trends, physical activity metrics, and macronutrient composition of meals.  
- **Explainability**: Counterfactual explanations that provide actionable insights for lifestyle adjustments.

## Citation 
If you use part of our code or dataset or mention this work in your paper, please cite the following two publications:

**_1. GlucoLens: Explainable Postprandial Blood Glucose Prediction from Diet and Physical Activity_**
````
@misc{mamun2025glucolens,
    title={GlucoLens: Explainable Postprandial Blood Glucose Prediction from Diet and Physical Activity},
    author={Abdullah Mamun and Asiful Arefeen and Susan B. Racette and Dorothy D. Sears and Corrie M. Whisner and Matthew P. Buman and Hassan Ghasemzadeh},
    year={2025},
    eprint={2503.03935},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
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
For questions, suggestions, or bug report: a.mamun@asu.edu
#### Read our other papers: https://abdullah-mamun.com

