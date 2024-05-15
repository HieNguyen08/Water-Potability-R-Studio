# Water-Potability-R-Studio
A project about predicting the quality of water that it cound be drunk or not
## Data Introduction
Access to safe drinking water is a fundamental human right and an essential component of effective health protection policies. It is a critical health and development issue at the national, regional, and local levels. In some regions, investments in water supply and sanitation have been shown to yield net economic benefits. This is because the reduction in negative health impacts and healthcare costs outweighs the costs of carrying out intervention measures. However, according to the World Health Organization (WHO), nearly 780 million people live without access to clean water, and over 2.5 billion people are in need of improved sanitation facilities. This information was released by the WHO on World Water Day (March 22nd).

Therefore, it is important to assess whether water is clean before using it. This project will use knowledge of probability and statistics and related tools to assess the quality of water through the provided parameters of various water samples in Africa.

The dataset that is used in this project is about drinking water potability. Here are some general details of the dataset:

- **Title**: Drinking water potability
- **Source information**: Aditya Kadiwal, *Senior Data Engineer at Blazeclan Technologies Pvt. Ltd. Pune, Maharashtra, India*.
- **Number of water bodies**: 3276 (1998 samples not potable and 1278 samples potable)
- **Number of Variables**: 10 (Described in section 2.2)

#### Variables Description

| **Variable**       | **Data type** (*cont = continuous, cate = categorical*)              | **Unit**   | **Description**                                     |
|--------------------|----------------------------------------------------------------------|------------|-----------------------------------------------------|
| ph                 | \(\left\{ x \in \mathbb{R} \mid x > 0 \right\}\), cont               | None       | pH of water                                         |
| hardness           | \(\left\{ x \in \mathbb{R} \mid x > 0 \right\}\), cont               | mg/L       | Capacity of water to precipitate soap               |
| Solids             | \(\left\{ x \in \mathbb{R} \mid x > 0 \right\}\), cont               | ppm        | Total dissolved solids                              |
| Chloramines        | \(\left\{ x \in \mathbb{R} \mid x > 0 \right\}\), cont               | ppm        | Amount of Chloramines                               |
| Sulfate            | \(\left\{ x \in \mathbb{R} \mid x > 0 \right\}\), cont               | mg/L       | Amount of Sulfate dissolved                         |
| Conductivity       | \(\left\{ x \in \mathbb{R} \mid x > 0 \right\}\), cont               | \(\mu\)S/cm | Electrical conductivity of water                    |
| Organic_carbon     | \(\left\{ x \in \mathbb{R} \mid x > 0 \right\}\), cont               | ppm        | Amount of organic carbon                            |
| Trihalomethanes    | \(\left\{ x \in \mathbb{R} \mid x > 0 \right\}\), cont               | \(\mu\)g/L | Amount of Trihalomethanes                           |
| Turbidity          | \(\left\{ x \in \mathbb{R} \mid x > 0 \right\}\), cont               | NTU        | Measure of light emitting property of water         |
| Potability         | \(x = 0\) or \(x = 1\), cate  

## A little review about data
### Pie chart
<div style="display: flex; justify-content: space-around;">
  <img src="pie chart.png" alt="Headquarter Network Diagram" width="80%">
</div>
### Boxplot and Histogram
<div style="display: flex; justify-content: space-around;">
  <img src="box plot-chlroamines.png" alt="Headquarter Network Diagram" width="45%">
  <img src="box plot-conductivity.png" alt="Branch Network Diagram" width="45%">
</div>
<div style="display: flex; justify-content: space-around;">
  <img src="box plot-hardness.png" alt="Headquarter Network Diagram" width="45%">
  <img src="box plot-organic-carbon.png" alt="Branch Network Diagram" width="45%">
</div>
<div style="display: flex; justify-content: space-around;">
  <img src="box plot-ph.png" alt="Headquarter Network Diagram" width="45%">
  <img src="box plot-solids.png" alt="Branch Network Diagram" width="45%">
</div>
<div style="display: flex; justify-content: space-around;">
  <img src="box plot-sulfate.png" alt="Headquarter Network Diagram" width="33%">
  <img src="box plot-trihalomethanes.png" alt="Branch Network Diagram" width="33%">
  <img src="box plot-tubidity.png" alt="Branch Network Diagram" width="33%">
</div>

### Corrheat map and scatter plot matrix
<div style="display: flex; justify-content: space-around;">
  <img src="cor heat map.png" alt="Headquarter Network Diagram" width="90%">
</div>
<div style="display: flex; justify-content: space-around;">
  <img src="scatter plot matrix.png" alt="Headquarter Network Diagram" width="90%">
</div>

## References

1. **Introductory Statistics with R**
   - *Author*: Peter Dalgaard
   - *Year*: 2008
   - *Publisher*: Springer

2. **Linear Regression on Student Grade Prediction**
   - *How published*: [Link](https://rstudio-pubs-static.s3.amazonaws.com/716359_6902dfdd88684340a5f5e11038b9ac22.html)

3. **DEFINITION histogram**
   - *How published*: [Link](https://www.techtarget.com/searchsoftwarequality/definition/histogram)

4. **Understanding Boxplots**
   - *How published*: [Link](https://builtin.com/data-science/boxplot)

5. **Water Quality-data table**
   - *How published*: [Link](https://www.kaggle.com/datasets/adityakadiwal/water-potability/data?fbclid=IwAR21lNIVgkGFLDkf9MHgmNKPH1ZTWDObRKnT-rO_flHctv9D3PEGw8aTMSw)

6. **The Kappa Statistic in Reliability Studies: Use, Interpretation, and Sample Size Requirements, Physical Therapy**
   - *Author*: Julius Sim, Chris C Wright
   - *Year*: 2005

7. **Detecting and Treating Outliers | Treating the odd one out!**
   - *How published*: [Link](https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/)

8. **A Step-by-Step Explanation of Principal Component Analysis (PCA)**
   - *How published*: [Link](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)

9. **Comparing Decision Tree Algorithms: Random Forest vs. XGBoost**
   - *How published*: [Link](https://www.activestate.com/blog/comparing-decision-tree-algorithms-random-forest-vs-xgboost/)

10. **Guide to AUC ROC Curve in Machine Learning**
    - *How published*: [Link](https://www.geeksforgeeks.org/auc-roc-curve/)

11. **What is SVM? Machine Learning Algorithm Explained**
    - *How published*: [Link](https://www.springboard.com/blog/data-science/svm-algorithm/)

12. **Support Vector Machine (SVM) Algorithm**
    - *How published*: [Link](https://www.geeksforgeeks.org/support-vector-machine-algorithm/)

13. **Understanding Overfitting and How to Prevent It**
    - *How published*: [Link](https://www.investopedia.com/terms/o/overfitting.asp)

14. **Overfitting and Underfitting – Common Causes & Solutions**
    - *How published*: [Link](https://www.analyticsfordecisions.com/overfitting-and-underfitting/)

15. **Overfitting vs Underfitting**
    - *How published*: [Link](https://medium.com/mlearning-ai/overfitting-vs-underfitting-6a41b3c6a9ad)

16. **Logistic Regression**
    - *How published*: [Link](https://www.geeksforgeeks.org/understanding-logistic-regression/)
