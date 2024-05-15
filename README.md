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
