# 🛒 Shopee App Customer Sentiment Analysis

### Project Background
In the highly competitive e-commerce market, customer experience is the primary driver of success. [cite_start]Shopee receives thousands of app reviews daily, making manual analysis impossible[cite: 18, 20].

This project utilises **Natural Language Processing (NLP)** and **Unsupervised Machine Learning** to transform unstructured text data into actionable business strategies. [cite_start]Using the **CRISP-DM framework**, I analysed over 700,000 user reviews to identify key pain points and drivers of satisfaction[cite: 98, 107].

### Business Objectives
* **Goal:** Identify the top 3 drivers of positive and negative user experience.
* [cite_start]**Target:** Provide data-backed recommendations to reduce negative reviews by 15% and increase average app store ratings from 4.3 to 4.5 stars[cite: 102].

---

### Technical Implementation

#### 1. Data Cleaning & Preprocessing (R)
Handling "noisy" user-generated content required rigorous cleaning:
* [cite_start]**Language Filtering:** Used `cld3` to isolate English reviews and remove non-English characters[cite: 334].
* [cite_start]**Slang Removal:** Engineered custom Regex filters to remove Singlish and Tagalog particles (e.g., "lah", "mah", "po", "naman") to improve model accuracy[cite: 335, 342].
* [cite_start]**Normalisation:** Applied Lemmatisation (`textstem`) rather than Stemming to preserve semantic meaning (e.g., keeping "delivery" instead of "deliveri")[cite: 660, 661].

![Word Cloud](Assets/wordcloud.png)
*Figure: Word cloud visualisation of the most frequent terms after stopword removal.*

#### 2. Topic Modelling (LDA)
I implemented **Latent Dirichlet Allocation (LDA)** to discover hidden themes in the reviews.
* **Algorithm Selection:** Tested both **VEM** and **Gibbs Sampling**. [cite_start]The Gibbs algorithm proved superior in creating distinct, non-overlapping topics[cite: 1160, 1166].
* [cite_start]**Optimising k (Number of Topics):** Used Perplexity Scores and Silhouette Analysis (Elbow Method) to determine that **k=8** was the optimal number of topics for interpretability[cite: 891, 1273].

![Elbow Plot](Assets/elbow_plot.png)
*Figure: Elbow plot of Silhouette scores determining k=8 as the optimal balance for interpretability.*

#### 3. Sentiment Analysis
Comparative analysis of three lexicon-based models:
* [cite_start]**Syuzhet (NRC):** Good for emotional nuance (joy, anger) but computationally expensive[cite: 976].
![Syuzhet Output](Assets/syuzhet.png)
* [cite_start]**Sentimentr:** Efficient but struggled with accuracy on 1-star reviews[cite: 1325].
![Sentimentr](Assets/sentimentr.png)
* [cite_start]**Tidytext (AFINN):** **Selected Model.** Demonstrated the highest correlation with actual star ratings and effectively captured valence shifters[cite: 1328, 1336].
![Tidytext](Assets/tidytext.png)

---

### Key Insights & Visualisations

#### Insight 1: The "Problem Areas" are Specific
The LDA model successfully clustered complaints into interpretable topics. [cite_start]The analysis revealed that **Topic 1 (Payment & Account Issues)** and **Topic 7 (Delivery Failures)** are the primary drivers of 1-star reviews[cite: 1359, 1361].

![Topic Distribution](Assets/topic_distribution.png)
*Figure: Stacked bar chart showing that Payment Issues (Topic 1) dominate the negative reviews.*

#### Insight 2: The "Convenience" Advantage
[cite_start]Positive reviews were heavily concentrated in **Topic 8 (Convenience Value)**, validating that Shopee's competitive advantage lies in its ease of use for remote shopping[cite: 1415].

---

### Business Recommendations
Based on the analysis, the following strategic actions are recommended:
1.  [cite_start]**Prioritise Payment Gateway Stability:** Since "Payment Issues" correlate most strongly with churn (1-star ratings), resources should be shifted from UI updates to backend stability[cite: 1416].
2.  **Courier Accountability:** "Delivery Failures" are a major pain point. [cite_start]Stricter SLAs for logistics partners could reduce negative sentiment significantly[cite: 1416].
3.  [cite_start]**Automated Triage:** Deploy the trained LDA model to automatically flag incoming reviews about "Scams" or "Refunds" for immediate human intervention[cite: 1397].

### Repository Structure
* `Scripts/`: 
  * `1_topic_modelling_lda.R`: Full pipeline for preprocessing and Gibbs Sampling LDA.
  * `2_sentiment_analysis.R`: Comparative sentiment analysis scripts.
* `Data/`: Contains the processed datasets and custom stopword lists (Singlish/Domain-specific).

---
*Created by Wong Jung Yong*