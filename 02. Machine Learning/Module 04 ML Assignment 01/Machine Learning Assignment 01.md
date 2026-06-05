# **ML Assignment 01** 

**Dataset:** Titanic Dataset (`titanic_data_updated.csv`)  
**Total Marks:** 100

Exam Instructions : 

1. প্রথমে একটি Google Colab ফাইল খুলবে , এরপর প্রথম cell এ নিজের নাম এবং কোর্সে registration করা ইমেইল দিবে   
2. Question wise numbering করে Text cell রাখবে এবং এর নিচে Code cell থাকবে, চেষ্টা করবে একটি code cell এ একটি question উত্তর দেয়ার  
3. Google colab এর মধ্যে কোডগুলো করবে   
4.  এবং সেই ফাইলটি ‘Anyone with the link’ & ‘View’ Access দিয়ে ফাইলটির Shareble Link টি সাবমিট করবে।

**Question Dataset Link:** [https://drive.google.com/file/d/1aHSSJkDphOCbdGRnnXUDPHvPr08Uwtsk/view?usp=sharing](https://drive.google.com/file/d/1aHSSJkDphOCbdGRnnXUDPHvPr08Uwtsk/view?usp=sharing)

---

## **Question 1 (10 Marks)**

Load the Titanic dataset and display:

* Dataset shape  
* First 10 rows  
* 5 random samples

---

## **Question 2 (10 Marks)**

Perform feature screening on the Titanic dataset by:

* Finding total missing values  
* Finding duplicate rows  
* Removing duplicate rows permanently

---

## **Question 3 (10 Marks)**

Perform statistical profiling of the Titanic dataset and display:

* Dataset information  
* Statistical summary of numerical columns

Also write 2 observations from the output.

---

## **Question 4 (10 Marks)**

Perform univariate categorical analysis on the `Survived` column by:

* Creating a countplot  
* Calculating percentage distribution  
* Creating a pie chart

Write 2 insights from the analysis.

---

## **Question 5 (10 Marks)**

Perform univariate numerical analysis on the `Age` column by:

* Creating a histogram  
* Creating a KDE plot

Write 2 observations from the plots.

---

## **Question 6 (10 Marks)**

Perform multivariate analysis between `Sex` and `Survived` using a countplot with hue.

Also calculate normalized survival ratios using `groupby()`.

Write 2 insights from the analysis.

---

## **Question 7 (10 Marks)**

Create a barplot showing the relationship between `Pclass` and `Fare`.

Write 2 observations from the visualization.

---

## **Question 8 (10 Marks)**

Create a KDE plot to compare the `Age` distribution of survived and non-survived passengers.

Write 2 insights from the graph.

---

## **Question 9 (10 Marks)**

Perform feature engineering and train-test split by:

* Dropping `PassengerId`, `Name`, and `Ticket` columns  
* Separating features (`X`) and target (`y`)  
* Splitting the dataset using `test_size=0.2` and `random_state=42`

Display the shapes of train and test datasets.

---

## **Question 10 (10 Marks)**

Handle missing values in the Titanic dataset by:

* Imputing missing `Age` values using `SimpleImputer` with mean strategy  
* Imputing missing `Embarked` values using most frequent strategy  
* Imputing missing `Cabin` values using constant strategy with `"Missing"`  
  Finally, check whether any missing values remain.

