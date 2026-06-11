**Mandatory**  
**Problem 1: Z-Score Method (Normal Data)**  
Dataset (Age of students in a class):

| \[18, 19, 20, 20, 21, 21, 22, 22, 23, 24, 25, 26, 27, 28, 30, 32, 35, 40, 45, 70\] |
| :---- |

Tasks:

* Calculate the mean and standard deviation of Age.  
* Compute Z-Score for each value.  
* Identify outliers using |Z| \> 3 conditions.  
* Remove those outliers and show the new dataset length.

**Problem 2: IQR Method (Skewed Data)**  
Dataset (Monthly salaries in thousand BDT):

| \[25, 28, 30, 32, 32, 33, 35, 35, 36, 38, 40, 42, 45, 48, 50, 55, 60, 65, 70, 80, 120, 150, 200, 250, 500\] |
| :---- |

Tasks:

* Calculate Q1 (25th percentile) and Q3 (75th percentile).  
* Find IQR \= Q3 \- Q1.  
* Calculate lower bound and upper bound.  
* Identify outliers.  
* Cap (clip) the outliers to the bounds instead of deleting them.  
* Show min and max after capping.


**Problem 3: Choose the Right Method**  
Dataset A (Exam scores):

| \[55, 58, 60, 62, 65, 65, 68, 70, 72, 75, 78, 80, 85, 90, 95, 98, 100, 100, 100, 15, 20\] |
| :---- |

Note: There are two very low scores (15, 20\) and three high scores (100).

Dataset B (House prices in lakh BDT):

| \[30, 32, 35, 38, 40, 42, 45, 48, 50, 55, 60, 65, 70, 80, 90, 100, 120, 150, 200, 300, 400, 500, 800, 1000, 1200\] |
| :---- |

Tasks:

* For Dataset A, plot a histogram (conceptually) – is it normal or skewed?  
* Which method (Z-Score or IQR) would you choose for Dataset A? Why?  
* For Dataset B, which method would you choose? Why?  
* Apply IQR on Dataset B and count how many outliers you get.

**Problem 4: Winsorization (Percentile Method)**  
Dataset (Product ratings out of 10):

| \[1, 2, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 1, 2, 10, 10, 10, 0.5, 9.5\] |
| :---- |

Tasks:

* Use 5% Winsorization (x \= 5, meaning cap at 5th and 95th percentiles).  
* Calculate 5th percentile and 95th percentile.  
* Cap all values below 5th percentile to the 5th percentile value.  
* Cap all values above 95th percentile to the 95th percentile value.  
* Show the min and max after Winsorization.  
* Explain why Winsorization is better than deletion here.

**Extra (Optional)**  
**Problem 5: Mixed Methods – Compare Z-Score & IQR on Same Data**  
Dataset (Product ratings out of 10):

| \[12, 15, 18, 20, 22, 22, 23, 24, 25, 25, 26, 27, 28, 30, 32, 35, 38, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 120, 150, 200, 500, 1000, 2000\] |
| :---- |

Tasks:

* Plot a histogram (conceptually) and decide if the data is normal or skewed.  
* Apply Z-Score method (|Z| \> 3\) – count outliers and list them.  
* Apply IQR method (1.5×IQR) – count outliers and list them.  
* Which method found more outliers? Why?  
* Which method would you trust here? Justify.

**Problem 6: Real-World Scenario – Employee Bonus**  
**Scenario:**  
A company gives an annual bonus (in thousand BDT). HR wants to remove extreme outliers before calculating average bonus for policy making.

Dataset (Bonus amounts):

| \[5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 12, 13, 15, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 120, 150, 200, 300\] |
| :---- |

Tasks:

* Calculate mean, median, and mode. What do you observe?  
* Apply IQR method and find outliers.  
* Instead of deleting, apply Winsorization with x \= 3 (i.e., cap at 3rd and 97th percentile).  
* Show the new min and max after Winsorization.  
* Calculate the new mean after Winsorization. Is it more reasonable than the original mean?

 