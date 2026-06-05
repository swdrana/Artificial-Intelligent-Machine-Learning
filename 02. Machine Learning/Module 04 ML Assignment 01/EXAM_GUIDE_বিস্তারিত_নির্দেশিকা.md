# 🎓 ML Assignment 01 — বিস্তারিত পরীক্ষার নির্দেশিকা
### Dataset: `titanic_data_updated.csv` | Total: 100 Marks

> ✅ এই গাইডটি তোমাকে বুঝিয়ে বলবে — কী করতে হবে, কেন করতে হবে, এবং কোন tool/library ব্যবহার করবে।  
> ❌ সরাসরি উত্তর বা code এখানে নেই — তোমাকে নিজে বুঝে লিখতে হবে।

---

## 📦 শুরুর আগে — কোন Library গুলো লাগবে?

তোমার পুরো assignment-এ মূলত **৫টি library** ব্যবহার হবে। এগুলো Module 1, 2, 3 এর class-এই ব্যবহার হয়েছে:

| Library | কী কাজে লাগে | Import করার উপায় |
|---|---|---|
| `pandas` | Data load করা, table দেখা, manipulation | `import pandas as pd` |
| `numpy` | গাণিতিক কাজ, NaN handle করা | `import numpy as np` |
| `matplotlib.pyplot` | Basic chart/graph আঁকা | `import matplotlib.pyplot as plt` |
| `seaborn` | সুন্দর statistical chart আঁকা | `import seaborn as sns` |
| `sklearn` | Machine learning tools (imputer, split) | আলাদাভাবে import করতে হবে |

> 💡 **টিপস:** প্রথম cell-এ এই সব library একসাথে import করে নাও। Google Colab-এ এগুলো pre-installed থাকে, আলাদা install লাগবে না।

---

## 📋 Dataset সম্পর্কে জানো

**Titanic Dataset** — Titanic জাহাজের যাত্রীদের তথ্য। এই dataset-এ যা যা column আছে:

| Column নাম | অর্থ | ধরন |
|---|---|---|
| `PassengerId` | যাত্রীর ID নম্বর | সংখ্যা |
| `Survived` | বেঁচেছে কিনা (0=না, 1=হ্যাঁ) | সংখ্যা (Target) |
| `Pclass` | টিকেটের শ্রেণী (1=1st, 2=2nd, 3=3rd) | সংখ্যা |
| `Name` | নাম | লেখা |
| `Sex` | লিঙ্গ (male/female) | লেখা |
| `Age` | বয়স | সংখ্যা |
| `SibSp` | জাহাজে ভাই-বোন/স্বামী-স্ত্রী সংখ্যা | সংখ্যা |
| `Parch` | জাহাজে বাবা-মা/সন্তান সংখ্যা | সংখ্যা |
| `Ticket` | টিকেট নম্বর | লেখা |
| `Fare` | টিকেটের দাম | সংখ্যা |
| `Cabin` | কেবিন নম্বর (অনেক missing) | লেখা |
| `Embarked` | যে বন্দর থেকে উঠেছে (C/Q/S) | লেখা |

---

## ❓ Question 1 (10 Marks)
### "Load the Titanic dataset and display: Dataset shape, First 10 rows, 5 random samples"

### 🔍 এই Question-এ কী শেখানো হচ্ছে?
Data নিয়ে কাজ শুরু করলে **সবার আগে** data কেমন দেখতে সেটা বুঝতে হয়। এটাকে বলে **"Data Exploration"** বা **"Analytical View"**।

### 📚 কোন concepts দরকার?
1. **CSV file load করা** — `pandas` দিয়ে CSV file পড়া
2. **Shape দেখা** — মোট কতটি row (যাত্রী) আর কতটি column (feature) আছে
3. **Head দেখা** — প্রথম কয়েকটি row দেখা
4. **Random sample** — random কিছু row দেখা

### 🛠️ কোন pandas function ব্যবহার করবে?

| কাজ | Function | উদাহরণ |
|---|---|---|
| CSV load করা | `pd.read_csv()` | `df = pd.read_csv('titanic_data_updated.csv')` |
| Shape দেখা | `.shape` | `df.shape` → (rows, columns) দেখাবে |
| প্রথম N row দেখা | `.head(n)` | `df.head(10)` → প্রথম ১০টি row |
| Random N row দেখা | `.sample(n)` | `df.sample(5)` → যেকোনো ৫টি row |

### ⚠️ মনে রাখো:
- `df.shape` একটু আলাদা — এটার শেষে `()` নেই। এটা property, function না।
- Dataset যে folder-এ আছে, সেই path সঠিকভাবে দিতে হবে।

---

## ❓ Question 2 (10 Marks)
### "Feature Screening — Missing values, Duplicate rows খোঁজা এবং duplicate remove করা"

### 🔍 এই Question-এ কী শেখানো হচ্ছে?
**Feature Screening** মানে data কতটা পরিষ্কার সেটা পরীক্ষা করা। Real-world data সবসময় নোংরা থাকে — কিছু জায়গায় data নেই (missing), কিছু row একই তথ্য বারবার আছে (duplicate)।

### 📚 ৩টি কাজ করতে হবে:

**কাজ ১ — Missing Values খোঁজা:**
- কোন column-এ কতটি missing value আছে সেটা count করো
- `isnull()` বলে প্রতিটি cell-এ True/False দেয় (True = missing)
- `.sum()` দিয়ে column-wise count করো

**কাজ ২ — Duplicate Rows খোঁজা:**
- একই যাত্রীর তথ্য কি দুইবার আছে?
- `duplicated()` বলে কতগুলো duplicate আছে সেটা দেখায়

**কাজ ৩ — Duplicate Rows Remove করা:**
- `drop_duplicates()` দিয়ে duplicate সরাও
- **`inplace=True`** দিতে ভুলো না — এটা না দিলে আসল df পরিবর্তন হবে না!

### 🛠️ কোন functions?

| কাজ | Function |
|---|---|
| Missing count দেখা | `df.isnull().sum()` |
| Duplicate count দেখা | `df.duplicated().sum()` |
| Duplicate সরানো | `df.drop_duplicates(inplace=True)` |

### ⚠️ `inplace=True` কী?
`inplace=True` মানে — এই পরিবর্তনটা আসল dataframe-এই করো, নতুন কোনো copy তৈরি করো না।  
এটা না দিলে: `df = df.drop_duplicates()` এভাবেও করতে পারবে।

---

## ❓ Question 3 (10 Marks)
### "Statistical Profiling — Dataset info এবং statistical summary + 2টি observation"

### 🔍 এই Question-এ কী শেখানো হচ্ছে?
**Statistical Profiling** মানে data-র গভীরে যাওয়া। প্রতিটি column-এর গড়, সর্বোচ্চ-সর্বনিম্ন, কতটি null এই সব একসাথে দেখা।

### 📚 ২টি কাজ:

**কাজ ১ — Dataset Information:**
- `df.info()` দিলে প্রতিটি column-এর নাম, কতটি non-null value আছে, এবং data type (int, float, object) দেখায়
- এটা দেখে বুঝবে কোন column-এ missing data আছে

**কাজ ২ — Statistical Summary:**
- `df.describe()` দিলে **numerical column**গুলোর জন্য দেখাবে:
  - `count` — কতটি value আছে
  - `mean` — গড়
  - `std` — standard deviation (কতটা বিচ্ছিন্ন)
  - `min` — সর্বনিম্ন
  - `25%`, `50%`, `75%` — quartile (মধ্যবর্তী values)
  - `max` — সর্বোচ্চ

### ✍️ 2টি Observation লেখার tips:
Output দেখে নিজের ভাষায় লেখো, যেমন:
- "Age column-এ অনেক missing value আছে"
- "গড় বয়স প্রায় __ বছর"
- "Fare column-এর maximum value অনেক বেশি, যার মানে outlier থাকতে পারে"

---

## ❓ Question 4 (10 Marks)
### "Univariate Categorical Analysis on `Survived` — countplot, percentage, pie chart + 2 insights"

### 🔍 এই Question-এ কী শেখানো হচ্ছে?
**Univariate Analysis** মানে একটি মাত্র column নিয়ে analysis। এখানে `Survived` column নিয়ে কাজ করতে হবে — কতজন বেঁচেছে, কতজন মারা গেছে।

### 📚 ৩টি কাজ:

**কাজ ১ — Countplot:**
- `seaborn` এর `countplot()` দিয়ে bar chart আঁকো
- x-axis এ `Survived`, y-axis এ count থাকবে

**কাজ ২ — Percentage Distribution:**
- মোট যাত্রীর মধ্যে কত % বেঁচেছে, কত % মারা গেছে
- `value_counts()` দিয়ে count বের করো, তারপর percentage calculate করো

**কাজ ৩ — Pie Chart:**
- `matplotlib` এর `plt.pie()` দিয়ে pie chart আঁকো
- `autopct` দিয়ে percentage label দেখাও

### 🛠️ কোন functions?

| কাজ | Library | Function |
|---|---|---|
| Countplot | seaborn | `sns.countplot(data=df, x='Survived')` |
| Percentage | pandas | `df['Survived'].value_counts() / len(df) * 100` |
| Pie chart | matplotlib | `plt.pie(counts, labels=labels, autopct='%1.1f%%')` |

### ✍️ 2টি Insights লেখার tips:
- কতজন বেঁচেছে vs মারা গেছে তার ratio কেমন?
- Dataset কি balanced নাকি imbalanced?

---

## ❓ Question 5 (10 Marks)
### "Univariate Numerical Analysis on `Age` — Histogram, KDE plot + 2 observations"

### 🔍 এই Question-এ কী শেখানো হচ্ছে?
এখানে **numerical column** (`Age`) কেমন distribute হয়েছে সেটা visualization দিয়ে বোঝা।

### 📚 ২ ধরনের Plot:

**Histogram:**
- Age-কে ছোট ছোট range-এ ভাগ করে (যেমন 0-10, 10-20...) কতজন সেই range-এ পড়ে সেটা bar দিয়ে দেখায়
- `bins` দিয়ে কতটি ভাগ করবে সেটা control করো

**KDE Plot (Kernel Density Estimate):**
- Histogram এর মসৃণ version
- একটি smooth curve দেখায় — কোন বয়সে সবচেয়ে বেশি যাত্রী ছিল সেটা বোঝা যায়
- Peak যেখানে, সেখানে সবচেয়ে বেশি মানুষ

### 🛠️ কোন functions?

| কাজ | Library | Function |
|---|---|---|
| Histogram | seaborn | `sns.histplot(df['Age'], bins=50)` |
| KDE Plot | seaborn | `sns.kdeplot(df['Age'])` |
| Title দেওয়া | matplotlib | `plt.title('...')` |
| দেখানো | matplotlib | `plt.show()` |

### ✍️ 2টি Observations:
- Age distribution কি normal (bell shaped) নাকি skewed?
- কোন বয়সীরা সবচেয়ে বেশি ছিল?

---

## ❓ Question 6 (10 Marks)
### "Multivariate Analysis — `Sex` vs `Survived` countplot with hue + normalized survival ratio + 2 insights"

### 🔍 এই Question-এ কী শেখানো হচ্ছে?
**Multivariate Analysis** মানে দুটি বা বেশি column একসাথে দেখা। এখানে দেখতে হবে — পুরুষ ও মহিলার মধ্যে বেঁচে থাকার হার কেমন ছিল।

### 📚 ২টি কাজ:

**কাজ ১ — Countplot with Hue:**
- `hue` মানে রঙ দিয়ে আলাদা করা
- x-axis এ `Sex`, আর `Survived` কে hue দিলে দেখাবে — male/female এর মধ্যে কতজন বেঁচেছে/মারা গেছে

**কাজ ২ — Normalized Survival Ratio with groupby():**
- `groupby()` মানে কোনো column দিয়ে group করা
- `normalize=True` দিলে count এর বদলে proportion (0 থেকে 1) দেয়
- এটা দেখাবে male-দের মধ্যে __ % এবং female-দের মধ্যে __ % বেঁচেছিল

### 🛠️ কোন functions?

| কাজ | Function |
|---|---|
| Countplot with hue | `sns.countplot(x=df['Sex'], hue=df['Survived'])` |
| Normalized groupby | `df.groupby('Sex')['Survived'].value_counts(normalize=True)` |

### ✍️ 2টি Insights:
- কারা বেশি বেঁচেছে — পুরুষ নাকি মহিলা?
- এই পার্থক্য কি অনেক বেশি?

---

## ❓ Question 7 (10 Marks)
### "Barplot — `Pclass` vs `Fare` + 2 observations"

### 🔍 এই Question-এ কী শেখানো হচ্ছে?
**Barplot** দিয়ে categorical এবং numerical column-এর সম্পর্ক দেখা। এখানে দেখবে — কোন class-এর যাত্রীরা কত fare দিয়েছে।

### 📚 কাজ:
- x-axis: `Pclass` (1, 2, বা 3)
- y-axis: `Fare` (গড় fare দেখাবে)
- `seaborn` এর `barplot()` ব্যবহার করবে
- Barplot automatically প্রতিটি group-এর **mean (গড়)** দেখায়, error bar সহ

### 🛠️ Function:

```
sns.barplot(x=df['Pclass'], y=df['Fare'])
```

### ✍️ 2টি Observations:
- 1st class কি সবচেয়ে বেশি fare দিয়েছে?
- Pclass 1 → 2 → 3 এর মধ্যে fare কেমন কমেছে?

---

## ❓ Question 8 (10 Marks)
### "KDE Plot — `Age` distribution of survived vs non-survived + 2 insights"

### 🔍 এই Question-এ কী শেখানো হচ্ছে?
Question 5 এ শুধু `Age` এর KDE দেখেছিলে। এখানে `Survived` দিয়ে **আলাদা করে** দুটো KDE line দেখাবে — একটি যারা বেঁচেছে তাদের age distribution, আরেকটি যারা মারা গেছে তাদের।

### 📚 কাজ:
- `hue='Survived'` দিলে seaborn দুটো আলাদা line আঁকবে
- দুটো line compare করে insight বের করবে

### 🛠️ Function:
```
sns.kdeplot(data=df, x='Age', hue='Survived')
```

### ✍️ 2টি Insights:
- কম বয়সী (শিশু) যাত্রীরা কি বেশি বেঁচেছে?
- দুটো distribution কোথায় আলাদা হয়েছে?

---

## ❓ Question 9 (10 Marks)
### "Feature Engineering & Train-Test Split — Columns drop, X/y split, 80-20 split + shapes দেখা"

### 🔍 এই Question-এ কী শেখানো হচ্ছে?
Machine Learning Model train করার আগে data-কে প্রস্তুত করতে হয়। এটাকে বলে **Feature Engineering**। এরপর data-কে training এবং testing-এ ভাগ করতে হয়।

### 📚 ৩টি ধাপ:

**ধাপ ১ — অপ্রয়োজনীয় Column Drop করা:**
- `PassengerId` — শুধু একটি নম্বর, prediction-এ কোনো কাজে আসে না
- `Name` — নাম দিয়ে survival predict করা যায় না (ML model এর জন্য)
- `Ticket` — টিকেট নম্বরও কাজে আসে না
- `df.drop(['col1', 'col2', 'col3'], axis=1, inplace=True)` ব্যবহার করবে
- `axis=1` মানে column বরাবর drop করো (row নয়)

**ধাপ ২ — Features (X) এবং Target (y) আলাদা করা:**
- `X` = সব column **বাদে** `Survived` — এগুলো দিয়ে prediction করবে
- `y` = শুধু `Survived` column — এটাই predict করতে হবে
- `X = df.drop(['Survived'], axis=1)` এবং `y = df['Survived']`

**ধাপ ৩ — Train-Test Split:**
- sklearn এর `train_test_split` function ব্যবহার করবে
- `test_size=0.2` মানে 20% data test-এ যাবে, 80% training-এ
- `random_state=42` মানে প্রতিবার একই ভাগ হবে (reproducible)
- এই function টি return করে: `X_train, X_test, y_train, y_test`

### 🛠️ Import এবং functions:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Shape দেখতে:**
- `X_train.shape`, `X_test.shape`, `y_train.shape`, `y_test.shape`

### 💡 কেন Train-Test Split করি?
Model যে data দিয়ে শেখে (train), সেই একই data দিয়ে তার পরীক্ষা নিলে সে cheating করতে পারে। তাই আলাদা test data রাখি।

---

## ❓ Question 10 (10 Marks)
### "Missing Value Handling — Age (mean), Embarked (most frequent), Cabin (constant 'Missing') + check"

### 🔍 এই Question-এ কী শেখানো হচ্ছে?
ML Model-এ missing value (NaN) দিলে error হয়। তাই missing value-কে কোনো একটা মান দিয়ে পূরণ করতে হয় — এটাকে বলে **Imputation**।

### 📚 ৩ ধরনের Imputation:

**১. Age → Mean Strategy (গড় দিয়ে পূরণ):**
- Age column-এ missing value আছে
- যেসব জায়গায় age নেই, সেখানে সব যাত্রীর **গড় বয়স** বসিয়ে দাও
- কেন? কারণ বয়স সংখ্যা, আর গড় একটি reasonable অনুমান

**২. Embarked → Most Frequent Strategy (সবচেয়ে বেশিবার থাকা মান):**
- Embarked column-এ কয়েকটি missing value আছে
- যে বন্দরের নাম সবচেয়ে বেশি (mode), সেটা দিয়ে পূরণ করো
- কেন? কারণ এটা categorical, গড় বের করা যায় না

**৩. Cabin → Constant Strategy ('Missing' string দিয়ে পূরণ):**
- Cabin column-এ অনেক বেশি missing value
- এত বেশি missing যে অনুমান করা ঠিক না, তাই `"Missing"` নামের একটি নতুন category তৈরি করো
- এটা দিয়ে model বুঝবে — "এই যাত্রীর cabin info ছিল না"

### 🛠️ SimpleImputer কী?
`sklearn.impute` এর `SimpleImputer` একটি tool যা:
1. Training data থেকে **শেখে** (mean/mode calculate করে)
2. সেই শেখা দিয়ে missing value **পূরণ করে**

```python
from sklearn.impute import SimpleImputer

# Age এর জন্য
age_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
age_imputer.fit(X_train[['Age']])             # train data থেকে mean শেখো
X_train['Age'] = age_imputer.transform(X_train[['Age']]).ravel()
X_test['Age'] = age_imputer.transform(X_test[['Age']])   # test-এও same mean ব্যবহার করো
```

### ⚠️ গুরুত্বপূর্ণ নিয়ম:
- **শুধু `X_train` দিয়ে `.fit()` করবে** — test data থেকে শিখলে cheating হবে
- **`X_test`-এ শুধু `.transform()` করবে** — train-এ শেখা মান দিয়ে পূরণ করবে

### 🎯 Strategy summary:

| Column | Strategy | কারণ |
|---|---|---|
| `Age` | `'mean'` | Numerical, গড় logical |
| `Embarked` | `'most_frequent'` | Categorical, mode logical |
| `Cabin` | `'constant'`, `fill_value='Missing'` | Too many missing, new category |

### শেষে Check করো:
- `X_train.isnull().sum()` — সব 0 হওয়া উচিত
- `X_test.isnull().sum()` — সব 0 হওয়া উচিত

---

## 🗺️ পুরো Assignment-এর ধাপ একনজরে

```
ধাপ ১ → Library Import করো
ধাপ ২ → Dataset Load করো (Q1)
ধাপ ৩ → Data দেখো: shape, head, sample (Q1)
ধাপ ৪ → Feature Screening: missing, duplicate (Q2)
ধাপ ৫ → Statistical Profiling: info, describe (Q3)
ধাপ ৬ → Univariate Analysis: Survived (Q4)
ধাপ ৭ → Univariate Analysis: Age (Q5)
ধাপ ৮ → Multivariate: Sex vs Survived (Q6)
ধাপ ৯ → Multivariate: Pclass vs Fare (Q7)
ধাপ ১০ → Multivariate KDE: Age by Survived (Q8)
ধাপ ১১ → Feature Engineering + Train-Test Split (Q9)
ধাপ ১২ → Missing Value Imputation (Q10)
```

---

## 💡 সাধারণ Tips ও Common Mistakes

### ✅ করবে:
- প্রতিটি question-এর আগে একটি **Markdown cell** রাখবে (question নম্বর ও প্রশ্ন লিখে)
- Chart আঁকার পর `plt.show()` দাও
- Observation/Insight গুলো Markdown cell-এ বাংলা বা ইংরেজিতে লিখতে পারো

### ❌ করবে না:
- `inplace=True` ভুলে যাবে না (Q2 তে duplicate remove করতে)
- Test data দিয়ে imputer fit করবে না (Q10 তে)
- `axis=1` ভুলবে না column drop করতে (Q9 তে)

### 🔧 Google Colab-এ Dataset Upload করবে কীভাবে?
Option 1: Google Drive থেকে question-এ দেওয়া link ব্যবহার করো  
Option 2: Local file upload করো (Files panel থেকে)  
Option 3: `gdown` দিয়ে Google Drive থেকে directly download করো

---

## 📁 তোমার কাছে যা আছে (Module 1-3 থেকে reference নাও)

| File | কী শেখানো হয়েছে |
|---|---|
| `Module 01/EDA_P_1_.ipynb` | Data load, shape, head, sample, Feature Screening, Statistical Profiling, Univariate Analysis |
| `Module 02/EDA_P_2.ipynb` | Multivariate Analysis, countplot with hue, groupby, barplot, KDE plot |
| `Module 03/.../FE_1_(...).ipynb` | Feature Engineering, Train-Test Split, SimpleImputer (Age, Embarked, Cabin) |

> ✅ এই তিনটি notebook খুলে দেখো — assignment-এর সব প্রশ্নের উত্তর এই তিনটি থেকেই বের করতে পারবে। কিন্তু শুধু copy করো না, বোঝার চেষ্টা করো!

---

*শুভকামনা! তুমি পারবে! 🚀*
