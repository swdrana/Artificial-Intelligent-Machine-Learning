# 📚 AI Programming with Python — Final Exam বাংলা গাইড
### লেখক: Masuduzzaman | ceo@swdrana.com

> প্রতিটি প্রশ্নের কোড ভাগ ভাগ করে সহজ বাংলায় বিস্তারিত ব্যাখ্যা দেওয়া হয়েছে।

---

## 🔰 0) Starter Code

### লাইব্রেরি import করা
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
- **`numpy`** → সংখ্যার array নিয়ে গণিত ও statistics করার library। `np` হলো এর সংক্ষিপ্ত নাম।
- **`pandas`** → Excel-এর মতো টেবিল (DataFrame) নিয়ে কাজ করার library। `pd` হলো সংক্ষিপ্ত নাম।
- **`matplotlib.pyplot`** → বার চার্ট, হিস্টোগ্রাম, স্ক্যাটার প্লট আঁকার library।
- **`seaborn`** → matplotlib-এর উপর তৈরি, আরও সুন্দর ও স্মার্ট চার্ট আঁকার library।

### Random Seed নির্ধারণ
```python
np.random.seed(42)
```
- **কেন?** Random সংখ্যা তৈরি করার সময় `seed(42)` দিলে প্রতিবার কোড চালালে **একই** সংখ্যা আসে।
- এর ফলে তোমার এবং শিক্ষকের output হুবহু মিলবে।

### Data তৈরি করা
```python
num_students = 50
student_ids = np.arange(201, 251)
student_ages = np.random.randint(15, 22, 50)
student_scores = np.random.randint(40, 100, 50)
student_attendance = np.random.randint(60, 100, 50)
```
- **`np.arange(201, 251)`** → ২০১ থেকে ২৫০ পর্যন্ত পর পর সংখ্যার array।
- **`np.random.randint(15, 22, 50)`** → ১৫ থেকে ২১-এর মধ্যে ৫০টি random পূর্ণসংখ্যা।

### Grade নির্ধারণ করা
```python
grade_conditions = [
    student_scores >= 80,
    (student_scores >= 65) & (student_scores < 80),
    (student_scores >= 50) & (student_scores < 65),
    student_scores < 50
]
grade_values = ["A", "B", "C", "F"]
student_grades = np.select(grade_conditions, grade_values, default="F")
```
- **`np.select()`** → প্রতিটি condition একটার পর একটা চেক করে। প্রথমটা সত্য হলে "A" দাও, দ্বিতীয়টা সত্য হলে "B" দাও, এভাবে চলে।
- **`& `** → দুটো condition একসাথে সত্য হতে হবে (AND)।

### DataFrame ও CSV তৈরি
```python
data = pd.DataFrame({...})
data.to_csv("school_data.csv", index=False)
```
- **`pd.DataFrame()`** → সব array নিয়ে Excel-এর মতো একটা টেবিল তৈরি করো।
- **`to_csv(index=False)`** → টেবিলটা CSV ফাইলে সেভ করো, `index=False` মানে বামে ০,১,২ সংখ্যা থাকবে না।

---

## ✅ Question 01 — NumPy Array Properties [10 Marks]

**প্রশ্ন:** score ও attendance কলাম থেকে NumPy array বানাও, তারপর dtype/ndim/shape/size দেখাও।

### ধাপ ১: Array তৈরি
```python
score_array = np.array(data['score'])
attendance_array = np.array(data['attendance'])
```
- **`data['score']`** → DataFrame-এর 'score' কলামটা নাও। এটা হলো Pandas Series।
- **`np.array(...)`** → সেই Series-কে NumPy array-তে রূপান্তর করো।
- **কেন রূপান্তর?** NumPy array-তে `.dtype`, `.ndim`, `.shape`, `.size` সরাসরি পাওয়া যায়।

### ধাপ ২: Properties দেখানো
```python
print("Data Type      :", score_array.dtype)
print("Dimensions     :", score_array.ndim)
print("Shape          :", score_array.shape)
print("Total Elements :", score_array.size)
```
| Property | কী দেখায় | উদাহরণ output |
|---|---|---|
| `.dtype` | কী ধরনের data সংরক্ষিত | `int64` |
| `.ndim` | কতটি মাত্রা আছে | `1` (একটাই সারি) |
| `.shape` | প্রতিটি মাত্রায় কতটি element | `(50,)` মানে ৫০টি |
| `.size` | মোট কতটি element | `50` |

---

## ✅ Question 02 — Statistics ও Filtering [10 Marks]

**প্রশ্ন:** max, min, গড় বের করো। গড়ের বেশি score যাদের তাদের আলাদা করো।

### ধাপ ১: Statistics বের করা
```python
max_score      = np.max(score_array)
min_score      = np.min(score_array)
avg_score      = np.mean(score_array)
avg_attendance = np.mean(attendance_array)
```
- **`np.max()`** → array-এর সবচেয়ে বড় সংখ্যা বের করে।
- **`np.min()`** → array-এর সবচেয়ে ছোট সংখ্যা বের করে।
- **`np.mean()`** → সব সংখ্যা যোগ করে মোট সংখ্যা দিয়ে ভাগ করে গড় বের করে।

### ধাপ ২: Boolean Indexing দিয়ে Filter করা
```python
above_avg_scores = score_array[score_array > avg_score]
```
এই একটি লাইনে আসলে দুটো কাজ হচ্ছে:

**প্রথম কাজ:** `score_array > avg_score`
- প্রতিটি score চেক করে: "এটা কি গড়ের বেশি?"
- যদি হ্যাঁ → `True`, না হলে → `False`
- ফলাফল: `[False, True, True, False, True, ...]` এই ধরনের array

**দ্বিতীয় কাজ:** `score_array[True/False array]`
- শুধু `True` যে জায়গাগুলোতে সেই score গুলো নেওয়া হয়
- এটাকে বলে **Boolean Indexing** — NumPy-র সবচেয়ে শক্তিশালী কৌশল

---

## ✅ Question 03 — Pandas DataFrame [10 Marks]

**প্রশ্ন:** CSV load করো, head/info/describe দেখাও, age ফিল্টার করো।

### ধাপ ১: CSV পড়া
```python
df = pd.read_csv("school_data.csv")
```
- Starter code-এ যে `school_data.csv` সেভ হয়েছিল, সেটা পড়ে `df` (DataFrame) তৈরি করো।

### ধাপ ২: DataFrame দেখানো
```python
print(df.head(3))
df.info()
print(df.describe())
```
- **`df.head(3)`** → টেবিলের প্রথম ৩টি row দেখাও।
- **`df.info()`** → প্রতিটি column-এর নাম, data type (int/object), এবং null কতটা আছে দেখাও।
- **`df.describe()`** → সংখ্যার column গুলোর গড়, min, max, std (standard deviation) একসাথে দেখাও।

### ধাপ ৩: নির্দিষ্ট Column বাছাই
```python
filtered_df = df[['student_id', 'student_age', 'department', 'grade']]
```
- **`df[[...]]`** → double bracket দিয়ে একাধিক column বাছাই করা হয়।
- এখানে মাত্র ৪টি column নেওয়া হলো, বাকি column (score, attendance) বাদ গেল।

### ধাপ ৪: Age দিয়ে ফিল্টার
```python
age_filtered = filtered_df[
    (filtered_df['student_age'] >= 16) & (filtered_df['student_age'] <= 20)
]
```
- **দুটো condition:** age ≥ 16 **এবং** age ≤ 20 — দুটোই একসাথে সত্য হতে হবে।
- **`&`** → দুটো condition একসাথে মেলাতে `&` ব্যবহার করো (Python-এর `and` কাজ করবে না)।
- **`()`** → প্রতিটি condition আলাদা `()` এ রাখা বাধ্যতামূলক।

### ধাপ ৫: Sort ও শেষ ৩ Row
```python
sorted_df = age_filtered.sort_values(by='student_id')
print(sorted_df.tail(3))
```
- **`sort_values(by='student_id')`** → student_id ছোট থেকে বড় ক্রমে সাজাও।
- **`tail(3)`** → সাজানো তালিকার শেষ ৩টি row দেখাও।

---

## ✅ Question 04 — `.loc` দিয়ে Score Update [10 Marks]

**প্রশ্ন:** CS ও Math-এ ১০%, বাকিদের ৫% বাড়াও। সর্বোচ্চ ১০০।

### ধাপ ১: আগের Score সংরক্ষণ
```python
df['previous_score'] = df['score']
```
- পরে `score_increase` হিসাব করতে আগের মান দরকার হবে।
- তাই পরিবর্তনের আগেই `previous_score` নামে কপি রেখে দিলাম।

### ধাপ ২: CS ও Math-এ ১০% বৃদ্ধি
```python
df.loc[df['department'].isin(['Computer Science', 'Mathematics']), 'score'] = (
    df.loc[df['department'].isin(['Computer Science', 'Mathematics']), 'score'] * 1.10
)
```
এই লাইনটা বোঝার সহজ উপায়:
```
df.loc[ কোন Row চাই , কোন Column চাই ] = নতুন মান
```
- **`df['department'].isin([...])`** → department কলামের মান যদি list-এর মধ্যে থাকে তাহলে `True`।
- ডানপাশে `* 1.10` → সেই row গুলোর score ১.১০ দিয়ে গুণ করো (= ১০% বাড়ানো)।

### ধাপ ৩: বাকিদের ৫% বৃদ্ধি
```python
df.loc[~df['department'].isin(['Computer Science', 'Mathematics']), 'score'] = (
    df.loc[~df['department'].isin(['Computer Science', 'Mathematics']), 'score'] * 1.05
)
```
- **`~`** (Tilde) → NOT অর্থাৎ বিপরীত। CS ও Math **ছাড়া** বাকি সব।
- `* 1.05` → ৫% বাড়ানো।

### ধাপ ৪: সর্বোচ্চ ১০০-তে সীমাবদ্ধ করা
```python
df['score'] = df['score'].clip(upper=100)
```
- ১০% বাড়ানোর পর কেউর score ১০০ ছাড়িয়ে যেতে পারে (যেমন ৯৫ → ১০৪.৫)।
- **`.clip(upper=100)`** → ১০০-এর বেশি হলে ১০০ করে দাও, নিচে থাকলে যা আছে তাই রাখো।

### ধাপ ৫: নতুন Column ও ফলাফল দেখানো
```python
df['new_score'] = df['score'].round(2)
df['score_increase'] = (df['new_score'] - df['previous_score']).round(2)
result = df[['student_id', 'department', 'previous_score', 'new_score', 'score_increase']]
print(result.to_string(index=False))
```
- **`.round(2)`** → ২ দশমিক পর্যন্ত রাখো।
- **`score_increase`** → নতুন score - আগের score = কত বাড়লো।
- **`to_string(index=False)`** → সুন্দরভাবে print করো, বামে ০,১,২ index নম্বর দেখাবে না।

---

## ✅ Question 05 — Bar Chart [10 Marks]

**প্রশ্ন:** department অনুযায়ী গড় score-এর bar chart আঁকো।

### ধাপ ১: Group করে গড় বের করা
```python
avg_score_by_dept = df.groupby('department')['score'].mean()
```
- **`groupby('department')`** → একই department-এর সব row একসাথে গ্রুপ করো।
- **`['score'].mean()`** → প্রতিটি গ্রুপের score-এর গড় বের করো।
- ফলে প্রতিটি department-এর জন্য একটা গড় score পাওয়া যায়।

### ধাপ ২: Chart আঁকা
```python
plt.figure(figsize=(12, 6))
plt.bar(avg_score_by_dept.index, avg_score_by_dept.values, color='steelblue', edgecolor='black')
plt.title("Average Score by Department")
plt.xlabel("Department")
plt.ylabel("Average Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```
- **`figsize=(12, 6)`** → chart-এর প্রস্থ ১২ ইঞ্চি, উচ্চতা ৬ ইঞ্চি।
- **`plt.bar(x, y)`** → x-axis এ department নাম, y-axis এ গড় score।
- **`rotation=45`** → x-axis এর label ৪৫ ডিগ্রি কোণে দেখাও (না হলে লেখা overlap করবে)।
- **`ha='right'`** → label-এর ডান প্রান্ত x-axis-এর tick-এর উপর থাকবে।
- **`tight_layout()`** → সব কিছু ফিট করে নাও, কিছু কেটে না যায়।

---

## ✅ Question 06 — Histogram [10 Marks]

**প্রশ্ন:** score column-এর histogram আঁকো, ১০টি bin।

### Histogram কী?
একটা উদাহরণ দিয়ে বুঝি: ৫০ জন student-এর score আছে ৪০ থেকে ৯৯-এর মধ্যে। Histogram দেখায়:
- ৪০-৪৬ range-এ কতজন আছে?
- ৪৬-৫২ range-এ কতজন আছে?
- এভাবে ১০টি range-এ ভাগ করে দেখায়।

```python
plt.figure(figsize=(10, 6))
plt.hist(df['score'], bins=10, color='cornflowerblue', edgecolor='black')
plt.title("Distribution of Student Scores")
plt.xlabel("Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()
```
- **`plt.hist(data, bins=10)`** → score range কে ১০টি ভাগে (bins) ভাগ করো।
- X-axis এ score range, Y-axis এ সেই range-এ কতজন student আছে।

---

## ✅ Question 07 — Scatter Plot (Matplotlib) [10 Marks]

**প্রশ্ন:** attendance vs score এর scatter plot আঁকো।

### Scatter Plot কী?
প্রতিটি student একটি বিন্দু। বিন্দুর X অবস্থান = তার attendance, Y অবস্থান = তার score।
এটা দেখায় যে attendance বাড়লে score বাড়ে কি না (সম্পর্ক দেখায়)।

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['attendance'], df['score'], color='darkorange', edgecolors='black', alpha=0.7)
plt.title("Attendance vs Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Score")
plt.tight_layout()
plt.show()
```
- **`plt.scatter(x, y)`** → x-axis এ attendance, y-axis এ score।
- **`alpha=0.7`** → বিন্দু ৭০% অস্বচ্ছ। অনেক বিন্দু একই জায়গায় থাকলে overlap বোঝা যায়।
- **`edgecolors='black'`** → প্রতিটি বিন্দুর চারপাশে কালো বর্ডার।

---

## ✅ Question 08 — Seaborn Boxplot [10 Marks]

**প্রশ্ন:** grade অনুযায়ী score distribution boxplot-এ দেখাও।

### Boxplot কী?
```
         ┌──────┐
  ───────┤      ├───────  ← এই রেখা = সর্বোচ্চ/সর্বনিম্ন
         │      │
         ├──────┤  ← উপরের রেখা = ৭৫% মান (Q3)
         │  ━━━ │  ← মাঝের রেখা = Median (৫০%)
         ├──────┤  ← নিচের রেখা = ২৫% মান (Q1)
         └──────┘
```
A গ্রেড পাওয়াদের score কোথায় কোথায় আছে সেটা এক নজরে বোঝা যায়।

```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='grade', y='score', data=df, palette='Set2', order=['A', 'B', 'C', 'F'])
plt.title("Score Distribution by Grade")
plt.xlabel("Grade")
plt.ylabel("Score")
plt.tight_layout()
plt.show()
```
- **`x='grade'`** → X-axis এ grade (A, B, C, F)।
- **`y='score'`** → Y-axis এ score।
- **`palette='Set2'`** → রঙের সেট (Set1, Set2, tab10 ইত্যাদি ব্যবহার করা যায়)।
- **`order=['A','B','C','F']`** → grade-এর প্রদর্শন ক্রম নির্ধারণ (না দিলে random হতে পারে)।

---

## ✅ Question 09 — Seaborn Countplot [10 Marks]

**প্রশ্ন:** প্রতিটি department-এ কতজন student আছে।

### Countplot কী?
Bar chart-এর মতোই, কিন্তু তুমি নিজে গণনা করতে হয় না। Seaborn নিজেই প্রতিটি department-এ কতটি row আছে গণনা করে bar আঁকে।

```python
plt.figure(figsize=(12, 6))
sns.countplot(x='department', data=df, palette='viridis')
plt.title("Number of Students per Department")
plt.xlabel("Department")
plt.ylabel("Number of Students")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```
- **`sns.countplot(x='department', data=df)`** → department কলামের প্রতিটি unique মান গণনা করে bar আঁকো।
- **`palette='viridis'`** → বেগুনি থেকে হলুদ রঙের gradient থিম।

---

## ✅ Question 10 — Seaborn Scatter রঙ দিয়ে [10 Marks]

**প্রশ্ন:** attendance vs score scatter plot, department অনুযায়ী আলাদা রঙে।

### Q7 vs Q10 পার্থক্য কী?
- **Q7:** সব বিন্দু একই রঙে (Matplotlib)
- **Q10:** department অনুযায়ী আলাদা রঙে (Seaborn) + legend

```python
plt.figure(figsize=(12, 7))
sns.scatterplot(
    x='attendance', y='score',
    hue='department',
    data=df,
    palette='tab10',
    s=100,
    edgecolor='black',
    alpha=0.8
)
plt.title("Attendance vs Score by Department")
plt.xlabel("Attendance (%)")
plt.ylabel("Score")
plt.legend(title='Department', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```
- **`hue='department'`** → এটাই সবচেয়ে গুরুত্বপূর্ণ! `hue` মানে কোন column-এর ভিত্তিতে রঙ দেবে।
- **`palette='tab10'`** → ১০টি আলাদা রঙের সেট (১০টি department-এর জন্য পারফেক্ট)।
- **`s=100`** → বিন্দুর আকার (size)। বড় করলে সহজে দেখা যায়।
- **`bbox_to_anchor=(1.05, 1)`** → legend-কে chart-এর ডান পাশে একটু বাইরে রাখো।
- **`tight_layout()`** → legend সহ সব chart-এ ফিট করো।

---

## 🧠 সবচেয়ে গুরুত্বপূর্ণ ধারণাগুলো

### NumPy — মনে রাখার বিষয়
| কাজ | কোড |
|---|---|
| Array তৈরি | `np.array([...])` |
| সর্বোচ্চ/সর্বনিম্ন/গড় | `np.max()`, `np.min()`, `np.mean()` |
| Boolean filter | `arr[arr > value]` |
| Condition দিয়ে মান বদলানো | `np.where(condition, x, y)` |

### Pandas — মনে রাখার বিষয়
| কাজ | কোড |
|---|---|
| CSV পড়া | `pd.read_csv("file.csv")` |
| প্রথম/শেষ n row | `df.head(n)`, `df.tail(n)` |
| Column বাছাই | `df[['col1', 'col2']]` |
| Row filter | `df[(cond1) & (cond2)]` |
| Group করে গড় | `df.groupby('col')['score'].mean()` |
| নির্দিষ্ট row/col update | `df.loc[condition, 'col'] = value` |
| List থেকে মেলানো | `df['col'].isin([...])` |
| বিপরীত | `~df['col'].isin([...])` |
| সীমা নির্ধারণ | `df['col'].clip(upper=100)` |

### Chart — কোনটা কখন?
| Chart | কখন ব্যবহার |
|---|---|
| Bar chart | Category-র তুলনা (যেমন dept-এ গড় score) |
| Histogram | একটি সংখ্যার বিতরণ দেখা |
| Scatter plot | দুটো সংখ্যার সম্পর্ক দেখা |
| Boxplot | Category অনুযায়ী বিতরণ দেখা |
| Countplot | Category-তে কতটি item আছে গণনা |

---

> 💡 **পরামর্শ:** এই file পড়ে প্রতিটি কোড Google Colab-এ নিজে টাইপ করো। Copy-paste না করে নিজে লিখলে বেশি শিখবে। ভুল হলেও চেষ্টা করো — ভুল থেকেই বেশি শেখা যায়! 🚀
