# Day-1 Setup Notes — Windows + VS Code + Conda

## লক্ষ্য

* আলাদা, পরিষ্কার Python **environment** তৈরি করা (msc-ai)
* VS Code-এ Notebook চালানো
* PowerShell-এর স্ক্রিপ্ট ব্লক সমস্যা ঠিক করা
* কাজের ফাইলগুলো চালিয়ে “Day 1” শুরু করা

---

## 1) মৌলিক ধারণা (Why Conda? Why new env?)

**Environment = আলাদা ঘর।**
প্রতিটা প্রজেক্টের জন্য আলাদা “ঘর” রাখলে প্যাকেজ-ভার্সন কনফ্লিক্ট হয় না।
**Conda** সেই ঘর বানায়/মেইনটেইন করে, **pip** ঘরের ভেতর লাইব্রেরি আনে।

* **Anaconda ইনস্টল** করলে কেবল একটি ডিফল্ট env থাকে: **base**
* আমরা প্রজেক্টের জন্য বানালাম নতুন env: **msc-ai**
* নতুন env খালি থাকে → তাই NumPy, Pandas ইত্যাদি **আলাদা করে** ইনস্টল হয়েছে
* env-গুলো সাধারণত থাকে: `C:\ProgramData\anaconda3\envs\...`
  (প্রজেক্ট ফোল্ডারের ভেতরে env রাখা হয় না — এটিই স্বাভাবিক)

---

## 2) ইনস্টল ও যাচাই (Recap)

* **Anaconda Install** (Windows 64-bit)
* **Conda যাচাই:**

  ```bat
  conda --version
  ```
* **Conda কোথায় আছে:**

  ```bat
  where conda
  ```

  আউটপুটের মতো:

  ```
  C:\ProgramData\anaconda3\Scripts\conda.exe
  C:\ProgramData\anaconda3\condabin\conda.bat
  ```

---

## 3) PowerShell সমস্যা ও সমাধান

### সমস্যা

PowerShell ডিফল্টে **Restricted execution policy** রাখে, ফলে `profile.ps1` বা conda-র init স্ক্রিপ্ট লোড হতে দেয় না। তাই `conda` কমান্ড চলছিল না।

### সমাধান

(শুধু CurrentUser স্কোপে, সিস্টেম-ওয়াইড নয়)

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

* **RemoteSigned**: নিজের PC-র লোকাল স্ক্রিপ্ট চলবে; ইন্টারনেট থেকে নামানো স্ক্রিপ্ট হলে সিগনেচার লাগবে।
* এরপর conda-কে PowerShell/CMD-এ সক্রিয় করা:

```powershell
conda init powershell
conda init cmd.exe
```

এতে PowerShell-এর `profile.ps1`-এ conda hook বসে, টার্মিনাল খোলার সাথে `(base)` দেখা যায় এবং `conda` কমান্ড চেনে।

> চাইলে VS Code-এ **Terminal → Select Default Profile** থেকে **Command Prompt (cmd.exe)** বেছে নিলেও কাজ চলে—তখন execution policy নিয়ে ভাবতে হয় না।

---

## 4) প্রজেক্ট env তৈরি

```bat
conda create -n msc-ai python=3.10 -y
conda activate msc-ai
pip install numpy pandas matplotlib scikit-learn jupyter
```

**যাচাই:**

```bat
python -c "import numpy as np, pandas as pd; print('OK', np.__version__, pd.__version__)"
```

---

## 5) VS Code সেটআপ

1. **Extensions**: *Python* (Microsoft), *Jupyter* (Microsoft)
2. **Folder Open**: `D:\Projects\AI-ML-DeepLearning`
3. **Notebook Open**: `day1_starter.ipynb`
4. **Kernel/Interpreter**: উপরে **Select Kernel → Python 3.10 (msc-ai)**
5. প্রথম সেল রান → “NumPy, Pandas ready!” আউটপুট

---

## 6) Day-1 ফাইল (স্টার্টার কিট)

* `day1_starter.ipynb` — NumPy রিক্যাপ + Titanic fallback লোড + প্লট
* `titanic_sample.csv` — স্যাম্পল ডেটা (আসল Titanic না থাকলেও চলবে)
* `daily_log_template.md` — দৈনিক নোট/রিপোর্ট টেমপ্লেট

---

## 7) কমান্ড লগ (Chronological)

```text
# Anaconda install (GUI)

where conda
conda --version

# PowerShell unblock (only once)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Shells enable
conda init powershell
conda init cmd.exe

# New project env
conda create -n msc-ai python=3.10 -y
conda activate msc-ai
pip install numpy pandas matplotlib scikit-learn jupyter

# VS Code: select kernel = msc-ai
```

---

## 8) কেন env প্রজেক্ট ফোল্ডারের ভিতরে নেই?

Conda env সাধারণত একটি কেন্দ্রীয় লোকেশনে থাকে (`.../anaconda3/envs/msc-ai`)।
কারণ:

* একই env **বহু প্রজেক্টে** ব্যবহার করা যায়
* প্রজেক্ট ফোল্ডার হালকা থাকে
* Git রিপোতে ভার্চুয়াল এনভ মেশানো লাগে না

**শেয়ার/ব্যাকআপের উপায় (Best Practice):**
প্রজেক্টে **environment.yml** দিয়ে দাও—

```bat
conda env export --name msc-ai --no-builds > environment.yml
```

অন্য মেশিনে:

```bat
conda env create -f environment.yml
conda activate msc-ai
```

---

## 9) Git (শুরু করার জন্য)

```bat
cd D:\Projects\AI-ML-DeepLearning
git init
git add .
git commit -m "Day 1: env + VS Code + starter notebook"
```

**.gitignore** (প্রস্তাব):

```
.ipynb_checkpoints/
__pycache__/
*.pyc
.env
.vscode/
```

---

## 10) Troubleshooting Cheatsheet

* **VS Code-এ conda কাজ করে না** → Terminal profile **Command Prompt** করো / `conda init powershell` / execution policy RemoteSigned
* **Kernel দেখা যাচ্ছে না** → `Ctrl+Shift+P → Python: Select Interpreter → msc-ai`
* **Jupyter not found** → `pip install jupyter` (msc-ai অ্যাক্টিভ)
* **Permission/Script error** → PowerShell-এ উপরের execution policy কমান্ড

---

## 11) পরবর্তী কাজ (Day-1 টাস্ক শুরু)

**NumPy Recap (Notebook):**

* `a.shape`, `b.shape` (1D vs 2D)
* `np.dot([1,2,3],[4,5,6])`
* Broadcasting: `np.arange(12).reshape(3,4) + [1,0,-1,2]`

**Practice:**

* ভেক্টর normalize
* `np.random.rand(100)` → mean/variance

**Titanic Intro:**

* `df.info()`, `df.describe()`
* female count, average age, missing age
* প্লট: age histogram, survived bar (Matplotlib)

---

## 12) কয়েকটি সোনার টিপস

* **এক env = এক উদ্দেশ্য**: thesis-এর জন্য `msc-ai` যথেষ্ট
* **Conda update** (চাইলে পরে):

  ```bat
  conda update -n base -c defaults conda
  ```
* **Notebook + Script**—দুটোতেই কাজ শেখো; ভারী কোড হলে `.py` তে নিয়ে যাও
* **রিপোর্টিং অভ্যাস**: প্রতিদিন `daily_log_template.md` আপডেট করো

---

এগুলোই আজ পর্যন্ত সম্পূর্ণ সেটআপ ও ব্যাখ্যার নোট।
এবার নোটবুকে **NumPy Recap** তিনটা ছোট টাস্ক রান করে আউটপুট শেয়ার করলেই আমরা পরের ধাপে যাব।
