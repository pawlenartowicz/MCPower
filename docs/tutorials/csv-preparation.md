# Tutorial: Preparing Your CSV for MCPower

## Goal

You have data in Excel, Google Sheets, SPSS, or another tool, and you want to export it as a CSV file that MCPower can read without errors.

This page covers:
1. How to **export a CSV** from common tools
2. What **format rules** your CSV must follow
3. How MCPower **auto-detects** variable types from your data

---

## Creating Your CSV File

### From Excel

1. Open your spreadsheet in Excel
2. Make sure your data starts in row 1 with column headers (no blank rows above)
3. Remove any merged cells, formulas, or special formatting
4. Go to **File > Save As**
5. Choose **CSV UTF-8 (Comma delimited) (*.csv)** as the file type
6. Click **Save**

> **Tip:** If you only see "CSV (Comma delimited)" without "UTF-8", that older format usually works too, but UTF-8 is safer for special characters (accents, non-English text).

### From Google Sheets

1. Open your spreadsheet in Google Sheets
2. Make sure your data starts in row 1 with column headers
3. Go to **File > Download > Comma Separated Values (.csv)**
4. The file is saved to your Downloads folder

Google Sheets exports as UTF-8 with comma separators by default — no extra settings needed.

### From SPSS

1. Open your dataset in SPSS
2. Go to **File > Save As**
3. Set the file type to **Comma Separated Values (*.csv)**
4. Check **"Write variable names to spreadsheet"** so the header row is included
5. Click **Save**

Alternatively, use SPSS syntax:
```
SAVE TRANSLATE
  /OUTFILE="pilot_data.csv"
  /TYPE=CSV
  /ENCODING="UTF8"
  /FIELDNAMES
  /REPLACE.
```

> **Note:** SPSS may encode missing values as `.` (dot). MCPower does not accept missing values — remove or impute them before exporting (see [Step 4](#4-no-missing-values-or-empty-cells) below).

---

## Full Working Example

A correctly formatted CSV file (`pilot_data.csv`):

```text
treatment,score,age,education
1,72.5,34,high_school
0,68.1,29,college
1,81.3,41,graduate
0,65.9,26,college
1,77.2,38,high_school
0,70.4,31,graduate
1,84.6,45,college
0,62.3,24,high_school
```

Loading and using it in MCPower:

```python
import pandas as pd
from mcpower import MCPower

data = pd.read_csv("pilot_data.csv")

model = MCPower("outcome = treatment + score + age + education")
model.upload_data(data[["treatment", "score", "age", "education"]])
# treatment: 2 unique values → binary
# score: 8 unique values → continuous
# age: 8 unique values → continuous
# education: 3 unique values ["college", "graduate", "high_school"] → factor
#   Reference: "college" (first alphabetically)
#   Dummies: education[graduate], education[high_school]

model.set_effects(
    "treatment=0.50, score=0.25, age=0.10, "
    "education[graduate]=0.40, education[high_school]=0.20"
)
model.find_power(sample_size=200)
```

---

## Step-by-Step Walkthrough

### 1. Use comma separators

MCPower expects standard comma-separated values. Do not use semicolons, tabs, or pipes.

**Correct:**
```text
group,score,age
A,85.2,34
B,72.1,28
```

**Wrong:**
```text
group;score;age
A;85.2;34
B;72.1;28
```

If your file uses a different delimiter, convert it first:

```python
# Read with semicolon delimiter, then save as comma-separated
data = pd.read_csv("data.csv", sep=";")
data.to_csv("data_fixed.csv", index=False)
```

### 2. Include a header row

The first row must contain column names. These names become the variable names in your formula.

**Correct:**
```text
treatment,score,age
1,72.5,34
0,68.1,29
```

**Wrong (no header):**
```text
1,72.5,34
0,68.1,29
```

### 3. Keep types consistent within each column

Every value in a column should be the same type -- all numbers or all strings. Do not mix.

**Correct:**
```text
group,score
control,72.5
treatment,81.3
control,68.1
```

**Wrong (mixed types in the group column):**
```text
group,score
control,72.5
1,81.3
treatment,68.1
```

### 4. No missing values or empty cells

MCPower does not handle missing data. Remove or impute incomplete rows before uploading.

**Correct:**
```text
treatment,score,age
1,72.5,34
0,68.1,29
1,81.3,41
```

**Wrong (missing values):**
```text
treatment,score,age
1,72.5,34
0,,29
1,81.3,
```

**Wrong (NA or NULL markers):**
```text
treatment,score,age
1,72.5,34
0,NA,29
1,81.3,NULL
```

Clean your data in pandas before uploading:

```python
data = pd.read_csv("data_with_missing.csv")
data = data.dropna(subset=["treatment", "score", "age"])  # drop rows with NAs
```

### 5. Save with UTF-8 encoding

Save your CSV as UTF-8. pandas tries `utf-8-sig` first (MCPower does not read CSV files directly), but UTF-8 is the safest choice.

In Excel: **File > Save As > CSV UTF-8 (Comma delimited)**

In R:
```r
write.csv(data, "pilot_data.csv", row.names = FALSE, fileEncoding = "UTF-8")
```

### 6. No trailing commas

Ensure rows do not end with extra commas. This is a common issue when exporting from spreadsheet software.

**Correct:**
```text
group,score,age
A,85.2,34
B,72.1,28
```

**Wrong (trailing commas create a phantom empty column):**
```text
group,score,age,
A,85.2,34,
B,72.1,28,
```

### 7. Use simple column names

Stick to letters, numbers, and underscores. Avoid spaces, special characters, and names that start with numbers.

**Good names:** `treatment`, `score`, `baseline_pain`, `group_A`, `age`

**Problematic names:** `treatment group`, `score (%)`, `1st_measure`, `age+1`

---

## How Auto-Detection Works

MCPower classifies each column based on the count of unique values:

| Unique Values | Detected Type | Example |
|---|---|---|
| 1 | Dropped with warning | A column that is all `1` |
| 2 | Binary | `treatment` with values `0, 1` |
| 3–6 | Factor | `education` with values `"high_school", "college", "graduate"` |
| 7+ | Continuous | `age` with values `24, 26, 29, 31, 34, 38, 41, 45` |

**String columns** with 2–20 unique values are auto-detected as factors. String columns with more than 20 unique values raise an error.

If auto-detection gets it wrong, override with `data_types`:

```python
# "rating" has 5 unique values, auto-detected as factor
# Override to continuous
model.upload_data(data, data_types={"rating": "continuous"})
```

---

## Quick Checklist

Before uploading your CSV, verify:

- [ ] Comma-separated (not semicolons or tabs)
- [ ] Header row present with column names
- [ ] Consistent types per column (all numbers or all strings)
- [ ] No missing values, empty cells, `NA`, or `NULL`
- [ ] UTF-8 encoding
- [ ] No trailing commas
- [ ] Column names use only letters, numbers, and underscores

---

## Common Pitfalls

| Problem | Symptom | Fix |
|---|---|---|
| Semicolons instead of commas | All data lands in one column | Re-export with comma separator |
| Missing header | First data row used as names | Add a header row |
| Mixed types in a column | Unexpected type detection | Clean column to one type |
| Empty cells or NA values | Error during upload | Drop or impute missing rows |
| Trailing commas | Phantom empty column detected | Remove trailing commas |
| BOM encoding marker | Column name has invisible prefix | Save as UTF-8 (not UTF-8 with BOM), or use `utf-8-sig` encoding |
| Too many unique strings (>20) | ValueError on upload | Override with `data_types={"col": "continuous"}` or reduce categories |

---

## Next Steps

- **[Tutorial: Power Analysis with Your Own Data](own-data.md)** — using your uploaded data in a full analysis
- **[`upload_data()` API Reference](../api/data.md)** — complete parameter documentation
