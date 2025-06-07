# MCPower

**Monte Carlo Power Analysis for Complex Linear Models**

MCPower provides flexible, simulation-based power analysis for linear models using R-style formula syntax. Perfect for researchers who need to analyze power for complex designs with interactions, multiple predictors, and non-normal distributions.

## ✨ Key Features

- **R-style formulas**: `"y = x1 + x2 + x3*m"` syntax for intuitive model specification
- **Flexible distributions**: Normal, skewed, high-kurtosis, binary, and uniform variables
- **Monte Carlo simulation**: Robust power estimates for complex models
- **Multiple testing**: Analyze power for individual effects and overall model
- **Sample size planning**: Automatic sample size determination with power curves

## 🚀 Quick Start

### Installation
```bash
# Install from GitHub
pip install git+https://github.com/pawlenartowicz/MCPower
```

### Basic Usage
```python
from mcpower import MCPower

# 1. Define your model with R-style formula
model = MCPower("y = x1 + x2 + x3*moderator")

# 2. Set up the analysis
model.set_sample_size(100)
model.set_effects("x1=0.5, x2=0.3, x3=0.2, x3:moderator=0.4")
model.set_variable_type(moderator=("binary", 0.3))  # 30% probability

# 3. Run power analysis
power_results = model.find_power("all")  # Test all effects
```

## 📊 Examples

### Power Analysis for Interaction Effects
```python
# Model with moderation
model = MCPower("satisfaction = treatment + motivation + treatment*motivation")
model.set_sample_size(150)
model.set_effects("treatment=0.6, motivation=0.4, treatment:motivation=0.5")
model.set_variable_type(treatment=("binary", 0.5))

# Test specific interaction
power_result = model.find_power("treatment:motivation")
```

### Sample Size Planning
```python
# Find required sample sizes
model.set_effects("treatment=0.3, treatment:motivation=0.2")
sample_results = model.get_sample_size(
    target_test="treatment:motivation",
    from_size=50, 
    to_size=300, 
    by=10
)
# Automatically plots power curves and identifies minimum N
```

### Multiple Variable Types
```python
model = MCPower("outcome = age + gender + treatment + age*treatment")
model.set_variable_type(
    age="normal",                    # Continuous normal
    gender=("binary", 0.6),         # 60% female
    treatment=("binary", 0.5)       # 50% treatment group
)
model.set_effects("age=0.3, gender=0.4, treatment=0.5, age:treatment=0.2")
```

## 🎯 Core Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `set_sample_size(n)` | Set sample size | `model.set_sample_size(100)` |
| `set_effects("x1=0.5, x2=0.3")` | Define effect sizes | `model.set_effects("treatment=0.6")` |
| `set_variable_type()` | Variable distributions | `model.set_variable_type(x1="binary")` |
| `find_power("x1, x2")` | Power analysis | `model.find_power("all")` |
| `get_sample_size()` | Sample size planning | `model.get_sample_size("treatment")` |

## 📈 Distribution Types

- **`"normal"`** - Standard normal (default)
- **`"right_skewed"`** - Right-skewed distribution
- **`"left_skewed"`** - Left-skewed distribution  
- **`"high_kurtosis"`** - High-kurtosis distribution
- **`"uniform"`** - Uniform distribution
- **`"binary"`** - Binary (0/1) with custom proportion


## 🛠️ Google Colab Ready

```python
# Install and run in one cell
!pip install git+https://github.com/yourusername/mcpower.git

from mcpower import MCPower
model = MCPower("y = treatment + baseline + treatment*baseline")
# ... analysis code ...
```

## 📚 Advanced Features

- **Custom test formulas**: Generate data with one model, test with another
- **Seed control**: Reproducible simulations
- **Progress tracking**: Monitor long-running sample size analyses

## 🤝 Contributing

MCPower is under active development. Feature requests and contributions welcome!

## 📄 License

Not yet licensed

---