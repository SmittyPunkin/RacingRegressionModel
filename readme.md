# Racing Model - Lap Time Prediction Using Linear Regression

**Author:** Evan Smith  
**Purpose:** Explore linear regression with NumPy & Pandas on race data to predict likely race winners and lap times.

---

## Overview

This Python project demonstrates a simple machine learning pipeline that predicts lap times based on historical racing data. It leverages linear regression trained with an Adam optimizer on features such as year, driver position, and points. The model predicts absolute lap times in seconds, which are then converted back into a human-readable format.

The data parsing includes special handling of race time strings, including gaps, DNFs, and lap differences, to normalize and prepare the dataset for training.

---

## Features

- Parses race times and gaps with various formats (e.g., `1:25:32.100`, `+4.210s`, `DNF`, `+1 lap`).
- Converts all times to seconds for regression.
- Normalizes feature data for training.
- Implements linear regression with Adam optimization.
- Outputs top predicted fastest drivers with formatted lap times.
- Handles multi-year datasets and multiple races.

---

### Prerequisites

- Python 3.7 or higher
- `numpy`
- `pandas`

