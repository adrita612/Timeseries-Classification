#Time Series Modeling with Deep Learning: Classification using Transformer and Forecasting using LSTM

This project entails a comparitive analysis between classificataions with deep learning using Transformer and Forecasting with LSTM. The baseline models showed that the transformer sequence continued to platue in accuracy and made no improvement in training. I picked the LSTM model to modify in this case. 

## Why the LSTM Model Was Chosen for Improvement

For the improvement task, the **LSTM forecasting model** was selected instead of the Transformer classification model. The LSTM workflow was easier to understand because the relationship between the **past input window** and the **future predicted value** is more direct.

Additionally, the LSTM model allows for easier experimentation with controlled changes such as:
- Hidden size  
- Input sequence length  
- Stacked recurrent layers  

This aligns with the project objective of making meaningful architecture and hyperparameter improvements to a baseline model.

---

## LSTM Improvements Made

### Part A: Increased Hidden Size

#### Change
- **Baseline:** `LSTM(32)`  
- **Modified:** `LSTM(64)`

#### Why this matters
Increasing the hidden size gives the model more capacity to learn detailed temporal patterns in the weather data. A larger hidden state allows the LSTM to retain more information from past observations, which can improve forecasting performance.

#### What I learned
- Small architectural changes can significantly impact performance  
- There is a tradeoff between model capacity and computational cost  
- Larger models may learn better representations but require more resources  

---

### Part C + B: Longer Input Window + Stacked LSTM

#### Change 1: Increased Input History Length
- **Baseline:** `past = 720`  
- **Modified:** `past = 1440`  

#### Why this matters
Doubling the input window allows the model to capture more temporal context. This is useful in weather forecasting where:
- Trends evolve slowly  
- Patterns repeat over time  

---

#### Change 2: Stacked LSTM Architecture

#### Change
- **Baseline:**
  ```python
  LSTM(32)

LSTM(64, return_sequences=True)
LSTM(32)
Dense(1)

This ensures that the model...
Learns hierarchical temporal features:
First layer → basic patterns
Second layer → higher-level dependencies
More expressive than single-layer model


#### Conclusion

This project helped me understand the difference between two major time-series deep learning approaches: Transformer-based classification and LSTM-based forecasting. The Transformer baseline was useful for showing how self-attention can classify patterns across an entire sequence, while the LSTM baseline showed how recurrent models can use past observations to predict future values.

For the improvement task, I chose the LSTM model because it was easier to interpret and modify. I made three meaningful changes across Parts A and C+B: increasing the LSTM hidden size, increasing the historical input window, and stacking two LSTM layers. Compared with the baseline, these changes made the forecasting model deeper and gave it access to more past information.

I found the **LSTM model** significantly easier to understand compared to the Transformer model.

The main reason is that the LSTM follows a very intuitive structure for time-series data:
- It takes a sequence of past values as input
- It processes them step-by-step in time
- It outputs a prediction for a future value

This clear relationship between **past → present → future** made it easier to reason about what the model was doing and how changes would affect performance.

In contrast, the **Transformer model** was more difficult to interpret. While it uses self-attention to learn relationships across the entire sequence, it is less intuitive because:
- It does not process data sequentially in the same way
- The attention mechanism is harder to visualize conceptually
- There are more interacting components (attention heads, embeddings, normalization, etc.)

Because of this complexity, it was harder to predict how modifying parameters (such as attention heads or layers) would impact the results.

Additionally, when experimenting with the Transformer model, changes did **not consistently improve performance**. The baseline Transformer already showed some instability, and tuning it did not lead to clear or reliable gains. This made it a less optimal choice for the improvement task compared to the LSTM model.

I applied three main improvements to the LSTM model:

1. **Increased hidden size** from `LSTM(32)` to `LSTM(64)`
2. **Extended the input sequence length** from `past = 720` to `past = 1440`
3. **Stacked LSTM layers** to create a deeper architecture (`64 → 32`)

Through these changes, I learned several important concepts:

- Increasing the hidden size improves the model’s ability to capture more complex temporal patterns, but also increases computational cost.
- Increasing the input sequence length provides more historical context, which helps the model learn long-term trends, especially in weather data.
- Stacking LSTM layers allows the model to learn hierarchical temporal features, making it more expressive than a single-layer model.

Overall, these improvements showed that **model structure and input design are critical in time-series forecasting**. Even relatively simple changes can significantly affect performance.

The key takeaway is that improving deep learning models is not just about increasing complexity, but about making meaningful changes that align with the structure of the data and the problem.
