# Assignment 2, Part 3: Arithmetic of CNNs

**Formula used throughout** (applied separately to height and width):

$$\text{output} = \left\lfloor \frac{n - k + 2p}{s} \right\rfloor + 1$$

where *n* = input size, *k* = kernel (filter) size, *p* = padding, *s* = stride.
The **depth** (number of channels) of a convolution's output always equals the **number of filters**.
A pooling layer does not change the depth.

---

## Question 1
*Input 32×32×3, convolutional layer with 8 filters of size 5×5, stride 1, no padding. What is the output size?*

- n = 32, k = 5, p = 0, s = 1
- Height = Width = (32 − 5 + 2·0) / 1 + 1 = 27 + 1 = **28**
- Depth = number of filters = **8**. (Each 5×5 filter spans all 3 input channels, i.e. it is 5×5×3, and produces one output channel.)

**Answer: 28 × 28 × 8**

## Question 2
*How does the output size change if padding is changed to "same"?*

"Same" padding adds enough zeros around the border that, with stride 1, the output keeps the input's spatial size.
For a 5×5 kernel that means p = (k − 1) / 2 = (5 − 1) / 2 = 2.

- Height = Width = (32 − 5 + 2·2) / 1 + 1 = 31 + 1 = **32**
- Depth is still **8** (8 filters).

**Answer: 32 × 32 × 8.** The spatial size grows from 28×28 to 32×32, matching the input. The depth is unchanged.

## Question 3
*A 3×3 filter with stride 2 and no padding is applied to a 64×64 input. What is the output spatial size?*

- n = 64, k = 3, p = 0, s = 2
- (64 − 3 + 0) / 2 + 1 = 61 / 2 + 1 = ⌊30.5⌋ + 1 = 30 + 1 = **31**

**Answer: 31 × 31.** The division is rounded down because the filter can't take a step that would hang off the edge.

## Question 4
*Max-pooling 2×2 with stride 2 on a 16×16 feature map. What is the output size?*

Pooling uses the same formula (no padding):

- n = 16, k = 2, p = 0, s = 2
- (16 − 2) / 2 + 1 = 7 + 1 = **8**

**Answer: 8 × 8.** Max-pooling 2×2 with stride 2 halves each spatial dimension. The number of channels is unchanged.

## Question 5
*A 128×128 image passes through two successive convolutional layers, each with a 3×3 kernel, stride 1, "same" padding. What is the output shape?*

"Same" padding for a 3×3 kernel: p = (3 − 1) / 2 = 1.

- After layer 1: (128 − 3 + 2·1) / 1 + 1 = 127 + 1 = 128
- After layer 2: (128 − 3 + 2·1) / 1 + 1 = 128

**Answer: 128 × 128 spatially, unchanged.** "Same" padding with stride 1 preserves height and width no matter how many layers are stacked.
The depth equals the number of filters in the second layer, which the question doesn't give, so the output is **128 × 128 × F₂**.

## Question 6
*In the class examples we ran `model.train()` before the training loop. What happens if you remove that line?*

`model.train()` puts the model, and every layer inside it, into **training mode** (`model.training = True`).
`model.eval()` switches it to **evaluation mode**. The mode does **not** turn gradient computation on or off (that is `torch.no_grad()`'s job), so the weights would still be updated either way.
What the mode changes is how certain layers behave:

| Layer | Training mode | Evaluation mode |
|---|---|---|
| **Dropout** | Randomly zeroes neurons (e.g. 50%) and rescales the rest, as regularization | Does nothing; all neurons are used |
| **BatchNorm** | Normalizes with the current batch's mean/variance and updates the running averages | Uses the stored running averages; does not update them |

What happens if the line is removed:

1. **On the very first run, nothing changes.** A freshly created PyTorch model is already in training mode by default.
2. **Once `model.eval()` has been called, the model stays in evaluation mode.** In the class notebook, the evaluation cell calls `model.eval()`. If you then train again (rerun the training cell, or evaluate between epochs inside the loop), the model trains in evaluation mode:
   - **Dropout is switched off**, so the regularization is lost and the network is more likely to overfit.
   - **BatchNorm uses frozen running statistics** instead of each batch's statistics and stops updating them. Training becomes less stable, and the stored statistics no longer match the weights being learned. That usually hurts accuracy.
3. **For a model with no Dropout or BatchNorm layers**, like the architecture in Part 1 (only Conv, ReLU, MaxPool and Linear), the two modes behave identically, so removing the line has no effect on the results.
   For the `EnhancedCNN` from class, which has BatchNorm and Dropout, it does matter.

That is why it's good practice to call `model.train()` at the start of every training epoch and `model.eval()` before evaluating.
