# Assignment 3, Part 3: Building Blocks of GAN

**Transposed convolution output size** (used throughout, applied separately to height and width, dilation 1):

$$O = (I - 1)\cdot S - 2P + K + OP$$

where *I* = input size, *K* = kernel size, *S* = stride, *P* = padding, *OP* = output padding.

Every number below was also checked by running the same layer in PyTorch (`nn.ConvTranspose2d`, `nn.BatchNorm1d`).

---

## Question 1
*Input 8×8, transposed convolution with kernel 4×4, stride 2, padding 1, output padding 1. What is the output size?*

- I = 8, K = 4, S = 2, P = 1, OP = 1
- O = (8 − 1) · 2 − 2 · 1 + 4 + 1
- O = 7 · 2 − 2 + 4 + 1
- O = 14 − 2 + 4 + 1 = **17**

**Answer: 17 × 17.** Without the output padding it would be 16 × 16 (an exact 2× upsample). The output padding adds one extra row and one extra column on one side.

## Question 2
*How does the output size change if the stride goes from 2 to 3, everything else fixed?*

Stride multiplies (I − 1), so each +1 in stride adds (I − 1) to the output size.

- Stride 2: O = (8 − 1) · 2 − 2 + 4 + 1 = **17**
- Stride 3: O = (8 − 1) · 3 − 2 + 4 + 1 = 21 − 2 + 4 + 1 = **24**

**Answer: the output grows from 17 × 17 to 24 × 24**, an increase of (I − 1) = 7 in each dimension. In a transposed convolution, a larger stride spreads the input pixels further apart before the kernel is applied, so the output gets **bigger**. This is the opposite of a regular convolution, where a larger stride makes the output smaller. (Output padding of 1 is still valid, since it must be smaller than the stride.)

## Question 3
*Formula for the output size of a 2D transposed convolution.*

$$O = (I - 1)\cdot S - 2P + K + OP$$

For a non-square input, apply it separately to height and width:
H_out = (H_in − 1)·S − 2P + K + OP and W_out = (W_in − 1)·S − 2P + K + OP.

(The general PyTorch form also has a dilation term D: O = (I − 1)·S − 2P + D·(K − 1) + OP + 1. With D = 1 this reduces to the formula above.)

## Question 4
*A Conv2DTranspose upsamples 16×16 to 32×32 with no padding. Give a kernel size and stride.*

With P = 0 and OP = 0:  32 = (16 − 1) · S + K = 15 · S + K

Try S = 2: 32 = 30 + K, so K = 2.

Check: (16 − 1) · 2 − 2 · 0 + 2 + 0 = 30 + 2 = **32** ✓

**Answer: kernel size 2 × 2, stride 2.** Each input pixel becomes its own non-overlapping 2 × 2 block, which doubles the size exactly. (S = 1 with K = 17 also works mathematically, but it is a far less common choice.)

## Question 5
*BatchNorm on the mini-batch [6, 8, 10, 6], no γ and β.*

BatchNorm computes x̂ = (x − μ) / √(σ² + ε) using the batch mean and the batch (population, divide-by-N) variance. ε is a tiny constant (PyTorch uses 10⁻⁵) that only prevents division by zero, so it is ignored here.

**Step 1: mean**
μ = (6 + 8 + 10 + 6) / 4 = 30 / 4 = **7.5**

**Step 2: variance**

| x | x − μ | (x − μ)² |
|---|---|---|
| 6 | −1.5 | 2.25 |
| 8 | 0.5 | 0.25 |
| 10 | 2.5 | 6.25 |
| 6 | −1.5 | 2.25 |
| | **sum** | **11.00** |

σ² = 11 / 4 = **2.75**, so σ = √2.75 ≈ **1.6583**

**Step 3: normalize**

- x = 6: (6 − 7.5) / 1.6583 = −1.5 / 1.6583 ≈ **−0.9045**
- x = 8: (8 − 7.5) / 1.6583 = 0.5 / 1.6583 ≈ **0.3015**
- x = 10: (10 − 7.5) / 1.6583 = 2.5 / 1.6583 ≈ **1.5076**
- x = 6: ≈ **−0.9045**

**Answer: [−0.9045, 0.3015, 1.5076, −0.9045]**

Check: the outputs sum to 0 (zero mean), and their squares average to (0.818 + 0.091 + 2.273 + 0.818) / 4 = 4 / 4 = 1 (unit variance). ✓

## Question 6
*Key mathematical difference between ReLU and LeakyReLU; give their formulas.*

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \le 0 \end{cases}$$

$$\text{LeakyReLU}(x) = \max(\alpha x, x) = \begin{cases} x & x > 0 \\ \alpha x & x \le 0 \end{cases}, \quad \text{small } \alpha \text{, e.g. } 0.01 \text{ or } 0.2$$

**The key difference is what happens to negative inputs.** ReLU sets them to exactly 0, so its slope (gradient) there is **0**. LeakyReLU keeps a small, scaled-down version αx, so its slope there is **α** instead of 0. For positive inputs the two are identical (output x, slope 1).

Example with α = 0.2 (the value used in this assignment's discriminator): ReLU(−5) = 0, while LeakyReLU(−5) = −1.

## Question 7
*Why might LeakyReLU be preferred over ReLU in deep networks?*

1. **It avoids "dying ReLU" neurons.** If a ReLU neuron's input becomes negative for every example (for instance after a large weight update), its output and its gradient are both 0. Its weights then stop updating and the neuron is permanently dead. LeakyReLU always passes a gradient of α, so the neuron can recover.
2. **Gradients keep flowing through deep stacks.** Every layer passes back at least α of the gradient instead of 0, which helps signal reach the early layers of a deep network.
3. **It matters especially in GANs.** The generator learns *only* through the gradients coming back through the discriminator. If parts of the discriminator output zero gradient, the generator gets no information about how to improve. That is why DCGAN-style discriminators, including the one in Part 1, use LeakyReLU(0.2). (The generator usually keeps plain ReLU, as in Part 1.)
4. **It is nearly free.** It costs no more to compute than ReLU and keeps ReLU's benefits for positive inputs (no saturation, a simple gradient of 1).

The trade-off is one extra hyperparameter, α. In practice values between 0.01 and 0.2 work well.
