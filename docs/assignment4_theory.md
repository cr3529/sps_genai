# Assignment 4 Theory Answers

Codie Rose | APANPS5560 Applied Generative AI | November 17, 2026

## Part 2 Building Blocks of a Diffusion Model

### Question 1 Sinusoidal time embedding

For an even embedding dimension d, let i = 0, ..., d/2 - 1 and let M denote the maximum period parameter. Define the frequency omega_i = M^(-2i/d). The conventional interleaved embedding is

    e_(2i)(t)   = sin(t * omega_i) = sin(t / M^(2i/d))
    e_(2i+1)(t) = cos(t * omega_i) = cos(t / M^(2i/d)).

Thus, the embedding alternates a sine and cosine at each frequency. Different frequencies describe the same timestep at different scales. The network can distinguish timesteps using a fixed-length vector. In a formula for a single zero-based dimension j, use i = floor(j/2), sine for even j, and cosine for odd j.

I use this convention in app/diffusion_model.py. Some implementations concatenate all sine coordinates and then all cosine coordinates, or reverse their order. Those are different coordinate orderings and must be stated before calculating a numerical vector. The instructor's practical also illustrates a noise-variance embedding with a different frequency range. The formula here follows the maximum-period convention explicitly specified in Question 2.

### Question 2 Eight dimensional embedding at t equals 1

Given d = 8, t = 1, M = 10000:

    2i/d = i/4, for i = 0, 1, 2, 3
    M^(2i/d) = [1, 10, 100, 1000]
    angles = [1/1, 1/10, 1/100, 1/1000]
           = [1, 0.1, 0.01, 0.001] radians.

Evaluate sine and cosine at each angle:

| i | Angle | Sine | Cosine |
|---|---:|---:|---:|
| 0 | 1 | 0.841470985 | 0.540302306 |
| 1 | 0.1 | 0.099833417 | 0.995004165 |
| 2 | 0.01 | 0.009999833 | 0.999950000 |
| 3 | 0.001 | 0.001000000 | 0.999999500 |

In the interleaved order defined in Question 1:

    e(1) = [0.841470985, 0.540302306,
            0.099833417, 0.995004165,
            0.009999833, 0.999950000,
            0.001000000, 0.999999500].

The last sine value rounds to 0.001000000 at nine decimal places. Running verify_theory.py reproduces the vector in float32; differences in the last digits reflect floating-point precision.

### Question 3 Relation to transformer positional encoding

Both techniques transform a scalar index into a vector of sine and cosine values at several frequencies. The mathematical construction can be identical. Transformer positional encoding represents where a token is in a sequence and is commonly added to its token embedding before attention. A diffusion time embedding represents the noise level or diffusion timestep of the entire image. It conditions the denoising network so the network knows how much corruption to remove. In my UNet, a learned projection of the time embedding is added to feature maps in each residual block. The key difference is what the index means and where the embedding is used: sequence position versus the stage of denoising.

### Question 4 Spatial resolution at the bottleneck

Assuming each stride-2 block halves height and width with the usual matching padding:

    64 x 64 -> 32 x 32 -> 16 x 16 -> 8 x 8
    bottleneck side length = 64 / 2^3 = 64 / 8 = 8.

Answer: 8 x 8. The number of channels can change independently. The implementation trains at CIFAR-10's native 32 x 32 resolution, giving a 4 x 4 bottleneck after its three downsampling blocks. Question 4 specifically uses a hypothetical 64 x 64 input.

### Question 5 UNet output and loss

For the noise-prediction parameterization used here, the UNet receives a noisy image x_t and a timestep t. It outputs epsilon_theta(x_t, t), an estimate of the noise added to that image. Its output has the same shape as x_t, including all image channels.

Let beta_t be the variance schedule, alpha_t = 1 - beta_t, and alpha_bar_t the product of alpha_s from s = 1 to t. Sample a clean image x_0, a timestep t, and independent noise epsilon ~ N(0, I). Construct

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.

Because epsilon is known during training, compute the mean squared error

    L = mean over batch, channels, height, width of
        (epsilon_theta(x_t, t) - epsilon)^2.

Backpropagate this loss to update the UNet weights. A clean-image estimate can then be recovered using

    x0_hat = (x_t - sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t))
             / sqrt(alpha_bar_t).

The sampler uses these estimates in a sequence of reverse diffusion steps. Predicting the clean image directly is an alternative parameterization, but it is not the one used in this implementation.

## Part 3 Building Blocks of an Energy Model

### Question 6 Basic gradient calculations

The supplied function is y = x^2 + 3x, with x = 2.

**a. Expected gradient**

    dy/dx = 2x + 3
    at x = 2: dy/dx = 2(2) + 3 = 7.

The supplied program prints:

    x.grad = tensor([7.])

**b. Setting requires_grad=False**

Neither x nor any other input requires gradients. Therefore y has no gradient history, and y.backward() raises RuntimeError:

    element 0 of tensors does not require grad and does not have a grad_fn

No gradient is stored for x; x.grad remains None. This is an error on the backward call, not a numerical gradient of zero.

**c. Default behavior**

The documented default is requires_grad=False in torch.tensor. Gradients are not tracked for a newly created tensor unless this flag is enabled. An operation can still be tracked if another input requires gradients, as in Question 7, but that does not make every input's .grad populate automatically.

### Question 7 Introduce weights

The function is y = w[0] * x^2 + w[1] * x, with x = 2 and w = [1, 3].

**a. Printing the original weight gradient**

The supplied code enables gradients for x but constructs w with the default requires_grad=False. The result is:

    x.grad = tensor([7.])
    w.grad = None

The weights participate in the numerical calculation, but PyTorch is not asked to compute their gradients. None means no gradient was accumulated, rather than a gradient of zero.

**b. Modified code and derivation**

Hold x fixed and enable gradients for w:

```python
import torch

x = torch.tensor([2.0])
w = torch.tensor([1.0, 3.0], requires_grad=True)
y = w[0] * x**2 + w[1] * x
y.backward()
print("w.grad =", w.grad)
```

The partial derivatives are

    dy/dw[0] = x^2 = 2^2 = 4
    dy/dw[1] = x = 2
    gradient_w y = [4, 2].

The program prints:

    w.grad = tensor([4., 2.])

If gradients are wanted for both x and w, construct both tensors with requires_grad=True. The same backward call then gives x.grad = tensor([7.]) and w.grad = tensor([4., 2.]).

**c. Default behavior**

As in Question 6, torch.tensor defaults to requires_grad=False. Explicitly enabling it on w is what makes weight-gradient accumulation happen.

### Question 8 Breaking the graph

The first operations form the graph x -> y = 3x. The statement z = y.detach() creates a tensor sharing y's data but disconnected from its autograd history. With w = 2z, no input to w requires gradients, so w.backward() raises the same RuntimeError as in Question 6.

To keep z while preserving the path back to x, use a differentiable clone:

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.clone()
w = z * 2
w.backward()
print("x.grad =", x.grad)
```

The chain rule gives

    dw/dx = (dw/dz) * (dz/dy) * (dy/dx)
          = 2 * 1 * 3 = 6.

Output:

    x.grad = tensor([6.])

Using z = y also works. In contrast, y.detach().requires_grad_(True) makes z a new leaf. It allows computing z.grad but does not reconnect z to x, so it does not solve the requested problem.

### Question 9 Gradient accumulation

The first graph is y1 = 2x, so dy1/dx = 2. After its backward call:

    After first backward: x.grad = tensor([2.])

The second graph is y2 = 3x, so dy2/dx = 3. PyTorch adds this gradient to the existing .grad buffer:

    x.grad = 2 + 3 = 5
    After second backward: x.grad = tensor([5.])

This accumulation is intentional and can be useful across mini-batches. For independent calculations, clear the previous gradient before the next backward call:

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
y1 = x * 2
y1.backward()
print("After first backward: x.grad =", x.grad)

x.grad = None  # alternatively: x.grad.zero_()
y2 = x * 3
y2.backward()
print("After second backward: x.grad =", x.grad)
```

Corrected output:

    After first backward: x.grad = tensor([2.])
    After second backward: x.grad = tensor([3.])

In neural-network training, optimizer.zero_grad(set_to_none=True) clears the gradients of parameters managed by the optimizer. For an input tensor outside the optimizer, clear its gradient directly or use a freshly detached input leaf. The two original backward calls use different computation graphs sharing only the leaf x, so retain_graph=True is unnecessary.

## Verification and sources

All numerical gradients, expected errors, and the embedding vector were executed in PyTorch with verify_theory.py. The captured outputs are in docs/assignment4_theory_results.json.

- Assignment 4 PDF supplied with the course assignment.
- Module 8 Energy Based and Diffusion Models lecture slides, Fall 2026.
- Instructor's energy practical: https://gurgentus.github.io/applied_genai_notebooks/Module%208/practical_1_energy_based/
- Instructor's diffusion practical: https://gurgentus.github.io/applied_genai_notebooks/Module%208/practical_2_diffusion/
- PyTorch torch.tensor reference, requires_grad default: https://docs.pytorch.org/docs/2.14/generated/torch.tensor.html
- PyTorch Tensor.detach reference: https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html
- PyTorch automatic differentiation tutorial: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

Implementation and written explanations prepared with assistance from OpenAI Codex. Training, code execution, and deployment checks were performed on the submitted project.
