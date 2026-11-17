"""Execute Assignment 4 Questions 2 and 6-9 and check all numerical results."""

import json
from pathlib import Path
import torch
from app.diffusion_model import SinusoidalEmbedding


def run():
    results = {"embedding_t1_dim8": SinusoidalEmbedding(8)(torch.tensor([1.0]))[0].tolist()}
    x = torch.tensor([2.0], requires_grad=True)
    (x**2 + 3*x).backward()
    results["q6_gradient"] = x.grad.tolist()
    x = torch.tensor([2.0], requires_grad=False)
    try:
        (x**2 + 3*x).backward()
    except RuntimeError as error:
        results["q6_no_tracking_error"] = str(error)
    results["default_requires_grad"] = torch.tensor([2.0]).requires_grad
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([1.0, 3.0])
    (w[0]*x**2 + w[1]*x).backward()
    results["q7_original"] = {"x_grad": x.grad.tolist(), "w_grad": w.grad}
    x = torch.tensor([2.0])
    w = torch.tensor([1.0, 3.0], requires_grad=True)
    (w[0]*x**2 + w[1]*x).backward()
    results["q7_weight_gradient"] = w.grad.tolist()
    x = torch.tensor([1.0], requires_grad=True)
    y = x*3
    z = y.detach()
    try:
        (z*2).backward()
    except RuntimeError as error:
        results["q8_detach_error"] = str(error)
    z = y.clone()
    (z*2).backward()
    results["q8_fixed_gradient"] = x.grad.tolist()
    x = torch.tensor([1.0], requires_grad=True)
    (x*2).backward()
    first = x.grad.item()
    (x*3).backward()
    accumulated = x.grad.item()
    x.grad = None
    (x*3).backward()
    results["q9_gradients"] = [first, accumulated, x.grad.item()]
    assert results["q6_gradient"] == [7.0]
    assert results["q7_original"] == {"x_grad": [7.0], "w_grad": None}
    assert results["q7_weight_gradient"] == [4.0, 2.0]
    assert results["q8_fixed_gradient"] == [6.0]
    assert results["q9_gradients"] == [2.0, 5.0, 3.0]
    assert results["default_requires_grad"] is False
    Path("docs/assignment4_theory_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
