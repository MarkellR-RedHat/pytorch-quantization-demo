# The arena


The arena is a booth demo, and I wanted it to hold up if someone who works on quantization kernels looks at it for more than ten seconds, so nothing in it is scripted.

| Bird | Weights | Hard gaps cleared* |
|---|---|---|
| BF16 | 16-bit brain float | 219 of 219 |
| FP8 | E4M3 with one scale per output row | 219 of 219 |
| INT4 AWQ | 4-bit, group size 16, activation-aware scaling | 219 of 219 |
| INT4 RTN | 4-bit, group size 16, plain round to nearest | 135 of 220 |
| Spec Decode | BF16 network plus a 4-unit draft network | same path as BF16, to the pixel |

\* Measured by `arena/train_policy.py` on 20 held-out courses. Every variant also clears every easy gap.

The birds are flown by a 6-64-64-1 MLP trained in PyTorch by imitation learning (DAgger) from a simple rule-based pilot. The four precision variants are the same trained weights quantized after training: BF16 and FP8 use PyTorch's own `torch.bfloat16` and `torch.float8_e4m3fn` casts, INT4 RTN is asymmetric round-to-nearest with one scale and zero point per group of 16 inputs, and INT4 AWQ scales each input channel by its mean activation raised to a power (searched per layer to minimize output error) before doing the same group rounding, which is the core idea of [AWQ](https://arxiv.org/abs/2306.00978). Biases stay in higher precision, as they do in W4A16 serving formats.

One input feature is fed in raw pixels while the other five are normalized, and that's deliberate. It plays the part of the outlier activation channels LLMs develop ([LLM.int8()](https://arxiv.org/abs/2208.07339), [SmoothQuant](https://arxiv.org/abs/2211.10438)), which is the reason plain INT4 rounding hurts and activation-aware scaling fixes it. If you normalize every feature, all of the INT4 variants fly perfectly, which is the same lesson from the other side.

The speculative decoding bird runs the BF16 network as the target. A 4-unit draft network guesses the next 4 moves along its own imagined path, the target scores all of those states at once, and the bird keeps the moves the target agrees with plus the target's own move at the first disagreement (or a bonus move when all four were right). That's greedy speculative decoding as described by [Leviathan et al.](https://arxiv.org/abs/2211.17192), so the committed moves are exactly the ones the target would have chosen alone. The sidebar shows the live acceptance rate, the moves per target pass, and the distance between the spec bird and a hidden BF16 bird flying the same spot, which stays at 0 px.

To check it yourself:

```bash
pip install -r arena/requirements.txt
python arena/train_policy.py      # retrains and re-exports static/arena/policy.json
pytest tests/test_arena.py        # browser engine vs PyTorch, frame for frame
```

On the same PyTorch version the retrain is byte-identical to the committed file. The tests run the JavaScript engine in Node and check that it makes the same decision as Python on every frame with zero drift, that the spec bird never leaves the BF16 path, that the INT4 weights really have at most 16 levels per group, and that the numbers quoted on screen match the exported evaluation.

The arena is a teaching model, and I'd say that out loud on stage. A 4,673-parameter policy isn't a 70B language model, and the toy draft agrees with its target about 93% of the time, which is higher than an 8B draft gets on a 70B target. What carries over is the mechanism.
