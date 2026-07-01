# Week 9 Reflection

How do scaling laws influence your current query choices? Do you see diminishing returns or steady improvements?

My query choices are starting to reflect diminishing returns rather than steady gains. The jump from 17 to 18 cumulative points helped keep the surrogate stable, but it did not suddenly change the landscape of the problem. For the lower-dimensional functions, the model is already close to saturation, so extra points mostly sharpen confidence rather than change the best query. For the higher-dimensional functions, each added point still matters, but the improvement is uneven and noisy. That makes me treat scaling laws as a warning against overclaiming: more data helps, but not equally everywhere, and not enough to justify assuming the next round will be much better by default.

Where might emergent behaviours alter your expectations, and how are you preparing for them?

I would expect emergent behaviour to appear first in the higher-dimensional functions, where the surrogate may miss sharp transitions or coordinated effects across variables. That could make a query look safe under the model but behave very differently in the real black-box function. I am preparing for that by keeping a search component that is not purely exploitative. The random candidate stage, nearest-neighbour distance check, and gradient refinement give me multiple views of the same candidate. If those signals disagree, I treat that as a sign that the local model may be hiding something unusual.

What trade-offs between cost, robustness and performance are shaping your strategy now?

Cost is keeping the strategy grounded. I am using a fixed architecture, limited candidate sampling, and a simple refinement loop because the weekly setting does not justify expensive tuning or repeated retraining. Robustness matters more than chasing the absolute best predicted value, so I prefer queries that are both promising and not too close to previous samples. Performance still matters, but I now see that a query with slightly lower predicted output but better coverage can be the smarter choice if it reduces the chance of repeating a misleading local optimum. The main trade-off is that the more I protect against risk, the less aggressively I can exploit the surrogate’s best prediction.

How do you balance predictable optimisation with the risk of sudden but uneven emergent capabilities?

I balance that by separating the search into two layers. The first layer is predictable optimisation: train the surrogate, score random candidates, and refine the most promising starts. The second layer is a guardrail against surprise: check distance from observed points, compare gradient direction against the candidate score, and avoid trusting a single signal too much. That does not eliminate risk, but it makes the process less brittle. If an emergent effect exists, I am more likely to notice it as instability between model confidence, geometry, and observed history rather than as a false sense of certainty.

How does this change the way you think about the next rounds of the capstone?

It makes me less interested in one global rule and more interested in adaptive behaviour. A single strategy can work well for the smooth, easier functions, but the harder ones probably need different exploration pressure, different regularisation, or a more explicit uncertainty estimate. The main lesson from Round 9 is that optimisation is still useful, but it should be treated as a controlled heuristic, not as proof that the next query will behave consistently. That is the mindset I want to keep as the problem scales.