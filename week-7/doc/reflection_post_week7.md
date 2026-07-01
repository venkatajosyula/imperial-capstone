# Week 7 Reflection

Which hyperparameters did you choose to tune, and why did you prioritise them?

I decided to focus on three key hyperparameters: network depth (two hidden layers), layer width scaled by input dimension, and L2 regularisation (alpha=1e-3). The two-layer architecture mirrors Module 16's concept of CNNs building progressively abstract features. Keeping it shallow helps with 16 observations, but a single layer might miss dimension interactions. Width scaling means higher dimensions automatically get more capacity without manual per-function decisions. I kept learning rate and iteration count as defaults because training completes quickly at this data size.

How has hyperparameter tuning changed your query strategy compared to earlier rounds?

Early rounds used fixed blending heuristics and SVM classification with baked-in hyperparameters and no adaptation. Round 4 introduced neural networks but with defaults. By Round 5, I began scaling width by dimensionality based on observing that higher dimensions need more parameters. Round 7 continues this, but now more deliberately: encoding prior knowledge about capacity into architecture rather than using off-the-shelf defaults. My queries depend on what the surrogate learns rather than my assumptions. When gradient analysis for F5 reveals dimensions 3 and 4 have much larger magnitudes, that insight comes from the model, not guessing.

Which tuning method(s) did you apply (manual adjustment, grid search, random search, Bayesian optimisation, Hyperband), and what trade-offs did you notice?

I used primarily manual adjustment rather than formal grid search or Bayesian optimisation. Width scales by dimension; alpha is fixed at 1e-3 globally. This is efficient but potentially suboptimal. A grid search over alpha [1e-4, 1e-3, 1e-2] might reveal F1 prefers stronger regularisation while F5 prefers weaker. However, at 16 points with one-query-per-round, the cost of per-function tuning outweighs likely marginal gains. I also sampled 5000 candidate points then refined top starts via gradient ascent, a hybrid approach balancing exploration and exploitation without formal framework. That felt practical given my constraints.

As your data set grows to 16 points, what limitations of your model become clearer through tuning?

Both training and cross-validation error show diminishing returns, suggesting the surrogate is hitting limits. F1 remains near-flat across all points, meaning my model structure cannot extract signal regardless of tuning. F5 and F4 show very high cross-validation error despite low training error, classic overfitting signs even with L2 regularisation. This worries me: 16 points may be insufficient for reliable generalisation in 4+ dimensions. I also assumed functions are smooth enough for gradient refinement, but F1's flatness reveals this breaks down.

How might you apply hyperparameter tuning techniques to larger data sets in future rounds of the BBO capstone project submissions or more complex models in future ML/AI projects?

With 30+ points per function, I would run grid search over alpha [1e-4, 1e-3, 1e-2] and layer widths using 5-fold cross-validation per function. For ensembles, I would tune component diversity. As data grows, I would explore adaptive regularisation adjusting alpha based on validation trends. For image or text models, tuning would expand to batch size, learning rate schedules, and augmentation intensity.

How does tuning in this black-box set-up prepare you to think like a professional ML/AI practitioner in real-world contexts with incomplete information?

This exercise taught me that perfect tuning is rarely possible without large budgets or domain knowledge. Real practitioners make hard trade-offs: tune high-impact hyperparameters or deploy quickly and learn. I chose to stabilise width and regularisation rather than exhaustively search, mirroring how practitioners actually triage efforts. I also learned that sometimes model limitations matter more than tuning. F1's persistent near-zero outputs suggest my strategy is fundamentally misaligned, not just poorly tuned. Professional maturity means knowing when tuning yields diminishing returns and when to switch strategy entirely. Early prototypes use defaults, mature models get tuned, production systems freeze hyperparameters for reproducibility. The real skill is disciplined trade-off thinking, not perfectionism.
