# Week 12 Reflection

How has your optimisation strategy evolved since your first few rounds of queries? Which elements now feel more structured or systematic?

My strategy has moved from simple heuristic guessing to a much more systematic workflow. In the first rounds, I relied mostly on direct heuristics and local intuition from limited points. By Week 12, the process is more structured: I train function-specific surrogates, score candidate queries with explicit components, and use local structure cues from clusters before final selection. I also keep reproducibility controls (fixed seed, consistent formatting rules, same pipeline order), which makes the process easier to audit and repeat. The main difference is that decisions now come from a defined scoring framework rather than from one-off choices.

If you think of your current data set as a 'high-dimensional' space, which variables or behaviours seem to drive the largest variation in your results – similar to principal components in PCA?

The largest variation seems to come from three behaviours. First, function dimensionality itself strongly affects stability: low-dimensional functions are easier to model, while high-dimensional ones still show broader uncertainty. Second, distance to previously sampled points drives how much new information a query is likely to add, especially when avoiding repeats. Third, local trend quality (cluster-level mean outcomes) appears to explain a meaningful part of performance variation, because some neighborhoods repeatedly produce stronger results than others. In a PCA analogy, these factors act like dominant directions of variation in my optimisation outcomes.

How do you decide which aspects of your strategy to keep exploring versus which to reduce or simplify, as PCA reduces dimensions while retaining essential information?

I decide this by keeping components that repeatedly add useful signal and reducing components that mostly add noise or complexity. I keep surrogate prediction, distance-based exploration, and local trend cues because they each provide distinct information. I simplify by avoiding too many extra heuristics or heavy parameter tuning when the gain is not clear at this stage. In PCA terms, I try to retain the high-variance useful directions and drop low-impact complexity. This helps me stay practical while still improving decision quality round by round.

How might this round of optimisation influence your next and final round of query submission in Module 24, especially when balancing exploration and exploitation?

This round makes me more confident in a controlled balance rather than an extreme choice. For the final round, I expect to exploit strong recurring regions for functions that already show stable behavior, but I will still keep targeted exploration for functions where uncertainty remains high. I will likely adjust structure weights per function so that exploration is not uniformly applied. The main influence from Week 12 is that exploration now feels intentional and evidence-based, not random.

Reflect briefly on how insights from PCA, such as focusing on variance and removing redundancy, might apply to how you interpret BBO results.

PCA thinking helps me interpret BBO results by asking which patterns explain most of the outcome changes and which signals are redundant. Instead of treating every metric equally, I prioritize the strongest explanatory cues, such as stable local trends and distance-informed novelty. I also try to remove redundant decision signals that repeat the same information with extra complexity. This perspective makes my interpretation cleaner: focus on meaningful variance, reduce noise, and keep only the components that materially improve query decisions.