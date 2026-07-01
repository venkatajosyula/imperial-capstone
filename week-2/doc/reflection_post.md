# Week 2 Reflection

What was the main change in your strategy this week compared to last week? What prompted this change?
The primary change in Week 2 was transitioning from a static weighted blend to a variant adaptive to the data. In Week 1, weights of 60%, 30%, and 10% were applied uniformly using 10 point datasets. In Week 2, newly evaluated evaluation outputs were incorporated before ranking, expanding each dataset to 11 points. The change was driven by observed feedback: Functions 5 and 8 produced outputs of 655.21 and 9.82, confirming high value regions worth reinforcing, while Functions 4 and 6 yielded negative values, indicating suboptimal query placement that required reorientation.

Did you focus more on exploration or exploitation? What trade-offs did you weigh?
Week 2 favoured exploitation. With one query per function per week across a 12 round horizon, allocating queries to uncertain regions carries a high opportunity cost when strong signals already exist. For Functions 5, 7, and 8, queries were concentrated near observed maxima. For Functions 4 and 6, the blend redirected toward the least suboptimal historical points, representing constrained exploitation rather than free exploration. The principal tradeoff was premature convergence to a local maximum against the cost of delayed exploitation within a limited evaluation budget.

Have any recent outputs influenced how you approached this week's submission?
The Week 1 evaluation outputs were the sole basis for the Week 2 revision. The eight returned values demonstrated that a uniform global heuristic produces inconsistent results across functions of differing dimensionality and response surface character. This motivated a function specific approach: functions exhibiting large, stable outputs justified tighter exploitation, while functions with negative or near zero outputs warranted a broader historical draw to reorient the search.

Which linear or logistic regression assumptions would you most likely violate for one of the functions?
Applying linear or logistic regression to Function 8 (8D, 41 observations) would most likely violate the linearity assumption and the requirement for sufficient observations per predictor. The function simulates hyperparameter optimisation, where inputs interact multiplicatively rather than additively. Homoscedasticity would also be questionable given the steep gradient near the observed maximum. Under a logistic framing, feature independence would be untenable due to the correlation structure inherent in hyperparameter interactions.

Are there regions where the output appears roughly linear, or where a logistic decision boundary might form?
Local linearity is plausible within narrow neighbourhoods of the current optimum when most dimensions are held fixed. Within such subspaces, a first order approximation may yield directionally useful gradient information. Under a binary framing, with outputs above the 75th percentile treated as the positive class, a logistic boundary could achieve coarse regional separation. However, discriminative capacity would be limited by class imbalance and boundary curvature, making it insufficient for fine grained ranking of candidate queries.

Was interpretability of individual feature effects useful when deciding on your query point?
Inspection of input dimension variation across top ranked observations provided useful heuristic guidance. For Function 8, dimensions 6 and 8 showed the strongest association with high outputs, informing the decision to apply smaller perturbations to those dimensions in the Week 2 blend. While informal rather than statistically rigorous, this feature level reasoning introduced structure into query selection beyond purely mechanical averaging.
