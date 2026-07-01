# Week 4 Reflection

In your function evaluations, which inputs seemed to act like support vectors - points near a decision boundary or region of rapid change? How might recognising them guide your next query?
In my results, support-vector-like inputs were points near sudden prediction changes in the surrogate. I saw this most in F5, F7, and F8. For F5, high dim3/dim4 values looked close to a boundary between medium and high predictions. For F7, changes in dim3, dim5, and dim6 caused sharper shifts. For F8, dim3 and dim1 had the strongest local effect. Recognising these regions helps me place the next query where both improvement and learning are likely.

If you trained a neural network or another surrogate model, did you explore how the outputs change in response to the inputs? How might these gradients point to directions that reduce the function value? If you did not train a neural network or surrogate model, explain why you chose not to.
Yes. I trained a neural network surrogate for each function and explored output sensitivity using numerical gradients. I used gradients for query refinement: positive direction for maximisation, negative direction if the goal is minimisation. So gradients directly show where to move in input space.

Imagine framing your BBO capstone project as a classification task ('good' vs 'bad' outputs). How could models such as logistic regression, SVMs or neural networks capture this decision boundary? What trade-offs would you face between misclassification and exploration?
I can label points above a threshold as good and below it as bad. Logistic regression gives a simple linear boundary, SVM gives a margin-based boundary (and can be non-linear with kernels), and a neural network can learn more complex curved boundaries. The trade-off is that confident classification can over-push exploitation, while misclassification near the boundary can send queries away from promising zones. I still need a small exploration budget.

Which type of model - linear regression, SVM or neural network - felt most appropriate for guiding your search? How did you balance interpretability against flexibility when making this choice?
For this round, the neural network felt most appropriate because several functions, especially high-dimensional ones, looked non-linear. Linear models are easier to interpret but seemed too limited. SVMs are useful for boundaries, but I needed gradient-based directions for query updates, which the NN provided more directly. I balanced interpretability and flexibility by interpreting gradient importance while using the NN for modeling power.

Looking at your neural network surrogate, which input variables showed the steepest gradients or the greatest influence on your predictions? How might you use this to prioritise your next experiments?
The steepest gradients were:
- F5: dimensions 4 and 3 (approximately 241.99 and 234.05).
- F7: dimensions 3, 5, and 6.
- F8: dimensions 3 and 1.
I will prioritise future experiments by changing these high-influence dimensions more and keeping lower-influence ones more stable. This should improve sample efficiency with a limited query budget.

When framing your BBO problem as a classification task ('good' vs 'bad' outputs), how effectively did your neural network approximate the decision boundary? In what ways did backpropagation help you interpret or visualise this boundary?
In my experience, the NN gave a reasonable boundary approximation, especially in higher-dimensional functions where simple boundaries are unlikely. It was not perfect, but it was useful for guiding query movement. Backpropagation-style sensitivity (via gradients) helped me interpret the boundary locally by showing which direction improves predicted quality and which variables move the boundary most.

Compared to simpler models such as linear or logistic regression, how well did your neural network capture non-linear patterns in the function? Was the added flexibility worth the extra complexity in tuning and interpretation?
The neural network captured non-linear patterns better than linear/logistic models where interaction effects seemed strong. For this round, the flexibility was mostly worth it because it improved directional guidance for query updates. The downside was extra tuning effort and uncertainty in some functions (for example higher CV errors in F4/F5). My takeaway is to keep using NNs, but with careful regularisation and validation as data grows.
