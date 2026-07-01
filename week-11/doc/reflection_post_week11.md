# Week 11 Reflection

How have patterns in your past queries influenced your latest choices?

Patterns from earlier rounds strongly shaped my Week 11 choices. I noticed that lower-dimensional functions were becoming more stable, so I allowed more targeted exploitation in those spaces when the surrogate and local neighborhood signals agreed. For higher-dimensional functions, my previous rounds showed higher variance and less consistent generalisation, so I kept stronger distance-based exploration. In simple terms, I used history as a guide for confidence calibration: stable functions got more focused queries, while unstable functions kept broader movement.

Have you identified any 'clusters' or recurring regions in your search space that seem promising? (Explain your reasoning – whether you have identified such regions or not.)

Yes, I identified recurring regions that appear promising for several functions. In Week 11, I used clustering to group historical points into local neighborhoods and then compared cluster-level average outcomes. For functions like F5, I repeatedly saw high values around upper-bound coordinates, which appeared as a strong-performing cluster rather than isolated random hits. For other functions, especially where outputs are mostly negative or flat, I found weaker or less separated clusters. In those cases, I treated cluster signals as suggestive rather than conclusive and relied more on balanced scoring that still keeps exploration active.

Which strategies or parameter choices have proven less effective, and how are you adjusting for them?

Two choices were less effective over time. First, using one fixed exploration intensity for every function did not work equally well, because function landscapes differ in smoothness and output scale. Second, relying only on surrogate prediction without local structure cues sometimes produced overconfident selections in sparse high-dimensional areas. My adjustment in Week 11 was to blend three cues: surrogate prediction, distance from observed points, and cluster trend quality. I also kept the nearest-centroid distance as a practical similarity check, so I can explain whether a query is intentionally near a known region or deliberately moving away.

In what ways do your refinements parallel how clustering algorithms separate meaningful patterns from noise?

My refinements parallel clustering logic by treating the search history as mixed signal plus noise and then focusing on persistent local structure. Clustering separates points into groups based on similarity; similarly, my Week 11 process tries to distinguish stable high-performing neighborhoods from random one-off outcomes. The distance cues help avoid overreacting to isolated noisy points, while cluster-level trend summaries reduce sensitivity to individual outliers. This does not remove noise completely, but it helps me make decisions from recurring structure instead of from single extreme observations.

If your query results were plotted, what trends or groupings might appear? How could these inform your next iteration?

If plotted, I expect to see tighter groupings for lower-dimensional functions around regions with gradually improving outcomes, and more dispersed point clouds for higher-dimensional functions where uncertainty is still larger. I would also expect a few clusters with consistently better historical means, especially in functions where surrogate confidence is improving. For the next iteration, these plots would guide where to push deeper versus where to diversify. Specifically, I would exploit clusters with repeated strong performance but keep a controlled fraction of queries allocated to less explored regions to avoid cluster lock-in and to reduce the risk of missing better basins.