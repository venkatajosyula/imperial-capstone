# Week 8 Reflection

Which prompt patterns (zero-shot, few-shot, etc.) did you use, and why? What changed when you simplified vs structured the prompt?

I mainly used a structured zero shot pattern. Instead of giving examples to imitate, I gave strict constraints: one query per function, exact dimension, six decimals, and values in [0, 0.999999]. This worked better for me than a few-shot style because format reliability mattered more than creativity. When prompts were open, outputs were more verbose and less consistent. When prompts were structured as a checklist, outputs were easier to validate and integrate into my scripts.

What temperature, top-p, top-k and max-tokens settings did you choose? How did they trade off coherence vs diversity? How did they affect your chosen query?

For final Round 8 query generation, I relied on deterministic Python scripts, so sampling settings were not the direct driver of the submitted values. This improved coherence and reproducibility, but reduced linguistic diversity. Diversity came from random candidate generation in the optimisation script (5000 points) and gradient refinement of top candidates. If I used LLM generation more directly for candidates, I would keep low randomness for schema-critical output and get diversity by running multiple constrained drafts.

Did token boundaries or unusual input strings affect the model’s behaviour? When did you notice token count limits or truncation influencing the outputs? If no such cases were observed, explain how you checked for those cases.

I did not observe serious token boundary effects in final queries because submission lines came from scripts, not long free-form generations. I still checked for truncation-like failures using automated validation: regex for 0.XXXXXX format, exact component count per function, and numeric bounds. Any malformed line would fail immediately. I also kept instruction text focused to reduce long-context drift.

With 17 data points, what limitations did you encounter, such as prompt overfitting, attention focusing on irrelevant context or diminishing returns from longer inputs?

At 17 points, I saw diminishing returns from longer instructions. Extra explanation did not materially improve final query quality once constraints and data flow were stable. I also noticed that long prompts can shift attention toward secondary details instead of core schema requirements. On the modelling side, high-dimensional functions still show unstable cross-validation error, which suggests prompting improvements alone cannot fully address limited-data generalisation.

Which strategies did you try to reduce hallucinations? For example, did you use tighter instructions, retrieval of prior relevant information or constraint in output format?

I used three practical controls. First, tighter instructions with explicit format and bounds. Second, retrieval of prior relevant artefacts: previous-week scripts and cumulative data lineage. Third, hard output constraints through automated validation before accepting results. This pipeline reduced hallucination risk because outputs that violated schema or dimensions were rejected before submission.

In future rounds, how would you scale your prompting and decoding strategies when working with larger data sets or more complex LLMs?

I would use a two-stage workflow: constrained planning text first, deterministic execution second. With larger datasets, I would chunk context by function and include only essential statistics to avoid overload. With more complex models, I would keep low-randomness decoding for format-critical steps and reserve higher-diversity decoding for idea generation only. I would also keep a reusable validator for syntax, dimensionality, and bounds before any candidate is accepted.

How did these design choices for prompts and decoding help you think like a practitioner balancing exploration, risk and computational constraints in a black-box setting with incomplete information?

This round reinforced that practitioner work is about controlled uncertainty. I balanced exploration through random candidate search and gradient refinement, controlled risk through strict validation and deterministic execution, and managed compute by avoiding expensive tuning loops every round. In incomplete-information settings, I think the professional habit is to make assumptions explicit, validate aggressively, and prioritise reproducible pipelines over impressive but fragile one-off outputs.
