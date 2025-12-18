# Research Notes

## Query 1

Query:
```text
What architectural approaches have been proposed for representing irreducible uncertainty in neural network systems - not calibration or confidence estimation as post-hoc additions, but fundamental changes to how uncertainty propagates through computation?
```

Resposnse:

```markdown
Architectural work here mostly falls into a few families: Bayesian/weight‑distribution networks, evidential / credal / imprecise‑probability networks, “epistemic networks” and hyper‑ensemble generative models, plus some explicit uncertainty‑propagation constructions that treat layers as probabilistic factors rather than deterministic maps.[1][2][3][4]

## Bayesian and stochastic-weight networks

Bayesian neural networks place explicit probability distributions over weights, so every forward pass is a sample from a stochastic function, and predictive distributions integrate over that function space instead of treating the network as point‑valued.[5][6][7]
This allows irreducible (aleatoric) uncertainty to appear as residual risk even under the Bayes‑optimal posterior, while epistemic uncertainty lives in the spread of the weight posterior and shrinks with more data, giving a principled way for uncertainty to propagate through the entire computation graph rather than as a post‑hoc head.[8][1]

Dropout-as-variational inference and similar schemes convert standard architectures into approximate BNNs and can be extended with variance propagation so that mean and variance (or higher moments) are tracked layer by layer, effectively turning each deterministic layer into a probabilistic one that passes uncertainty forward.[9][7]

## Evidential and credal architectures

Evidential neural networks output parameters of evidence distributions (e.g., Dirichlet or normal‑inverse‑gamma) instead of single probabilities, so each prediction carries a belief mass and an explicit “vacuity” or imprecision term capturing lack of evidence.[2][10]
Recent “credal” and interval deep evidential classifiers generalize this by letting the network represent sets of distributions (credal sets) or intervals over class probabilities, so irreducible ambiguity appears as non‑singletons in the output belief set and this imprecision can be propagated through downstream decision rules without collapsing it to a scalar confidence.[10][4]

Imprecise‑probability inspired constructions (credal networks, Dempster–Shafer style layers) similarly treat internal representations as belief functions or bounded probability ranges; the architecture is designed so that combination rules (e.g., Dempster’s rule, convex credal updating) are applied at each layer, making ambiguity and underdetermination first‑class rather than noise to be averaged away.[11][12]

## Epistemic neural networks and auxiliary heads

“Epistemic neural networks” (epinets) augment a base network with an auxiliary epistemic head that receives the same representation and additional task‑indexing inputs, and is trained so that its output matches ensemble‑style epistemic uncertainty.[13]
The key move is architectural: the epistemic head is part of the model, not a post‑hoc calibration layer, and shares internal state with the predictive head in a way that lets uncertainty about parameters and data coverage be expressed as a structured function of the representation rather than as an external estimator.[13]

## Hyper‑ensembles and generative posteriors

Hyper‑diffusion models (HyperDM) implement an implicit ensemble by conditioning a diffusion model on a latent “model index” or hyper‑variable, so sampling different hyper‑codes corresponds to sampling different plausible models from an implicit ensemble.[3][14][15]
This lets a single architecture represent both epistemic and aleatoric uncertainty: variation across hyper‑samples reflects uncertainty over models, while variation within each conditional posterior reflects irreducible stochasticity in the data, and both are obtained by propagating randomness through the entire generative computation rather than attaching a confidence readout at the end.[14][3]

## Explicit uncertainty propagation through layers

There is also work that treats a trained deep net as a probabilistic factor graph: each layer becomes a time step, activations are random variables, and connections are factors, so input distributions (with their own aleatoric uncertainty) are propagated analytically or via constrained optimization through the network.[2]
Compared to scalar calibration, this explicitly models how uncertainty transforms through nonlinearities and skip connections, allowing one to track where irreducible noise is amplified or attenuated and to quantify output uncertainty as a structured function of the entire computation rather than as an after‑the‑fact score.[2]

Overall, the main “fundamental” approaches are: put distributions on parameters (BNNs), put distributions or belief sets on activations/outputs (evidential/credal, D–S), build dedicated epistemic heads tied into the representation (epinets), use hyper‑generative architectures that embody ensembles (HyperDM), or recast the network as a probabilistic graphical model so uncertainty is a state that flows through every layer rather than an auxiliary scalar attached at the end.[1][3][10][13][2]

[1](https://d-nb.info/1199537667/34)
[2](https://arxiv.org/html/2312.05946)
[3](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c693c3ff83259aebcd55a41ab19a5d84-Abstract-Conference.html)
[4](https://openreview.net/pdf?id=-I6La-GJS5b)
[5](https://people.csail.mit.edu/lrchai/files/Chai_thesis.pdf)
[6](https://indico.cern.ch/event/1208723/contributions/5230073/attachments/2600859/4521507/An%20Introduction%20to%20Bayesian%20Neural%20Network%20and%20Uncertainty%20Quantification%20in%20Deep%20Learning-Jacopo%20Talpini.pdf)
[7](https://repository.dl.itc.u-tokyo.ac.jp/record/2013241/files/A41855.pdf)
[8](https://www.gdsd.statistik.uni-muenchen.de/2021/gdsd_huellermeier.pdf)
[9](https://www.sciencedirect.com/science/article/pii/S0893608021003555)
[10](https://www.arxiv.org/pdf/2512.05526.pdf)
[11](https://academic.oup.com/logcom/advance-article/doi/10.1093/logcom/exae069/7833415)
[12](https://www.sciencedirect.com/science/article/pii/S0888613X23000981)
[13](https://liner.com/review/epistemic-neural-networks)
[14](https://openreview.net/forum?id=WPxa6OcIdg)
[15](https://github.com/matthewachan/hyperdm)
[16](https://arxiv.org/html/2501.03282v1)
[17](https://iclr-blogposts.github.io/2025/blog/reexamining-the-aleatoric-and-epistemic-uncertainty-dichotomy/)
[18](https://www.sciencedirect.com/science/article/pii/S0016003225000420)
[19](https://www.nature.com/articles/s41598-025-19002-w)
[20](https://www.sciencedirect.com/science/article/abs/pii/S0957417414000384)
```

Query:

```markdown
Have any of these approaches utilizes tensor based structures to represent direction of indeteriminacy?
```

Response:

```markdown
Most of the mainstream work you’d group under Bayesian, evidential, or credal deep learning stays at the level of scalar (per-dimension) variances, intervals, or belief masses, not explicit tensor fields encoding “direction” of indeterminacy in representation space.[1][2][3]

## What exists that is close

Neutrosophic convolutional neural networks map each pixel (or feature) into a triple \((T,I,F)\), and then process those as multi‑channel tensors through parallel “truth” and “indeterminacy” paths; this gives a tensorial *distribution* of indeterminacy over space, but not a directional vector field of “where the uncertainty wants to go next.”[4]
Type‑1 and type‑2 neutrosophic models in ML similarly use multi‑dimensional membership degrees (truth / indeterminacy / falsity, sometimes nested) but again treat them as stacked channels or higher‑order memberships rather than explicitly modeling gradients or directions of indeterminacy in a latent manifold.[5][6]

Credal and interval deep evidential classifiers represent second‑order uncertainty via credal sets or intervals per class, so outputs are tensors of lower/upper probabilities or evidential parameters, but these axes correspond to classes and evidence components, not to a geometric direction of “indeterminacy flow” in the representation space.[2][3][7]

## Directionality vs. tensor structure

Some BNN+latent‑variable work decomposes predictive uncertainty into epistemic vs aleatoric components per input and per output dimension, which you can view as a tensor over examples × dimensions × (types of uncertainty), but the “direction” is implicit (e.g., via gradients, Jacobians) rather than represented as a dedicated architectural object.[8][9]
Hyper‑diffusion / hyper‑ensemble approaches effectively define a distribution over models and outputs indexed by a latent hyper‑code, which is again tensorizable (samples × pixels × time, etc.), but the directionality of uncertainty is learned as a property of the diffusion dynamics, not surfaced as a separate directional indeterminacy tensor.[10]

## Direct answer

So far, published architectures that *explicitly* introduce tensor‑structured variables whose axes are “lenses” or “directions of indeterminacy” (in your sense: e.g., per‑cause, per‑judge, per‑hypothesis, with signed or oriented interactions over time) do not appear in the mainstream uncertainty‑in‑DL literature; where tensors occur, they mostly carry multi‑channel degrees or credal bounds, not directional semantics.[11][12][13]
That leaves your “sparse neutrosophic tensor of lenses with temporal interaction directions” idea in a genuinely underexplored space: adjacent to neutrosophic CNNs and credal/evidential nets, but more structured than anything currently standardized for uncertainty propagation.[4][2]

[1](https://arxiv.org/html/2302.09656v4)
[2](https://www.arxiv.org/abs/2512.05526)
[3](https://lirias.kuleuven.be/4266418)
[4](https://pmc.ncbi.nlm.nih.gov/articles/PMC9206226/)
[5](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644987/)
[6](https://fs.unm.edu/neut/ComparisonOfNeutrosophicSentiment.pdf)
[7](https://openreview.net/pdf?id=-I6La-GJS5b)
[8](https://mediatum.ub.tum.de/doc/1482483/1482483.pdf)
[9](https://homes.esat.kuleuven.be/~konijn/publications/2020/postels.pdf)
[10](https://proceedings.neurips.cc/paper_files/paper/2024/file/c693c3ff83259aebcd55a41ab19a5d84-Paper-Conference.pdf)
[11](https://radar.brookes.ac.uk/radar/file/23bec6a0-9c3f-4673-99ab-e7eb8acbbb25/1/A_Review_of_Uncertainty_Representation_and_Quantification_in_Neural_Networks.pdf)
[12](https://pubmed.ncbi.nlm.nih.gov/41150238/)
[13](https://www.semanticscholar.org/paper/Prior-and-Posterior-Networks:-A-Survey-on-Deep-For-Ulmer-Hardmeier/f9a522bae646c7d3a1b33e16a595f4c938558068)
[14](https://openreview.net/forum?id=PCgnTiGC9K&noteId=R5wZOgSZKe)
[15](https://www.youtube.com/watch?v=toTcf7tZK8c)
[16](https://digitalrepository.unm.edu/cgi/viewcontent.cgi?article=2414&context=nss_journal)
[17](https://d-nb.info/1199537667/34)
[18](https://pure.manchester.ac.uk/ws/portalfiles/portal/348403964/cbdl.pdf)
[19](https://papers.nips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty)
[20](https://arxiv.org/abs/2509.05551)
[21](https://arxiv.org/html/2501.03282v1)
```

## Query 2

Query:

```markdown
Are there neural architectures that preserve entropy through generation rather than collapsing to point estimates at each step? What work exists on non-autoregressive approaches that maintain multiple hypotheses?
```

Response:
```markdown
Yes. There are several families of architectures and inference schemes that explicitly preserve multimodality or distributions instead of collapsing to a single point estimate at each step, including diffusion models, stochastic decoding schemes (e.g., stochastic beam search), and lattice-/graph-based sequence models.[1][2][3][4]

## Diffusion and score-based models

Diffusion and score-based generative models parameterize a full stochastic process over time, and every denoising step keeps a distribution over states rather than committing to a single trajectory. Recent work on “generative uncertainty in diffusion models” and Bayesian diffusion shows how to extract calibrated uncertainty and posterior samples from these models, underscoring that they are fundamentally distribution-preserving generators. There is also work on diffusion-based uncertainty quantification (DBUQ) that uses diffusion as a surrogate prior for Bayesian inverse problems, again focusing on full posteriors rather than point predictions.[5][6][7][8][1]

## Non-autoregressive sequence models

Classical non‑autoregressive translation models (e.g., CTC-style and insertion-based decoders) generate tokens in parallel or with latent-length/ordering variables, effectively modeling a joint distribution over sequences without a strict left‑to‑right factorization, though many training setups still optimize a mode-like objective. Double‑path and related multi-path seq2seq models maintain multiple latent paths or decoders in parallel, using them to represent multiple candidate hypotheses before final aggregation.[9]

## Lattice and graph-based sequence models

Neural lattice-to-sequence models explicitly ingest upstream uncertainty (e.g., word lattices from ASR or segmenters), propagating a graph of possible inputs instead of a single best path, and learning to attend over this structure. Relatedly, neural program lattices represent the internal computation itself as a lattice of approximate program states across time, so at each timestep the network maintains a grid of possible states instead of a single hidden state trajectory.[10][3][4]

## Decoding schemes that keep multiple hypotheses

Even for standard autoregressive models, decoding strategies like beam search or best‑first beam search maintain a frontier of hypotheses in parallel, approximating inference over multiple modes rather than a single greedy trajectory. Stochastic beam search generalizes this by introducing controlled randomness into beam expansion, giving a principled algorithm that both tracks multiple high‑probability paths and better approximates expectations and entropy over the model’s sequence distribution.[2][11][12][13]

## Uncertainty-centric work

In text generation, systematic studies of decoding (e.g., “On Decoding Strategies for Neural Text Generators”) analyze how deterministic, mode-seeking strategies reduce entropy versus stochastic sampling methods that better reflect the model’s full distribution. In vision, several recent papers focus on pixel-wise or semantic uncertainty in diffusion models (e.g., BayesDiff, generative uncertainty for text‑to‑image), explicitly framing these generators as tools for uncertainty quantification rather than point estimators.[14][15][7][16]

If you want, a follow-up could sketch a more “architectural” design where the hidden state itself is explicitly a structured, normalized distribution over hypotheses (e.g., a recurrent particle filter or learned lattice) rather than a single vector.

[1](https://arxiv.org/html/2502.20946v1)
[2](https://www.emergentmind.com/topics/stochastic-beam-search)
[3](https://aclanthology.org/anthology-files/pdf/D/D17/D17-1145.pdf)
[4](https://arxiv.org/pdf/1704.00559.pdf)
[5](https://arxiv.org/html/2503.03007v1)
[6](https://arxiv.org/html/2502.20946v2)
[7](https://proceedings.neurips.cc/paper_files/paper/2023/file/374050dc3f211267bd6bf0ea24eae184-Paper-Conference.pdf)
[8](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JH000234)
[9](https://aclanthology.org/C18-1259.pdf)
[10](https://openreview.net/pdf?id=HJjiFK5gx)
[11](https://aclanthology.org/D17-1227.pdf)
[12](https://web.engr.oregonstate.edu/~huanlian/papers/optimal_beam_search.pdf)
[13](https://rycolab.io/papers/meister+al.tacl20.pdf)
[14](https://openaccess.thecvf.com/content/CVPR2025/papers/Franchi_Towards_Understanding_and_Quantifying_Uncertainty_for_Text-to-Image_Generation_CVPR_2025_paper.pdf)
[15](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00502/113024/On-Decoding-Strategies-for-Neural-Text-Generators)
[16](https://openreview.net/forum?id=YcM6ofShwY)
[17](https://proceedings.mlr.press/v286/jazbec25a.html)
[18](https://raw.githubusercontent.com/mlresearch/v286/main/assets/jazbec25a/jazbec25a.pdf)
[19](https://www.isca-archive.org/interspeech_2019/michel19_interspeech.pdf)
[20](https://milvus.io/ai-quick-reference/how-do-you-prevent-mode-collapse-in-diffusion-models)
```

## Query 3

Query:

```markdown
What work exists at the intersection of resilience engineering, uncertainty quantification, and neural architecture design? Are there systems approaches to AI that treat uncertainty as a first-class architectural concern rather than a training problem?
```

Response:

```markdown
There is a small but real body of work that treats uncertainty as an architectural and systems-level design concern, but it is fragmented across: (1) probabilistic/“uncertainty-native” neural architectures, (2) complex-systems resilience with explicit UQ, and (3) software/system architectures for resilience and antifragility that now grapple with AI modules.[1][2][3][4][5][6][7][8]

## Uncertainty-native neural architectures

Within deep learning, most “uncertainty work” is still framed as a training or post-hoc calibration problem, but several families explicitly bake UQ into the architecture or forward pass:
- Bayesian neural networks and variational approximations: weights are distributions, so the forward pass is explicitly stochastic; recent work pushes these onto extreme-edge hardware and integrates UQ into memory-centric architectures, not just as a wrapper.[9][10]
- Deep ensembles and hyper-ensembles: some papers explicitly characterize ensembles as UQ frameworks, where the architecture is “a population of networks plus aggregation” rather than a single model, and robustness under perturbation is analyzed through ensemble uncertainty statistics.[11][12][6]
- UQ-specific training schemes: approximate Bayesian computation and related methods replace standard gradient-based training with algorithms that treat the whole network as a probabilistic simulator whose uncertainty is part of the object of inference, blurring the line between architecture and inference machinery.[13][10]

A recent survey in engineering applications explicitly organizes DNN-UQ methods into representation-level vs architecture-level categories, and reviews probabilistic, evidential, and imprecise-probability inspired nets as “uncertainty-representing” architectures rather than simple confidence heads.[12][10][6]

## Resilience and UQ in complex networks

On the resilience engineering side (outside ML), there is work that takes uncertainty as an intrinsic part of system-level dynamics and designs analysis frameworks accordingly:
- Multi-scale resilience with arbitrary polynomial chaos: a framework for networked nonlinear systems that quantifies how parameter uncertainty propagates to network-level and node-level resilience, using arbitrary polynomial chaos expansions to support arbitrary input distributions.[1]
- Probabilistic and dynamic Bayesian network–based resilience models: resilience of infrastructure systems is modeled explicitly in probabilistic graphical form (e.g., DBNs), with resilience measures defined over distributions of performance trajectories rather than single-point metrics.[14][15]
- Machine-learning-based resilience inference: ResInf combines transformers and GNNs to infer resilience from observational data, optimizing a representation specifically for predicting “closeness to critical thresholds”; this is a systems-style architecture that treats resilience (and implicitly, uncertainty about tipping points) as a first-class output, not an afterthought.[3]

This literature typically embeds UQ structurally (random parameters, random topology, stochastic dynamics) and then asks how resilience degrades or is maintained as uncertainty varies, rather than treating uncertainty as a nuisance to be driven to zero.[15][1]

## Systems and software architecture for resilience

In software/systems engineering, there is explicit architectural thinking about resilience and antifragility, increasingly in the presence of AI components:
- Antifragile software and systems: recent work proposes a conceptual and architectural characterization of antifragility and argues for design guidelines in which systems are structured to benefit from variability and shocks; uncertainty and disturbance are treated as normal operating conditions to be metabolized by the architecture, not as rare failures.[5]
- System-of-systems UQ with ML: work on SoS-level UQ compares neural networks with parametric Bayesian methods to propagate uncertainty through large engineering systems, treating UQ as an integral part of system evaluation and design choices rather than a surrogate metric.[4]
- Resilience assessment with ANNs: structural and infrastructure resilience studies use ANNs as surrogates in optimization loops that map uncertain hazard/response parameters to resilience metrics, using UQ across these surrogates to inform design tradeoffs.[16][14]

These are not “neural architectures that internalize uncertainty semantics” per se, but they do adopt a systems view in which uncertainty is part of the design surface (e.g., resilience curves over uncertainty distributions) rather than a training defect.[16][14][4][1]

## AI systems treating uncertainty architecturally

The closest match to “systems approaches to AI that treat uncertainty as a first-class architectural concern” shows up in emerging AI/controls and governance work:
- “Uncertainty architecture” as a control-theoretic stack: one recent position piece describes a four-layer “uncertainty architecture” where sensing, modeling, control, and governance layers are all explicitly designed around uncertainty propagation and feedback, arguing that AI governance should be treated as a control problem over uncertain dynamics.[8]
- Resilience engineering for AI in critical infrastructure: engineering articles aimed at OT/ICS contexts argue that AI modules need explicit failure contracts, graceful degradation paths, observability by default, and boundary conditions for uncertain behavior—effectively making uncertainty behavior part of the API and architectural contract, not just a model metric.[2][17]
- Bayesian deep RL for control under uncertainty: work in Bayesian DRL designs agents whose policy and value networks are Bayesian, with uncertainty feeding into exploration and robust control; here, uncertainty is a core state passed through the decision architecture, not only a confidence score.[18]

These are still early and somewhat siloed, but they are explicitly architectural: they define how uncertainty must flow through sensing, inference, actuation, and governance loops, and how system-level resilience depends on that flow.[17][2][18][8]

## How well does this answer your target?

Putting this together:
- There is substantial probabilistic and imprecise-probability neural network work, but most of it still talks the language of “better UQ for predictions” rather than “uncertainty as a load-bearing architectural state.”[10][6][9][12]
- Complex-systems resilience and infrastructure engineering already adopt a stance where uncertainty is intrinsic, and architectures (both analytic and data-driven) are designed to trace how uncertainty reshapes resilience; this is probably the richest conceptual source if you want to import resilience engineering into neural/system design.[3][14][15][1]
- The most “systems-architectural” AI work so far appears in control/governance/antifragility discussions, where uncertainty is built into multi-layer control loops and explicit failure contracts, rather than in mainstream neural architecture design per se.[2][18][5][8]

If helpful, a next step could be to carve out a more precise niche like “resilience-aware neural architectures for multi-scale systems,” and map concrete design patterns: e.g., explicit uncertainty channels across layers, resilience metrics as architectural design objectives, and failure contracts at module interfaces.

[1](https://pmc.ncbi.nlm.nih.gov/articles/PMC9831990/)
[2](https://www.linkedin.com/pulse/resilience-engineering-age-ai-designing-fail-safe-outcomes-butler-sqslf)
[3](https://pmc.ncbi.nlm.nih.gov/articles/PMC11502705/)
[4](https://incose.onlinelibrary.wiley.com/doi/10.1002/sys.21561)
[5](https://www.sciencedirect.com/science/article/pii/S0164121224000967)
[6](https://radar.brookes.ac.uk/radar/file/23bec6a0-9c3f-4673-99ab-e7eb8acbbb25/1/A_Review_of_Uncertainty_Representation_and_Quantification_in_Neural_Networks.pdf)
[7](https://research.ibm.com/projects/imprecise-probabilistic-logic)
[8](https://www.linkedin.com/pulse/uncertainty-architecture-why-ai-governance-actually-control-oborskyi-oqhpf)
[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC10661910/)
[10](https://www.diva-portal.org/smash/get/diva2:1648236/FULLTEXT02.pdf)
[11](https://arxiv.org/html/2503.04142v2)
[12](https://wjarr.com/sites/default/files/WJARR-2021-0421.pdf)
[13](https://www.sciencedirect.com/science/article/pii/S0952197621003596)
[14](https://lre.mb.tu-dortmund.de/storages/lre-mb/r/Pre-prints/RUENG-1244_R1.pdf)
[15](https://pure.tudelft.nl/ws/portalfiles/portal/70261238/Manuscript_Bayesian_postprint.pdf)
[16](https://www.sciencedirect.com/science/article/pii/S235201242200889X)
[17](https://pmc.ncbi.nlm.nih.gov/articles/PMC12171450/)
[18](https://www.nature.com/articles/s41598-025-19002-w)
[19](https://pubsonline.informs.org/doi/10.1287/isre.2022.0440)
[20](https://arxiv.org/html/2503.12687v1)
```
