---
layout: distill
title: Dynamic Parameter Reuse Augments Reasoning via Latent Chain of Thought
description: Standard language models often rely on massive parameter counts for their performance, utilizing each parameter only once per inference pass. This prompts consideration of recurrent structures, where models reuse parameters across sequential time, depth, or training progression to achieve improved performance and reduced training cost. We draw connections in the landscape of parameter reuse, from growing models via stacking to recurrent looping, and postulate that these architectural priors act as a form of Latent Chain of Thought (LCoT), allowing models to reason in a continuous state space. By shifting towards deeper and dynamic computation, grown and recurrent architectures offer a path toward improved reasoning in compact networks, ascending beyond scaling laws of standard architectures.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

authors:
  - name: Anonymous

bibliography: 2026-04-27-recurrefinereason.bib

toc:
  - name: Introduction
  - name: Reuse in Time vs. Depth
    subsections:
      - name: The Time Axis (RNNs and SSMs)
      - name: The Depth Axis (Looping)
  - name: Implicit Looping
    subsections:
      - name: Depth Growth via Stacking
      - name: Induced Looping via Surgery
  - name: Depth Recurrence as Latent Chain of Thought
  - name: Sequential Refinement
  - name: Recursion and Hierarchicality
  - name: Dynamic Routing
  - name: Towards Compositional Recurrence
  - name: Conclusion


---

<em><center>Reduce, Reuse, Recycle. <br>
Recur, Refine, Reason.</center></em>

## Introduction

In the current AI era, progress has been dominated by scale: increase the model size and thus the cost in exchange for improved performance <d-cite key="kaplan2020scaling"></d-cite>. In standard sequence modeling, exemplified by the transformer architecture <d-cite key="vaswani2017attention"></d-cite>, each learnable parameter is used only once per processed token <d-footnote>An exception is using the transpose of the embedding table to decode the final embeddings into token space, a technique common in some language models.</d-footnote>. This "single-pass" paradigm effectively treats the network as a rigid, acyclic circuit. After standard training, this model cannot improve its own performance to achieve a more capable model: instead the model must be discarded and a new one reinitialized at a larger scale with more parameters to learn from scratch. This is algorithmically inefficient. 

Decoupling parameter count from computation, parameter reuse within the architecture allows the training process to occur in a reduced search space, forcing the model to learn generalizable functions within the repeated modules. A classic example is Convolutional Neural Networks (CNNs) <d-cite key="fukushima1988neocognitron"></d-cite> and, more broadly, equivariant neural networks in geometric deep learning <d-cite key="bronstein2021geometric"></d-cite>. By sliding the same kernel across an image, CNNs enforce translational symmetry that unlocked new levels of computer vision performance beyond what was possible by scaling up previous architectures without such parameter reutilization. Even if the functionality of a CNN can be expressed in an unconstrained multilayer perceptron (MLP) and thus could be found via gradient descent, the size and nonconvexity of the MLP's search space makes such a solution infeasible. This demonstrates the benefit of using architectural priors in trained models.

We are now seeing a similar renaissance of reuse, both in architectural elements and in training methods, burgeoning in sequence modeling, but in different and thus far unconnected flavors. In the taxonomy of AI architectures, we focus this blog post on the order containing Transformers and (Time) Recurrent Architectures, including State Space Models (SSMs), for sequence modeling. Broadly, such models process and generate sequences of data through a stack of layers, often autoregressively. These layers follow repeated block patterns across the depth but are individually parametrized. We will discuss how blocks of these layers may be looped, grown, and reused to unlock efficiency and, crucially, reasoning capabilities. The intent of this blogpost is to discuss the parallels of these flavors of recurrence and related methods across various axes: both those which are inherent to architectural genuses such as time recurrence as well as those which can augment any models within this order that follow the blocked layer structure. These flavors have thus far mostly only been considered in isolation, but exploring their analogues from a unified lens may unlock further synergies. 


## Reuse in Time vs. Depth

To understand recent advances in looping and reuse, we first distinguish between two orthogonal axes of recurrence: over the input sequence (Time) and over the computation (Depth).

### The Time Axis (RNNs and SSMs)

This is the classical domain of Recurrent Neural Networks (RNNs), including models containing Long Short-Term Memory modules (LSTMs) and Gated Recurrent Units (GRUs). Here, the model reuses the same function $f_\theta$ at every position in the sequence. A latent state $h_{t-1}$, representing a compressed memory of the past, is fed along with the current token $x_t$ into the network to produce $h_t$ (possibly in addition to an output $y_t$).

While Transformers largely overtook RNNs due to the latter's training inefficiencies such as the inability to parallelize, the concept has returned in State Space Models (SSMs) <d-cite key="gu2022efficiently"></d-cite>. These models offer the inference efficiency of RNNs, achieving linear time complexity in sequence length as opposed to the quadratic complexity of transformers, with the training parallelizability of Transformers. Hybrid models interleave time-recurrent layers with attention-based layers throughout the depth of the architecture.

We note that the model is still performing a fixed amount of compute per token. This limitation may be bypassed via Chain-of-Thought (CoT) <d-cite key="wei2022chain"></d-cite>, utilizable in any autoregressive model beyond just those that use explicit time recurrence. By generating extra intermediate tokens, a model effectively uses the time axis to "buy" more compute, treating the generated sequence as an extended memory buffer for reasoning. We will revisit this concept later when discussing latent reasoning.

### The Depth Axis (Looping)

Instead of recurring across time, looped models portray recurrence in computation depth on the current token. The most basic approach loops the entire model (or specific blocks) on the same input representation multiple times.

This traces back to Universal Transformers <d-cite key="dehghani2019universal"></d-cite>, which dynamically loops a transformer block until a halting mechanism triggers. Bai et al. <d-cite key="bai2019deep"></d-cite> take this to the extreme with Deep Equilibrium Models (DEQs), defining the output of a network not via a finite sequence of layers, but as the fixed point $z^*$ of an infinitely applied function, found via root-finding algorithms. This further enables deriving efficient constant-memory training algorithms that make use of the implicit function theorem, as shown much earlier by Pineda <d-cite key="pineda1989recurrent"></d-cite> and Almeida <d-cite key="almeida1988backpropagation"></d-cite>. These renditions of looped models <d-cite key="dabre2019recurrent"></d-cite> <d-cite key="lan2020albert"></d-cite> repeated a single transformer layer, which is an attention layer followed by a two layer MLP. More recent methods often loop larger blocks of layers <d-cite key="saunshi2025reasoning"></d-cite>.

Generally, most methods either loop immediately from initialization or only loop after the full pretraining as a method of inference-time scaling. Next, we will explore related methods that yield similar effects to looping but utilize dynamic architectures across their training lifecycle.

## Implicit Looping

If depth is the goal, how else do we reach it without the training instability and cost of massively deep networks?

At initialization, signal propagation in a very deep network is hindered compared to a shallow one. While residual connections solve the gradient vanishing problem to an extent <d-cite key="sun2025curse"></d-cite>, they do not handle the "training efficiency" problem.

Equating training steps or FLOPs, smaller models tend to train much more efficiently at the beginning but plateau early. Intra-training model growth strategies try to follow the rapid training curve of a small model, then expand the loss landscape dimensionality to surpass the plateau towards the training curve of the larger architecture as if it were trained from scratch yet saving training costs.

For the context of depth recurrence, we focus the following subsection on depth growth (and particularly on one established strategy for it), although models may also be grown in other dimensions <d-cite key="gesmundo2023composable"></d-cite> <d-cite key="du2024stacking"></d-cite>.


### Depth Growth via Stacking 

Depth growth, when accomplished by stacking gradually during training, can be viewed effectively as "progressive initialization via looping". MIDAS (Middle Gradual Stacking) proposed by Saunshi et al. <d-cite key="saunshi2024inductive"></d-cite> takes a trained block of $N$ layers from the middle of the model depth, copies it, and stacks it on top of itself. This process repeats multiple times periodically throughout training, growing the model from a single block of layers to full standard model depths, for example from 4 layers and adding on 4 layers at defined intervals until reaching 32 layers then cooling down the pretraining phase. The MIDAS strategy, depicted below, of initializing the new block using pretrained parameters from a neighboring block outperforms other depth-growth strategies, including interwoven layer insertions and random initializations <d-cite key="karp2024landscape"></d-cite>.  

<div class="row mt-3">
    <div class="col-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-recurrefinereason/midas.png" class="img-fluid" width="70%" %}
    </div>
    <div class="col-5 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-recurrefinereason/cossim.png" class="img-fluid" width="30%"%}
    </div>
</div>
<div class="caption">
    The Middle Gradual Stacking (MIDAS) Strategy <d-cite key="saunshi2024inductive"></d-cite>. Left: This visualization shows how model depth grows during training, where each blue block is a block of layers. Right: Weight similarity, measured by cosine similarity, across the six blocks each consisting of four layers of a MIDAS-grown model. Figures from Saunshi et al. <d-cite key="saunshi2024inductive"></d-cite>.
</div>

Because parameters are deep-copied instead of merely shared, they may diverge upon further training. Each block thus shares at least part of its training history with other blocks throughout the depth of the network. The weight similarity between blocks of a MIDAS-grown model are depicted above: blocks near the middle of the network were forked more recently than at the beginning and end, thus yielding more similar weights.

Such initialization of each newly inserted block provides an inductive bias as the model continues training at the increased depth. The final model is a standard single-pass network (albeit trained more efficiently), but the initialization scheme simulates soft looping with refinement due to the shared training histories. Thus, this style of depth stacking yields a model that is akin to a looped model, where the loop depth is grown over the course of training, but each loop has a parametrization that is most similar between middle blocks and most customized for the initial and final blocks. This patterning is also seen in looping methods that utilize a prelude and/or coda <d-cite key="geiping2025efficient"></d-cite><d-cite key="mcleish2025teaching"></d-cite><d-cite key="alabdulmohsin2025recursive"></d-cite> to augment the initial and final computations beyond simply embedding and un-embedding with a single layer.

Other depth-growth methods stack the entire sequence of layers, multiplying the depth at each growth <d-cite key="gong2019efficient"></d-cite><d-cite key="du2024stacking"></d-cite>. The resulting weights are thus more aligned across the depth, more analogous to looping methods that loop the entire depth of layers.

### Induced Looping via Surgery

Depth growth via stacking is not the only strategy to achieve this loop-like effect. Bae et al. <d-cite key="bae2025relaxed"></d-cite> and McLeish et al. <d-cite key="mcleish2025teaching"></d-cite> perform parameter surgery on pretrained unlooped models to fit a soft looping architecture before continuing training. The useful inductive bias for mid-training initialization is thus also present, but the efficiency gains come from reusing parameters of previously trained models instead of within the training process itself.

These surgery techniques induce (soft) looping across the depth axis, where intermediate layers now share weights. During subsequent finetuning, the model trains within the constrained parameter space, enforcing this recurrence in the final architecture and at inference time. Other surgery techniques perform surgery to initialize deeper models from trained shallow ones, but without weight tying constraints during further training <d-cite key="kim2023solar"></d-cite>, thus even closer to the paradigm of stacking depth growth.

Both progressive stacking and parameter surgery are effective strategies for imposing a recurrent prior onto models: stacking achieves this by gradual growth during pre-training, while surgery does so by structurally modifying a pre-trained vanilla model.

## Depth Recurrence as Latent Chain of Thought

Beyond efficiency, a rather exciting implication of looping and parameter reuse is reasoning.

In standard Chain-of-Thought (CoT) <d-cite key="wei2022chain"></d-cite>, the model explicitly materializes its reasoning into discrete tokens (e.g., "Let's think step by step..."), which are recycled as context for subsequent steps. This mechanism effectively allows the model to trade time for compute at inference-time, expanding its effective depth by utilizing the context window as a scratchpad. However, this introduces a bottleneck: to generate a token, the model must collapse the high-dimensional internal state into a single discrete symbol before reuse in future token generations. This discretization discards the nuanced information held in the hidden states, forcing the model to commit to a specific textual path. 

Looping enables the model to reason entirely within the continuous embedding space, avoiding the 'lost in translation' effect caused by the discretization noise inherent in token generation. In this regime, the input to the repeated block comes directly from the previous iteration's latent state. Through superposition and uncertainty, this may simultaneously represent multiple conflicting reasoning paths, refining the probability distribution continuously rather than committing to a single, potentially erroneous branch early in the derivation. The ability to hold multiple hypotheses simultaneously within the vector space is a key advantage, circumventing the information loss associated with discrete token sampling.

Saunshi et al. <d-cite key="saunshi2025reasoning"></d-cite> and Hao et al. <d-cite key="hao2024training"></d-cite> further explore the similarities of looping as Latent Chain-of-Thought, or LCoT. However, as these works only consider strict looping for LCoT, the repeated function must be identical with each loop. 

Depth growth via stacking and induced soft looping present an even more powerful evolution of LCoT: By initializing layers as copies (stacking) during depth growth or sharing parameters with learned offsets (soft-looping), we start with the useful inductive bias of a loop but allow for computational specialization throughout the depth of recurrence. This adaptive approach means the model can dynamically learn specialized transformations, such as early layers focusing on feature extraction and later layers focusing on logical inference, to different stages of the softly recurrent reasoning process, which is otherwise impossible in a strictly weight-tied looped model. The weight similarity pattern of MIDAS portrays this depth-wise specialization. 

Ultimately, this suggests that depth recurrence is not just a compression trick, but a functional analogue to time recurrence operating in a richer, continuous, and potentially adaptive design space. 

## Sequential Refinement

To complement refinement in the depth axis, how can refinement be useful in the token axis? Most models are autoregressive: they process and generate one token at a time. This "hard commit" nature of autoregressive decoding means that earlier errors cannot be corrected based on later context, limiting the quality of sequential generation. To overcome this, sequential refinement strategies may be employed to allow the model to review and refine tokens. This is particularly effective on latent depth-recurrent models, where the recurrent module receives both the embedded input and a latent state that is initialized to noise, refined with each recurrence, and finally decoded to the output. This is reminiscent of the hidden state used in time-recurrent models discussed previously, except the same input is injected at each recurrence. 

Geiping et al. <d-cite key="geiping2025efficient"></d-cite> propose diffusion forcing sampling for latent depth-recurrent models to enable recent token refinement at each step: the current token goes through the recurrent unit only once before beginning generation of on the next one, so recent tokens continue to be recurrently refined in a sliding windowing manner. This further ameliorates the token discretization issue previously discussed for standard CoT.

Hierarchical Reasoning Models <d-cite key="wang2025hierarchical"></d-cite> and their successor Tiny Reasoning Models <d-cite key="jolicoeur2025less"></d-cite> feed the entire latent sequence back through its hierarchical modules, looping multiple times on the latent “scratchpad” for each output refinement. The lower level module receives the embedded imput sequence and both the high-level and low-level latent states to update the low-level latent state, and the higher level, lower freqency module receives only the latent states to update the high-level latent state that will eventually be decoded to the full output seqeunce. 

In between sequential refinement and explicit CoT, a simple approach lies in the use of a "pause" token <d-cite key="goyal2024think"></d-cite>. Although such a token still requires a hard commit, incorporating learnable "pause" tokens allows the model to delay the output generation both token by token during generation as well as after processing the input and beginning to generate the output, effectively utilizing the additional computation to refine its internal state before committing to a final answer nor using an explicit chain of thought.

Collectively, these methods demonstrate that the token axis need not be a one-way street: allowing the model to 'look back' or 'pause' introduces a temporal buffer that serves a similar refining purpose to latent depth-wise recurrence.

## Recursion and Hierarchicality

While sequential refinement allows a model to "polish" its recent output by looking back, it typically remains constrained to the flat, linear granularity of the token stream. However, complex reasoning is rarely linear but rather structural: a reasoning model may benefit from processing and organizing information hierarchically rather than purely sequentially. This motivates a shift from recurrence over raw tokens to recursion over levels of abstraction.

Hierarchical sequence models break the monotony of the single token stream. The Hierarchical Joint-Embedding Predictive Architecture (H-JEPA) <d-cite key="lecun2022path"></d-cite> exemplifies this by abandoning pixel-level or token-level prediction, rather predicting the latent representation of future states based on current states and actions, at both low and high levels of abstraction. 

This philosophy extends to language and continuous sequence modeling through architectures that explicitly decouple global context from local generation <d-cite key="ho2024block"></d-cite> <d-cite key="bhirangi2024hierarchical"></d-cite> <d-cite key="hwang2025dynamic"></d-cite>. These methods generally perform coarse-grained processing in earlier layers and fine-grained computations in later layers. By abstracting the "time" axis into a hierarchy of resolutions, these models reuse parameters to reason over longer horizons, effectively disentangling high-level planning from low-level syntax.

As discussed previously, Hierarchical Reasoning Models <d-cite key="wang2025hierarchical"></d-cite> and Tiny Reasoning Models <d-cite key="jolicoeur2025less"></d-cite> also utilize modules at different levels, although here the frequencies of application are in depth rather than time.

Establishing such hierarchies naturally raises the question of resource allocation: if the model possesses different levels of abstraction, must every input generically pass through every level? This leads us directly to the concept of conditional computation.

## Dynamic Routing 

Another axis that breaks the hard-coded paradigm is dynamic routing. This family of techniques customizes the computational path for each input based on its content, seeking to enhance both efficiency and capacity via specialization. One of the most well-known dynamic routing techniques is portrayed by Mixture-of-Experts (MoEs) <d-cite key="shazeer2017outrageously"></d-cite>, which replaces the standard Feed-Forward Network (FFN) with several specialized subnetworks (experts) and a router that selects only a few experts per token. However, classic MoE architectures often still fix the total depth and FLOPs per token, merely selecting which parameter sets to activate within these computations.

The concept of input-dependent layer-dropping pushes dynamic routing further by aiming to customize the computational depth per token. Early-exit strategies aim to accelerate inference in large autoregressive models by allowing tokens to skip later blocks. Depth-recursive models are even more naturally adept at modulating the effective depth per token. 

This optimization introduces significant complications particularly related to maintaining the Key-Value (KV) cache, a mechanism utilized by autoregressive attention-based models that stores the computed keys and values from attention layers corresponding to previously generated tokens to accelerate sequence generation at inference time. When a token exits early, the KV caches for all subsequent (skipped) layers are missing, complicating the generation of future tokens that might require a deeper path. Solutions include duplicating the cached items from current token's exit layer to subsequent layers <d-cite key="schuster2022confident"></d-cite> or using batched forward passes upon cache miss <d-cite key="bae2023fast"></d-cite>. Depth recurrence methods with partial or full weight sharing can counteract some of these inefficiencies, allowing for depth-wise batching such that tokens at different recurrence depth may be re-batched together <d-cite key="bae2025relaxed"></d-cite>.

While dynamic routing optimizes where extra computation happens, and sequential refinement optimizes when it happens, a lucrative path forward lies in combining these mechanisms with the structural priors of recurrence discussed earlier.

## Towards Compositional Recurrence

Many of the high-level methods presented are orthogonal and thus composable, perhaps even synergistically, offering a flexible toolkit for designing resource-efficient reasoning models. One such example is Zamba <d-cite key="glorioso2024zamba"></d-cite>, which alternates blocks of time-recurrent SSM layers with a shared-weight attention module. As further hypothetical examples, an MoE with a looped block that shares attention layer weights but not experts could be grown in recurrence depth during pre-training, using stacking to initialize the new experts. A fully trained hybrid model could undergo parameter surgery to induce soft looping, followed by inference-time utilization of sequential refinement and depth-wise batching. The convergence of these methods, where efficiency meets algorithmic capability, points toward a future of modular and resource-adaptable models rather than brute-force scale.  

## Conclusion

With the desire towards deeper reasoning, the rigid, single-pass transformer architecture appears increasingly insufficient and ineffective. By reusing parameters through looping and simulating recurrence depth through growing, models can effect deeper computation without necessarily becoming larger or more costly to train. The synergies of depth growth, parameter sharing, and looping can be utilized across the training cycle of a model to achieve reasoning capabilities that ascend beyond scaling laws of standard models. 

