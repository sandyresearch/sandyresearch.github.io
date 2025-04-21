---
layout: post
title: "Chipmunk: Training-Free Acceleration of Diffusion Transformers with Dynamic Column-Sparse Deltas (Part I)"
---

*Austin Silveria, Soham Govande, Dan Fu \| [Star on GitHub](https://github.com/sandyresearch/chipmunk)*

This is the first part of a three-part series. Part I (this part) will cover an overview of Chipmunk's algorithms. [Part II](https://sandyresearch.github.io/chipmunk-part-II/) will build theoretical intuition for column-sparse deltas. [Part III](https://sandyresearch.github.io/chipmunk-part-III/) will be a deep dive on  GPU kernels & systems optimizations.

**TL;DR:** We present Chipmunk, a training-free method to accelerate diffusion transformers with hardware-aware dynamic sparsity.  Chipmunk caches attention weights and MLP activations from previous steps and dynamically computes a sparse “*delta*” against the cached weights. Chipmunk achieves up to 3.7x faster video generation on HunyuanVideo at 720x1280 resolution for a 5s video, and 1.6x faster image generations on FLUX.1-dev at 1280x768 resolution.

<video controls autoplay style="width: 100%">
  <source src="https://sandyresearch.github.io/images/chipmunk/grid-video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

![][comparison]
<center>Images of cute chipmunks can be generated 1.37x faster! <b>Left</b>: Fully Dense FLUX.1-dev. <b>Right</b>: Ours (84% sparse attention and 70% sparse MLP)</center>

**Motivation:** Diffusion Transformers (DiTs) have become the standard for video generation, but the time and cost of generation keeps them out of reach of many applications. We raise two questions: (1) What do the model activations want to do? (2) What does the hardware want to do? We then use these insights to design hardware-friendly algorithms that maximize quality per unit of generation time.

In this post, we unpack:

1. **Slow-Changing, Sparse Activations:** DiT activations for MLP and attention change slowly across steps, and they are naturally sparse.  
2. **Cross-Step Deltas:** Because of the slow changing activations and natural sparsity, reformulating them to compute cross-step deltas make them even sparser.  
3. **Hardware-Aware Sparsity Pattern:** For both attention and MLP, we can pack dense shared memory tiles from non-contiguous columns in global memory. We open-source fast kernels for this!


| Hunyuan | VBench Quality | VB Semantic | VB Total | Resolution | Sparsity | Latency | Speedup |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| FlashAttention-3 | 85.09% | 75.82% | 83.24% | 720 x 1280 x 129 | 0% | 1030s | 1x  |
| Sliding Tile Attention (Training-Free) | 84.63% | 73.83% | 82.46% | 768 x 1280 x 117 | 58% | 945s \-\> 527s | 1.79x  |
| Chipmunk (Training-Free) | 84.60% | 76.29% | 82.94% | 720 x 1280 x 129 | 82% \* | 1030s \-\> 477s | 2.16 x  |
| Chipmunk \+ Step Caching (Training-Free) | 84.22% | 75.60% | 82.50% | 720 x 1280 x 129 | 87% | 1030s \-\> 277s | 3.72x  |

 \* 93% sparsity on 44 out of 50 steps for an average of 82% sparsity.

| FLUX.1-dev\* (bf16) | ImageReward | MLP Sparsity | Attn Sparsity | Speedup |
| :---- | :---- | :---- | :---- | :---- |
| Baseline (with FlashAttention-3) | 76.6% | 0% | 0% | 1x |
| Chipmunk | 80.2%	 | 70% | 83.5% | **1.37x** |
| Chipmunk \+ Step Caching | 78.0% | 70% | 83.5% | **1.63x** |

These FLUX.1-dev numbers were evaluated on 1280x768 images, and we’ve found that if we increase image size to 2304x1280, we can get speedups of up to 1.65x per-image without stacking on top of step caching methods, and 1.9x with step caching! We’ve also found that we can sparsify FP8 Flux to get a 1.1x end-to-end speedup over the fastest open-source implementation.

## Slow-Changing, Sparse Activations

Chipmunk exploits two simple observations about diffusion transformers:

1. **Activations move slowly:** In each step a Diffusion Transformer (DiT) denoises a latent noise vector. This noise vector changes slowly across successive steps in the diffusion process – and so do the [per-layer](https://arxiv.org/abs/2411.02397) [activations](https://arxiv.org/abs/2410.05317).  
2. **Activations are sparse:** In attention, it is common to see queries place a very large percentage of their attention probability mass on a small subset of keys–this means that the output will mostly be made up of the small subset of associated rows of V. And in MLP, previous works have observed significant sparsity in the intermediate activations of both [ReLU](https://arxiv.org/abs/2210.06313) and [GeLU](https://arxiv.org/abs/2408.14690)\-based layers, meaning that the output will mostly be made up of the top activated rows of W2.

## Activation Deltas Across Diffusion Steps are *Very* Sparse

Chipmunk uses these two observations to reduce the compute costs of the diffusion model – we can effectively capture nearly all the cross-step changes in the activations by ***only*** **recomputing a small subset of attention and MLP..** 

What does this mean, concretely? Let’s revisit the attention and MLP equations:

1. **Attention:**  softmax(Q @ KT)  @ V  
2. **MLP:**                 gelu(X  @ W1) @ W2

Both operations use a non-linearity to compute the scalar coefficients for a linear combination of value vectors. In attention, the value vectors are dynamic (V is projected from the current token representation). In MLP, the value vectors are static (rows of the weights W2). Thus, in attention, our outputs are a sum of scaled rows in the V matrix, and in MLP, our outputs are a sum of scaled rows in the W2 matrix (the bias is one extra static vector). We can visualize these individual vectors as being summed to produce the total operation output.

![][sum]

Chipmunk’s key insight is that the value vectors (the colored columns of **v** above) change slowly, as do the scalar weights themselves (the colored values in the attention matrix above). Chipmunk caches the value vectors and the scalar weights, and dynamically chooses which ones to recompute in each step:

![][cache]

Given an attention/MLP output cache, an equivalent definition of a normal dense forward pass on step n is the following: Subtract all of step n-1’s output vectors from the cache, and add all of step n’s new vectors. Therefore, given the natural sparsity in intermediate matrices, we can reformulate attention and MLP to compute a *delta* based on the previous step’s outputs. That is, we *replace* a subset of the output vectors and reuse the rest from the previous step. The output vectors that we replace correspond to sparsifying keys/values at the granularity of a single token in the intermediate matrix.

## Hardware-Efficient Sparsity Pattern

The sparsity pattern we’ve been describing thus far, recomputing individual scaled output vectors for each token, corresponds to [1, 1] unstructured sparsity on the intermediate activations. GPUs do not like this. What they do like is computing large blocks at once, in the size ballpark of [128, 256] (in the current generation). This corresponds to 128 contiguous tokens and 256 contiguous keys/values.

<div style="text-align: center; width: 60%; margin: 0 auto;">
<img src="https://sandyresearch.github.io/images/chipmunk/tiles.png">
</div>

Computing with block sparsity that aligns with the native tile sizes of the kernel is essentially free because the GPU is using the same large matrix multiplication sizes and skips full blocks of work.

However, there is one optimization we can make to efficiently get to [128, 1] column sparsity. Looking at our matrix multiplication diagram, let’s think through what happens if we reorder the columns of kT and **vT**. A reordering of kT will apply the same reordering to the columns of A \= q @ kT. And if we apply the same reordering to **vT**, then the end result **o** is actually the same because the columns of A still align with the correct columns of **vT.**

What this allows us to do is compute attention or MLP with any ordering of the keys/values in shared memory–thus we can pack our sparse keys/values from non-contiguous rows in global memory into a [dense tile in shared memory](https://arxiv.org/abs/2301.10936).

![][sram]

The more granular loads incur a small performance penalty, but we find that the sparsity levels make up for this–e.g. at 93% sparsity, our column-sparse attention kernel in [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) is \~10x times faster than the dense baseline.

Ok, so now we’re working with [128, 1] column sparsity, which corresponds to 128 contiguous tokens recomputing the same set of individual output vectors across steps. Intuitively, we expect that small 2D *patches* of an image have similar color and brightness. And in video, we expect the same for small 3D cubes (*voxels*). Yet, the natural token order is *raster order* from left to right, top down, and frame zero onwards. To create 128-size chunks with the most similar tokens, we **reorder** the tokens (and RoPe embeddings) once at the beginning of the diffusion process such that a **chunk** in the flattened sequence corresponds to a **patch/voxel**. These similar tokens, which we expect to interact with similar keys/values, now share the same set of sparse indices because they occupy contiguous rows of the input matrix. At the end of the diffusion process, we then reverse this reordering before decoding to pixel space.

## Kernel Optimizations

Our kernel optimizations achieve efficient dynamic sparsity and caching through:

1. **Fast sparsity identification**—fusing custom kernels to quickly compute sparse indices by reusing softmax constants and implementing a fast approximate top-k CUDA kernel with shared memory atomics, which is ≥2x faster than PyTorch’s native implementations  
2. **Fast cache writeback**—we use the CUDA driver API to overlap the cache writeback with subsequent GEMM computations by allocating leftover streaming multiprocessors (SMs) to custom TMA-based reduction kernels (with PTX instructions like `cp.reduce.async.bulk`) during the tail effects of wave quantization, achieving a 2x speedup over naive implementations and saving \~20 microseconds per MLP invocation.  
3. **Warp-Specialized Persistent Kernel:** We let the producer warpgroup’s memory loads to overlap with consumer epilogues (which are expensive because of all the caching computation), and store swizzle offsets in registers, minimizing address computation overhead when using granular `cp.async` loads instead of TMA.

## Come and play with Chipmunks!

The only thing we love more than chipmunks is the open-source community! Check out our repo at [https://github.com/sandyresearch/chipmunk](https://github.com/sandyresearch/chipmunk) and make your image and video models go brrrr. 

<div style="text-align: center; width: 50%; margin: 0 auto;">
<img src="https://sandyresearch.github.io/images/chipmunk/kittens.png" />
</div>
<center>
<i>We're big fans of ThunderKittens, and so are our chipmunks! Our sparse attention and MLP kernels let our chipmunks play nicely with their kitten friends.</i>
</center>

## What's next?

If you'd like to continue reading more, checking out Parts II and III of this series! [Part II](https://sandyresearch.github.io/chipmunk-part-II/) will build theoretical intuition for column-sparse deltas and [Part III](https://sandyresearch.github.io/chipmunk-part-III/) will be a deep dive on  GPU kernels & systems optimizations.

If you're interested, reach out! Austin (austinsilveria@gmail.com), Soham (govande@stanford.edu), Dan (danfu@ucsd.edu).


[comparison]: https://sandyresearch.github.io/images/chipmunk/comparison.png
[sum]: https://sandyresearch.github.io/images/chipmunk/sum.png
[cache]: https://sandyresearch.github.io/images/chipmunk/cache.png
[tiles]: https://sandyresearch.github.io/images/chipmunk/tiles.png
[sram]: https://sandyresearch.github.io/images/chipmunk/sram.png
[kittens]: https://sandyresearch.github.io/images/chipmunk/kittens.png
