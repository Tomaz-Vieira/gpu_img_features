# Experiments with GPU pixel classification

*Very rough* experiments on running random forest pixel classification on the GPU

## What

This project does pixel classification (not training yet) completely on the GPU: compute pixel features, run random forest on those fatures and classify the pixel.

## How

1. In python, I created a scikit Random Forest and trained it on features that are Gaussian Smooths of the original image
    1. Export the trees in the forest via `export_graphviz` (I saved a forest in `10_feats_tree/`)
    2. This is just so we don't have to implement the training ourselves, and so we can compare results with `scikit`
2. In Rust, we generate the Gaussian Smoothing (`gaussian_blur.rs`) kernels matching the Random Forest training
   1. Then, we upload those kernels to the GPU (`KernelsInBuffSlot::new`) as GPU buffers for now,
      but maybe we should use textures instead? Would samplling be faster?
3. Parse the output of step 1.1 in `src/decisiton_tree.rs`
4. Generate a compute shader that applies every kernel to every pixel and stores each "feature"
    in a compute shader variable called `feature_<feature_index>`
5. Use the parsed trees from 1.1 to produce a bunch of `if/else` statements at the end of the compute shader
   via `forest.write_wgsl()`, which classify the pixel
6. Send an image over to the GPU, run the compute shader on it, then copy the results back

## Why?

### Features are computed in order, for every pixel, instead of separately

So we load the first pixel, compute all features for the first pixel, then classify the first pixel. Then we load the second pixel, compute all features for the second pixel, then classify the second pixel, and so on. I'd like to believe we'd getter better cache usage computing features like this instead of _"all pixels for the first feature, then all pixels for the second feature, and so on"_, because this requires loading the raw data pixels as many times as there are features (I still don't know what the caching effects of fetching the halos are, though).

### The GPU has *a lot* of streaming processors and Pixel Classification is embarassingly parallel

With this approach we are parallellizing at the pixel level. Also, a GPU can produce 4K real-time graphics these days; that's `3840 × 2160` pixels every `16ms`, which surely we could use somehow, but:
- There is the monstrous caveat that when gaming, one is not moving all those pixels into and out of the GPU
  - But we also wouldn't need to keep moving data if we could render our results directly to the screen.
  - Also, you can run real-time applications "in software", say via LLVMPIPE, and that is somehow shipped over to the GPU pretty quickly


## What needs work?

### CRITICAL: Moving stuff in and out of the GPU is way too slow

It takes the majority of the processing time and feels kinda wrong (see comments about LLVMPIPE above)

### Decent measuring of the compute/transfer timing

Right now I just take a timestamp where I _think_ makes sense

### There is only Gaussian blur for now, we need other features

Hessian of Gaussian, Structure Tensor eigenvalues, etc all need an implementation

### Gaussian blur is done in the most naïve way possible.

No separated kernels, for example. Also, maybe it would be faster to compute the kernel value on the fly instead of reading it out of a buffer or texture

### Are we even maxing out the GPU or at least the PCIe bus?

How do we even test for that? Maybe just firing multiple processes would give us an idea of how much bandwidth/compute we're wasting

### Is running on an integrated GPU better than running on the CPU?

We know that transfer speeds are faster in this case, as we'd expect

### Is running on the software implementation (LLVMPipe) better than running on static CPU code?

### Is wgpu holding us back?

The `wgpu` library (and maybe the WebGPU spec itself) is far more high level than something like Vulkan, and it lacks some features that could help with performance; In particular, it only has one submission queue (where you put work for the GPU to do), and this might hinder our ability to upload more data while the previous batch is still crunching, but I'm not sure.


## What has been tried already

### Multiple kernel buffers vs a single buffer for all kernels

There is a limit to how many buffers you're allowed to have in a shader, but separating the kernels in independent buffers or putting them all into a single giant buffer made no difference

### Running multiple pipelines in multiple threads

I think the `wgpu` library is internally locked/synchronized anyway, and this made no difference in throughput

### Splitting images into small tiles vs sending just a gigantic image

No difference, except that the small tiles would give better latency, since the user could see gradual progress

