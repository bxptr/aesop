# aesop 

AESOP is a tiny neural story generator that runs fully offline on a TI-84 Plus CE Python Edition. The checkpoint included in `aesop/checkpoints/model.npz` is an int-quantized H128 token RNN with CE-native sampling, paging, persistent RNG, and a recurrent-step asm megakernel. I've done sweeps and playing around with hyperparameters and memory constraints on CE and this seems like the best performance one can get, at least with these somewhat naive optimizations and an RNN architecture. A larger H160 model errors out because of insufficient memory.

The model included has a vocab size of 384 and 48 latent buckets. I've taken the time to setup a recurrent-step megakernel and a H128 output row-dot kernel to allow for impressively fast execution. The included trained checkpoint is around 155k parameters, which I found impressive for the z380 CPU on the CE. For 208 characters, the benchmarked CPU cycles on my CE is `423,018,983`.

Put [CEdev](https://ce-programming.github.io/toolchain/) at `tools/CEdev` and install the host dependencies via npm. See the `Makefile` for build, evaluation, and deploy commands.

Before you run AESOP on the TI, export and download [ASMHook2](https://github.com/RoccoLoxPrograms/AsmHook2) onto it. If, for whatever reason, you don't wish to use ASMHook2, you can use [arTIfiCE](https://yvantt.github.io/arTIfiCE/) to as a launcher. Nonetheless, on a successful run, you should see a loading bar as the model shards are loaded and warmed up with a selection of hidden latent tokens, and then a stream of tokens generating. 

* `ENTER` generates a new story.
* `RIGHT` takes you to the next page. This can either continue generating the running story or, if the story has been completed, generate a new story from scratch.
* `LEFT` takes you to the previous page.
* `MODE` or `CLEAR` allows you to exit, including during generation.

