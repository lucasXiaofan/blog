---
title: What is Typical Sampling
summary: journey from speculative sampling to typical sampling
date: 2024-3-22
authors:
  - admin
tags:
  - Transformer Inference
  - Truncate Sampling
image:
  caption: 'Image credit: [**Unsplash**](https://unsplash.com)'
---

# 🎯Motivation:
1. speculative sampling[[#🪙references| 2]] [[2| ]]是一个很麻烦的sampling 方法，其中最让我想要优化的方面就是speculative sampling需要比较target model和draft model的logits。由于我目前的研究是异构大模型推理加速，互相传输logits是一个可以被优化的点，而从medusa [[#🪙references| 1]]  论文提出的typical sampling就没有互相传输logits的必要，所以我想更深入的学习typical sampling看看它的数学解释，并希望未来能用LLM的benchmark来测试typical sampling和speculative sampling的差异
2. 
# ✅ Prerequisite：
1. 理解speculative sampling/decoding
3. 理解transformer inference
4. 基础的machine learning 知识
	1. softmax
	2. logits
# 🤨Expectation： 
1. 这只是我学习typical sampling的学习笔记，我还没有完全理解透彻typical sampling是什么，若有错误和不懂的，欢迎指正与讨论
2. 这篇文章会很长，而且必需很长，因为这就是科研的厚重，短一点都会产生很多疑惑

---
# Content: 
## 使用typical sampling的动机：
medusa [[#🪙references| 1]]  论文提出了typical acceptance这个概念，主要原因是: " speculative sampling results in diminished efficiency as the sampling temperature increases " 然后medusa 给这个原因的更深的解释是即时draft model 和target model一模一样，因为draft 和target model “sample independently” draft model的结果还是会被target model 拒绝掉。

这是文章中的原话
```
In speculative decoding papers [Leviathan et al., 2022, Chen et al., 2023], authors employ rejection sampling to yield diverse outputs that align with the distribution of the original model. However, subsequent implementations [Joao Gante, 2023, Spector and Re, 2023] reveal that this sampling strategy results in diminished efficiency as the sampling temperature increases. Intuitively, this can be comprehended in the extreme instance where the draft model is the same as the original one. Here, when using greedy decoding, all output of the draft model will be accepted, therefore maximizing the efficiency. Conversely, rejection sampling introduces extra overhead, as the draft model and the original model are sampled independently. Even if their distributions align perfectly, the output of the draft model may still be rejected.
```

就这一段话，信息量极大，需要彻底理解
1. speculative decoding的algorithm和数学证明
2. 以及temperature 在transformer inference里面究竟有什么作用，


其实说实话，在我写这个learning note的时候，我懂 speculative decoding的algorithm和数学证明，但没懂透，还差临门一脚，至于temperature在transformer inference里面到底有什么用，我就只知道非常模糊的概念：
1. 什么temperature和transformer 的creativity有关啊
2. temperature 在softmax里面有用到

#### temperature 在transformer inference里面意味着什么： 
我们先来解释temperature，
直接google “temperature in transformer inference” 就有一个很契合我们的来自huggingface forum的答案 https://discuss.huggingface.co/t/what-is-temperature/11924 ，在这个答案中有两个资源可以解释我们的答案
1. stackoverflow的答案：
	1. https://stackoverflow.com/questions/58764619/why-should-we-use-temperature-in-softmax/63471046#63471046
2. huggingface 的blog，这个答案就跟贴切transformer inference，而且还讲了temperature之外的，和transformer inference有关的内容，比如greedy search，top-k，top-p： 
	1. https://huggingface.co/blog/how-to-generate
读完两个resource之后总结一下，temperature的主要任务是控制一个distribution的confidence level。temperature越高，distribution的confidence level就低，而temperature越低（至少要大于0），distribution的confidence level就高

比如说：（从stackoverflow的答案里面的例子）
[0.01,0.01,0.98] 就是一个confidence level非常高的distribution，sample 100 遍，预计98次都会是选择0.98概率的选项

[0.2,0.2,0.6] 就是一个相对confidence level毕竟低的distribution，sample 100 遍，预计60次回选0.6的选项

而temperature在码里面怎么实现呢？
``` python
logits = logits / temperature
probs = F.softmax(logits, dim=1)
```
就softmax之前把logits除以temperature就可以了
所以控制confidence level的意义是什么呢？意义是让text generation能有更多不同的生成，如果每一次LLM的output distribution的confidence level都非常高，那么生成的文字就会出现很多重复的，规律性过强的特征。当然生成文字的质量，深究的话就是玄学了，建议是直接跑一个benchmark看看分数。

但是话谈到这里，我其实是有一个疑问的，就是当前LLM pretrain的objective function其实非常简单，目的就是maximize negative log likelihood of next token，而这么做的目的就是为了让每一个transformer output distribution变得非常confident，但很多时候我们又要控制confidence以及其他的奇技淫巧来获得更高质量的文字生成结果，这是不是恰恰说明了目前的objective function太肤浅了，我们需要一个更有解释性的，更复杂的objective function 来做LLM的pretrain？

⚠️ 不过，如果你读huggingface transformer inference blog和medusa的paper毕竟仔细的话不难发现，他们都说当temperature =0的时候sampling是greedy。
from huggingface transformer inference blog: 
```
OK. There are less weird n-grams and the output is a bit more coherent now! While applying temperature can make a distribution less random, in its limit, when setting `temperature` →0→0, temperature scaled sampling becomes equal to greedy decoding and will suffer from the same problems as before.
```

from medusa paper: 
```
Examining this scheme leads to several insights. Firstly, when the temperature is set to 0, it reverts to greedy decoding, as only the most probable token possesses non-zero probability. As the temperature surpasses 0, the outcome of greedy decoding will consistently be accepted with appropriate ϵ, δ, since those tokens have the maximum probability, yielding maximal speedup. Likewise, in general scenarios, an increased temperature will correspondingly result in longer accepted sequences, as corroborated by our experimental findings.
```

但是在stackoverflow对temperature的解答里面temperature =1的时候，sampling才是greedy （虽然对stackoverflow的答案里面没有说greedy，但只要logits，softmaxed distribution没有任何改动，直接sample highest probability 就是greedy）
```
if it is 1, the output distribution will be the same as your normal softmax outputs
```
而huggingface自己的transformer library对temperature的定义是这样的：
https://huggingface.co/transformers/v3.4.0/_modules/transformers/generation_utils.html
``` python
if do_sample: # Temperature (higher temperature => more likely to sample low probability tokens) 
	if temperature != 1.0: 
		scores = scores / temperature
```
不难看出，首先temperature肯定不能是0，其次temperature = 1，对distribution没有影响，但着也不是什么大事，0，和1就是一个不同人写码的convention吧。而且这样不妨碍，temperature 高，confidence level低，temperature低，confidence level 高
```
exp(6) ~ 403
exp(3) ~ 20

t = 1.5
exp(6/1.5) ~ 54
exp(3/1.5) ~ 7.4

t = 0.5
exp(6/0.5) ~ 162754.791419
exp(3/0.5) ~ 403.428793493
```
