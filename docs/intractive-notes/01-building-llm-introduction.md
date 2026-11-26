# **Notes 01 — Building Large Language Models**
**From raw text to GPT-3: tokens, pretraining, finetuning, attention, BERT vs GPT, prompting, and emergence**

---

## 1. What is an LLM?

A **Large Language Model (LLM)** is a neural network trained on a huge amount of text. Its main job is simple: **given the tokens so far, predict the next token.**

```text
Input:  The lion roams in the
Model gives a probability for every token in its vocabulary:
  jungle  0.71
  forest  0.26
  car     0.04   (numbers are only an example)
```

The model does not look up a fixed answer. It learned patterns from data and gives a **probability distribution**. We then pick (or sample) one token, add it to the input, and repeat.

### Key terms

| Term | Simple meaning |
|---|---|
| Token | A small piece of text: a word, part of a word, punctuation, or space |
| Vocabulary | The list of all tokens the model knows (GPT-3: 50,257) |
| Parameter (weight) | A number the model learns during training |
| Embedding | A list of numbers (vector) that represents a token |
| Context window | Maximum number of tokens the model can read at once |
| Training | Changing the weights to reduce prediction error |
| Inference | Using the trained model to produce output (weights do not change) |
| Loss | A score of how wrong the model's predictions are (lower is better) |

> **Important:** The model never reads words directly. Text → tokens → token IDs → embedding vectors → Transformer math. Everything inside is numbers.

---

## 2. Where LLMs fit in AI

```text
Artificial Intelligence
└── Machine Learning
    └── Deep Learning
        └── Large Language Models
```

- **AI:** machines doing tasks that need "intelligence".
- **Machine learning:** the system learns rules from data instead of being hand-coded.
- **Deep learning:** machine learning with many-layered neural networks.
- **LLM:** a deep learning model (usually a Transformer) trained on lots of text.
- **Generative AI:** models that create new content (text, images, audio, code, video). An LLM is the text branch.

### Older NLP models vs LLMs

| | Older task-specific NLP | LLMs |
|---|---|---|
| Task | One model per task (spam filter, sentiment...) | One model, many tasks |
| Data | Labelled data for each task | Huge unlabelled text first, small labelled data later |
| New task | Build/retrain a new model | Write a prompt, or finetune lightly |
| Output | Label or score | Free-form text |

---

## 3. Tokenization (very common interview topic)

Computers need numbers, so text is cut into **tokens**.

**Why not whole words?** The vocabulary would be huge, and any new word (a name, a typo) would be "unknown".
**Why not single characters?** Sequences become very long and slow.
**Solution: subword tokenization.** Common words stay whole. Rare words are split into smaller known pieces.

```text
"unbelievable"  →  "un" + "believ" + "able"    (example only; real splits vary)
```

### Byte Pair Encoding (BPE) — how it works
1. Start with single characters (or bytes) as the vocabulary.
2. Count which two neighbouring tokens appear together most often.
3. Merge them into one new token.
4. Repeat until the vocabulary reaches the target size.

GPT-2 and GPT-3 use **byte-level BPE** with a vocabulary of **50,257** tokens (256 byte tokens + 50,000 merges + 1 special end-of-text token). Byte-level means any text, in any language, can always be represented, so there is no true "unknown word" problem.

**Pipeline:** `text → tokens → token IDs → embedding vectors`

Rule of thumb for English: $1 \text{ token} \approx \tfrac{3}{4} \text{ word}$ (about 4 characters).

---

## 4. The GPT family (checked numbers)

| Model | Year | Parameters | Context | Key idea |
|---|---|---|---|---|
| GPT-1 | 2018 | ~117M | 512 | Pretrain on unlabelled text, then finetune per task |
| GPT-2 | 2019 | 124M / 355M / 774M / 1.5B | 1,024 | Bigger model, zero-shot task ability |
| GPT-3 | 2020 | 175B | 2,048 | Huge scale, few-shot in-context learning |

**GPT-1:** 12 Transformer decoder layers, hidden size 768, 12 heads, trained on BooksCorpus (about 7,000 books). About **117M** parameters. (The 124M number belongs to the smallest GPT-2, not GPT-1.)

**GPT-2:** Four sizes. The paper originally reported 117M / 345M / 762M / 1542M, but OpenAI later said those counts were slightly off, so people now use **124M / 355M / 774M / 1.5B (1558M)**. Trained on WebText (~40 GB of text from web pages linked on Reddit with decent upvotes). It showed a model can do some tasks **zero-shot**, with no task-specific training.

**GPT-3:** 175B parameters, 96 layers, hidden size 12,288, 96 heads (128 dimensions per head), context 2,048, trained on about 300B tokens. The paper (2020) trained 8 model sizes, from 125M to 175B, to study scaling.

---

## 5. Two-stage training: pretraining, then finetuning

Think of **pretraining as school** (broad general education) and **finetuning as job training** (specific skill).

### 5.1 Pretraining
Train on a very large, mixed, mostly unlabelled text collection. Task: predict the next token.

```text
Text:    The lion roams in the jungle
Input:   The lion roams in the
Target:  jungle
```

The label (the next word) is already inside the text, so no human labelling is needed. This is **self-supervised learning**. The result is a **base model** (also called a foundation model).

### 5.2 Finetuning
Continue training the base model on a smaller, targeted dataset.

| Type | Data | Result |
|---|---|---|
| Classification finetuning | Text + label (review → positive/negative) | A classifier |
| Instruction finetuning (SFT) | Instruction + good response pairs | A model that follows requests |
| Domain finetuning | Legal, medical, finance text | Better in that field |
| Preference tuning (RLHF / DPO) | Human rankings of answers | More helpful, safer, preferred style |

### 5.3 Base model vs instruction model

A base model only **continues text**. If you type a question, it may continue with more questions, because that is what text on the internet often looks like.

```text
Prompt: What is the capital of France?
Base model might write:  What is the capital of Germany? What is the capital of Italy?
Instruction model writes: The capital of France is Paris.
```

---

## 6. The full LLM development pipeline

1. **Collect and clean data:** remove duplicates, junk, and unsafe text.
2. **Tokenize:** text → token IDs.
3. **Make batches:** group sequences for efficient GPU use.
4. **Design the model:** layers, width, heads, context length, vocabulary.
5. **Pretrain:** next-token prediction on a huge corpus.
6. **Evaluate:** loss/perplexity, benchmarks, safety checks.
7. **Save checkpoints:** so training can resume or the model can be reused.
8. **Finetune / align:** instruction tuning, then preference tuning.
9. **Deploy and monitor:** serve the model, collect feedback, update safely.

### Training details worth knowing
- **Optimizer:** usually AdamW.
- **Learning rate:** short warm-up, then slow decay (often cosine).
- **Gradient clipping** stops sudden huge updates.
- **Mixed precision** (16-bit math) saves memory and time.
- Training is split across many GPUs (data, tensor, and pipeline parallelism).

### Loss function: cross-entropy
The model outputs a probability for each token in the vocabulary. The loss is:

$$L = -\ln\big(p_{\text{correct}}\big)$$

where $p_{\text{correct}}$ is the probability the model gave to the correct token.

$$L = -\ln(0.705) \approx 0.349 \quad \text{(good)}$$

$$L = -\ln(0.259) \approx 1.351 \quad \text{(worse)}$$

Lower loss $\Rightarrow$ the model gave more probability to the right answer.

### Softmax (turns scores into probabilities)

$$p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

Here $z_i$ is the raw score (logit) of token $i$. For scores $z = (3,\ 2,\ 0)$:

$$e^{3} \approx 20.086,\quad e^{2} \approx 7.389,\quad e^{0} = 1 \;\Rightarrow\; \text{total} \approx 28.475$$

$$p(\text{jungle}) = 0.705,\quad p(\text{forest}) = 0.259,\quad p(\text{car}) = 0.035 \quad (\text{they sum to } 1)$$

### Perplexity
$$\text{PPL} = e^{L} \qquad (L \text{ in natural log})$$

It roughly means "how many tokens the model is confused between". Lower is better. A loss of $0.349$ gives perplexity $e^{0.349} \approx 1.42$.

---

## 7. GPT-3 training data (verified against the paper)

GPT-3 was trained for about **300B tokens**, sampled from five datasets:

| Dataset | Tokens available | Share of available | Weight in training mix | Epochs in 300B budget (paper) |
|---|---|---|---|---|
| Common Crawl (filtered) | 410B | 82.2% | 60% | 0.44 |
| WebText2 | 19B | 3.8% | 22% | 2.9 |
| Books1 | 12B | 2.4% | 8% | 1.9 |
| Books2 | 55B | 11.0% | 8% | **0.43** |
| Wikipedia (English) | 3B | 0.6% | 3% | 3.4 |

(Total available $\approx 499$B tokens. "Epoch" = one full pass over that dataset. The epoch numbers are the ones printed in the GPT-3 paper; the mix weights are rounded, so small differences from your own arithmetic are normal.)

### What this tells us
- The model did **not** see data in proportion to size. Common Crawl was 82% of the data but only 60% of what the model saw.
- Small, cleaner datasets (WebText2, Wikipedia, Books1) were **repeated more than once**. Common Crawl was seen **less than once**.
- So **quality was deliberately favoured over raw quantity**.

### How Common Crawl was cleaned
1. **Quality filter:** a classifier was trained to tell high-quality text (WebText, books, Wikipedia) from raw Common Crawl, and was used to keep better pages. Raw data was about 45 TB compressed before filtering.
2. **Deduplication:** near-duplicate documents were removed (fuzzy matching).
3. **Add curated sources:** known high-quality datasets were mixed in.

### Data contamination
Some benchmark test questions leaked into the training data because of a bug in the filtering. The authors reported this and could not retrain because of cost. This is a real problem: if the model saw the test, scores look better than the model's true ability.

---

## 8. The Transformer

Introduced in **"Attention Is All You Need" (2017)**. It replaced RNNs and LSTMs.

**Why it won:**
- RNNs read one token at a time, so training is slow and cannot be parallelised.
- RNNs struggle to remember things from far back.
- Transformers read all tokens at once (parallel), and any token can look directly at any other token.

### Three families

| Type | Example | Attention | Best for |
|---|---|---|---|
| Encoder-decoder | Original Transformer, T5 | Encoder sees both sides; decoder generates | Translation, summarization |
| Encoder-only | BERT | Bidirectional | Classification, search, entity tagging |
| Decoder-only | GPT, Claude, Llama | Causal (left to right) | Text generation, chat, code |

### One Transformer block (decoder, GPT style)
```text
input vectors
  → LayerNorm → Masked multi-head attention → add residual
  → LayerNorm → Feed-forward network        → add residual
output vectors   (repeat this block L times: GPT-3 has 96)
```

- **Feed-forward network (FFN):** two linear layers with a non-linearity, applied to each token on its own. Usually $4\times$ wider than the model width. This is where much of the model's stored knowledge is believed to sit.
- **Residual connection:** adds the block's input to its output, so gradients flow easily in deep networks.
- **LayerNorm:** keeps activations in a stable range so training does not blow up.

### Embeddings + position
```text
token IDs → token embeddings + position information → Transformer blocks → next-token probabilities
```

**Why position information is needed:** attention treats the tokens like a bag. If you shuffle the input vectors, the attention outputs are just shuffled the same way (this property is called *permutation-equivariant*). So "dog bites man" and "man bites dog" would look the same without position information.

| Method | Used by | Idea |
|---|---|---|
| Sinusoidal | Original Transformer | Fixed sine/cosine patterns added to embeddings |
| Learned absolute | GPT-1/2/3, BERT | A learned vector for each position (0, 1, 2, ...) |
| RoPE (rotary) | Llama and most modern LLMs | Rotates query/key vectors by an angle based on position, so attention depends on **relative distance**; easier to extend to longer context |
| ALiBi | Some models | Adds a distance penalty directly to attention scores |

---

## 9. Attention, step by step

Attention lets each token decide **how much to take from every other token**.

For every token the model makes three vectors (using three learned weight matrices):

- **Query (Q):** what am I looking for?
- **Key (K):** what do I offer to be matched against?
- **Value (V):** the actual information I pass on if I get picked.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V$$

Here $d_k$ is the size (dimension) of each key vector, and $K^{\top}$ is the transpose of $K$.

**Steps:**
1. Compare each Query with every Key (dot product) → relevance scores.
2. Divide by $\sqrt{d_k}$ to keep scores from getting too large.
3. Apply softmax → weights that add up to 1.
4. Take a weighted sum of the Value vectors.

**Why divide by $\sqrt{d_k}$?** If vectors are long, dot products get big. Big numbers push softmax into an extreme "one value gets all weight" state where gradients almost vanish and training stalls. Dividing by $\sqrt{d_k}$ keeps the variance around $1$.

### Small hand example
Token "roams" gets scores and values from three tokens (simplified, one number per value):

| Token | Score | Value |
|---|---|---|
| the | 1 | 2 |
| lion | 3 | 8 |
| roams | 2 | 4 |

$$\text{softmax}([1,3,2]) = [0.090,\ 0.665,\ 0.245]$$

$$\text{output} = 0.090\times 2 + 0.665\times 8 + 0.245\times 4 = 0.18 + 5.32 + 0.98 = 6.48$$
"lion" gets about 66.5% of the attention, so the output is closest to its value.

### Multi-head attention
Instead of one attention pattern, the model runs many **heads** in parallel. Each head has its own Q, K, V weights and can learn a different relationship (grammar, pronoun links, nearby words, long-range links). Outputs are joined (concatenated) and projected back to the model width.

$$d_{\text{head}} = \frac{d_{\text{model}}}{h}$$

where $h$ is the number of heads. For GPT-3:

$$d_{\text{head}} = \frac{12{,}288}{96} = 128$$

### Causal mask (why GPT cannot see the future)
During training the model predicts every position at once. If token 3 could see token 4, it would simply copy the answer. A **causal mask** sets the scores for future tokens to $-\infty$ before softmax, which makes their weights 0.

```text
Token:      the   lion   roams   in
the          ✔
lion         ✔     ✔
roams        ✔     ✔      ✔
in           ✔     ✔      ✔       ✔
```

In formula form, a mask matrix $M$ is added to the scores before softmax:

$$\text{MaskedAttention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V, \qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

Position $i$ may look at position $j$ only if $j \le i$. Since $e^{-\infty} = 0$, the blocked positions get weight $0$.

### Cost of attention
Every token compares with every other token, so time and memory grow with $n^2$ ($n$ = sequence length). Doubling the context makes the attention part about $4\times$ more expensive. This is why long context is costly, and why tricks like FlashAttention exist (it computes exact attention but uses GPU memory smartly, so it is much faster).

---

## 10. BERT vs GPT

| | BERT (2018, Google) | GPT (OpenAI) |
|---|---|---|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional (sees left and right) | Causal (sees left only) |
| Pretraining task | Masked Language Modeling (+ Next Sentence Prediction) | Next-token prediction |
| Naturally generates text? | No | Yes |
| Best at | Understanding: classification, search ranking, NER, extractive QA | Generation: chat, writing, code, summarizing |
| Sizes | BERT-base 110M, BERT-large 340M | 117M → 175B → much bigger |

**Masked Language Modeling:** hide about 15% of tokens and ask the model to guess them. (Of the chosen tokens: 80% replaced by [MASK], 10% by a random token, 10% kept.)

**Next Sentence Prediction (NSP):** was in the original BERT, but later work (RoBERTa) showed it adds little, so most BERT-style models dropped it.

```text
"The lion roams in the ___ ."
BERT: uses words on BOTH sides to fill the blank.
GPT:  sees only "The lion roams in the" and predicts the next word.
```

**Why did decoder-only win for chat models?** Next-token prediction is one simple objective that works for any text, scales well, and gives generation directly. With enough scale and good prompts, one decoder-only model can also do the understanding tasks.

---

## 11. GPT-3 and prompting

GPT-3's big contribution: a very large model can do many tasks from **examples inside the prompt**, with **no weight updates**. This is **in-context learning**.

### GPT-3 specs

| Property | Value |
|---|---|
| Parameters | 175B |
| Layers | 96 |
| Model width | 12,288 |
| Heads | 96 (128 dims each) |
| Context | 2,048 tokens |
| Training tokens | ~300B |
| Vocabulary | 50,257 |
| Paper | *Language Models are Few-Shot Learners* (2020) |

### Prompting modes

| Mode | What you give | Weights change? |
|---|---|---|
| Zero-shot | Instruction only | No |
| One-shot | Instruction + 1 example | No |
| Few-shot | Instruction + several examples | No |
| Finetuning | Many labelled examples (training) | **Yes** |

```text
English: Hello        → French: bonjour
English: thank you    → French: merci
English: good night   → French:
```
The model sees the pattern and continues. Nothing is saved; if you start a new chat the "learning" is gone.

### Parameter count estimate
$$N \approx 12\,L\,d^{2} + V\,d$$

with $L = 96$ layers, $d = 12{,}288$ (width), $V = 50{,}257$ (vocabulary):

$$12 \times 96 \times 12{,}288^{2} \approx 173.9 \text{ billion}$$

$$50{,}257 \times 12{,}288 \approx 0.62 \text{ billion}$$

$$N \approx 174.6 \text{ billion} \quad (\text{reported: } 175\text{B})$$

Where does $12d^2$ per layer come from? Attention has 4 matrices of size $d \times d$ (Q, K, V, output), giving $4d^2$. The FFN has two matrices, $d \times 4d$ and $4d \times d$, giving $8d^2$. Total: $4d^2 + 8d^2 = 12d^2$.

### Memory and compute estimates
Weights in 16-bit (2 bytes each):

$$175\times 10^{9} \times 2 \text{ bytes} \approx 350 \text{ GB}$$

Training memory is about $16$ bytes per parameter with Adam in mixed precision (weights, gradients, and optimizer states):

$$175\times 10^{9} \times 16 \text{ bytes} \approx 2.8 \text{ TB}$$

Training compute, with $N$ parameters and $D$ training tokens:

$$C \approx 6ND = 6 \times (175\times 10^{9}) \times (300\times 10^{9}) \approx 3.15\times 10^{23} \text{ FLOPs}$$
So one GPU cannot even hold GPT-3's weights. You need many GPUs working together.

### GPT-3 limits (good to mention)
- Not trained to follow instructions; it often continued text instead of answering. (InstructGPT/ChatGPT later fixed this using SFT + RLHF.)
- Can hallucinate (sound confident but be wrong).
- Small context (2,048 tokens).
- Reflects biases in internet data.

---

## 12. Scaling laws and compute-optimal training

- **Kaplan et al. (2020):** loss falls smoothly and predictably as you increase model size, data, and compute (a power law). This encouraged building bigger models.
- **Chinchilla (Hoffmann et al., 2022):** for a fixed compute budget, many big models were **undertrained**. Model size and training tokens should grow together, roughly **20 tokens per parameter**. Chinchilla (70B params, 1.4T tokens) beat the much larger Gopher (280B).
- GPT-3 used only about **1.7 tokens per parameter** ($300\text{B} / 175\text{B} \approx 1.7$), so by Chinchilla's view it was undertrained.
- **Today:** many models are trained on far more than 20 tokens per parameter on purpose, because a smaller, longer-trained model is **cheaper to run** after training (inference cost matters).

---

## 13. From base model to assistant: SFT, RLHF, DPO

This is how GPT-3 style models became ChatGPT-style assistants (InstructGPT, 2022).

1. **SFT (Supervised Finetuning):** train on human-written instruction → good response pairs.
2. **Reward model:** humans rank several model answers; a model learns to predict which answer humans prefer.
3. **RL step (PPO):** the LLM is trained to produce answers that get high reward, with a penalty (KL) so it does not drift too far from the original model.

**DPO (Direct Preference Optimization):** a simpler alternative. It trains directly on pairs (preferred answer vs rejected answer) with no separate reward model and no RL loop.

A famous result: humans preferred the outputs of a 1.3B InstructGPT over the 175B GPT-3, showing that **alignment can matter more than size** for usefulness.

### Parameter-efficient finetuning
- **LoRA:** freeze the original weights, and learn small low-rank matrices added to some layers, so a weight matrix $W$ becomes $W + BA$, where $B$ and $A$ are small. Trains a tiny fraction (often under 1%) of parameters. The small adapter files are easy to store and swap.
- **QLoRA:** LoRA on top of a model compressed to 4-bit, so large models can be finetuned on one GPU.

---

## 14. Emergent abilities

**Definition (Wei et al., 2022):** an ability that is absent in smaller models but present in larger ones, so you could not have predicted it just by extending the smaller models' trend.

Examples often discussed: multi-step arithmetic, chain-of-thought reasoning, following instructions, some code and translation tasks.

### Why might abilities appear with scale?
The model is trained only to predict the next token. But to predict text well across millions of topics, it has to learn grammar, facts, how arguments flow, how code works, and what question-answer formats look like. Bigger models have room to store more of these patterns.

### Is emergence real? (debated — know both sides)
- **For:** Wei et al. showed tasks where performance stays near random until a certain scale, then jumps.
- **Against:** Schaeffer et al. (2023, "Are Emergent Abilities of LLMs a Mirage?") argued many jumps come from the **metric**. All-or-nothing scores such as exact-match accuracy make smooth improvement look like a sudden jump. With smoother metrics (e.g., partial credit), progress looks gradual.

**Safe interview answer:** "Emergent abilities are skills that show up in larger models but not in smaller ones. Some researchers think part of the sudden-jump effect comes from how we measure, so it is still debated."

---

## 15. Interview questions and answers

**How to read this section:** each answer has a **Short answer** (say this first in an interview), then a spaced-out **Explanation**. Calculation questions are solved step by step.

---

### A. Basics

#### Q1. What is an LLM?

> **Short answer:** A Transformer trained on a huge amount of text to predict the next token.

**Explanation**

- An LLM is a neural network, almost always a Transformer.
- It is trained with one simple task: look at the tokens so far and predict the next one.
- By doing this billions of times, it learns grammar, facts, writing style, and even reasoning patterns.
- After training, the same model can answer questions, summarize, translate, and write code. You only change the prompt.

---

#### Q2. What is a token, and why not just use words?

> **Short answer:** A token is the unit of text the model reads. It can be a word, part of a word, or punctuation.

**Explanation**

- **Whole words** need a gigantic vocabulary and cannot handle new words or typos.
- **Single characters** make sequences far too long.
- **Subword tokens** are the middle path: common words stay whole, rare words split into known pieces, so nothing is truly "unknown".

**Example (illustrative)**

```text
"unbelievable"  →  "un" + "believ" + "able"
```

---

#### Q3. What is pretraining?

> **Short answer:** The first, very large training stage, where the model learns to predict the next token on huge unlabelled text.

**Explanation**

- No human labels anything, because the next word in the text is the label.
- The result is a **base model** with broad knowledge of language and the world.
- It is the most expensive stage (millions of dollars of compute for big models).

---

#### Q4. What is finetuning?

> **Short answer:** Extra training of a pretrained model on a smaller, specific dataset to change its behaviour.

**Explanation**

- Instruction-response pairs make the model follow commands.
- Medical or legal text makes it better in that domain.
- It is much cheaper than pretraining, because the model already knows language.

---

#### Q5. Why is next-token prediction called self-supervised?

> **Short answer:** Because the labels come from the text itself, so no human labelling is needed.

**Explanation**

- In normal supervised learning, a human labels every example.
- Here, for the text "the lion roams in the jungle":
  - **Input:** everything before "jungle"
  - **Label:** "jungle"
- Labels are free and unlimited, which makes training on trillions of tokens possible.

---

#### Q6. What is a foundation model?

> **Short answer:** A large pretrained model that can be adapted to many tasks.

**Explanation**

- The idea is "train once, reuse for many things".
- It can be adapted by prompting, finetuning, retrieval (RAG), or tools.
- GPT-3 and Llama base models are examples.

---

#### Q7. What is generative AI, and where do LLMs fit?

> **Short answer:** Generative AI creates new content. LLMs are its text-focused part.

**Explanation**

- Generative AI covers text, images, audio, video, and code.
- An LLM generates new text one token at a time.

---

#### Q8. What is an embedding?

> **Short answer:** A list of numbers (a vector) that represents a token.

**Explanation**

- The numbers are learned during training.
- Tokens used in similar ways end up with similar vectors.
- The Transformer does all its math on these vectors, not on text.

---

#### Q9. What is the context window?

> **Short answer:** The maximum number of tokens the model can look at in one go.

**Explanation**

- It includes your prompt **and** the answer the model is writing.
- GPT-3's context window was $2{,}048$ tokens.
- A bigger window lets the model use longer documents, but it costs more:
  - more compute (attention grows with $n^2$)
  - more memory (the KV cache)
- Models also do not use a long context equally well. Information in the middle is often used less reliably ("lost in the middle").

---

#### Q10. What is inference, and does the model learn during it?

> **Short answer:** Inference means using the trained model. The weights stay fixed.

**Explanation**

- When a model seems to "learn" from examples in your prompt, that is **in-context learning**.
- It is temporary. It disappears when the context ends.

---

#### Q11. How do LLMs handle words they have never seen?

> **Short answer:** They split the word into smaller known pieces using subword tokenization.

**Explanation**

- With BPE, an unknown word is broken into pieces that are in the vocabulary.
- Byte-level BPE can always fall back to raw bytes, so any text can be read or written.
- Rare words may use many tokens and be handled less well.

---

#### Q12. What does the model output at each step?

> **Short answer:** A probability for every token in the vocabulary.

**Explanation**

- For GPT-3, that is $50{,}257$ probabilities that add up to $1$.
- A decoding method (greedy, temperature, top-p) then picks one token.
- That token is added to the input, and the process repeats.

---

### B. Pretraining and finetuning

#### Q13. What is the difference between pretraining and finetuning?

> **Short answer:** Pretraining teaches general language ability. Finetuning teaches a specific behaviour.

| | Pretraining | Finetuning |
|---|---|---|
| Data | Huge, broad, mostly unlabelled | Small, targeted |
| Cost | Very high | Much lower |
| Gives | General knowledge | A skill, style, or domain |

**In one line:** pretraining gives knowledge; finetuning shapes how that knowledge is used.

---

#### Q14. Why does a base model sometimes answer a question with more questions?

> **Short answer:** Its only training goal was to continue text, not to answer.

**Explanation**

- On the internet, a list of questions is often followed by more questions (think quiz pages).
- So continuing with another question is a perfectly valid prediction.
- The model was never taught that a question should get an answer.
- **Instruction finetuning** fixes this.

---

#### Q15. What is instruction finetuning?

> **Short answer:** Training on many (instruction, ideal response) pairs.

**Explanation**

- Example pair: "Summarize this article" with a good summary as the response.
- The model learns the format "user asks, assistant answers".
- It then generalizes to new instructions it has never seen.

---

#### Q16. When should you finetune, use RAG, or just prompt?

> **Short answer:** Try prompting first, add RAG for facts, finetune for style or skill.

**Explanation**

- **Prompting**
  - Fastest and cheapest.
  - Good when the task can be explained with an instruction and a few examples.
- **RAG (retrieval)**
  - Use when the model needs facts that are private, new, or change often.
  - You fetch documents and put them in the prompt.
  - Easy to update, and it can show sources.
- **Finetuning**
  - Use when you need a consistent style, format, or skill that prompting cannot reliably give.
  - Also useful for lower cost and latency at scale.
  - It is **not** a good way to add lots of new facts.

Many real systems combine all three.

---

#### Q17. What is catastrophic forgetting?

> **Short answer:** When finetuning makes the model lose abilities it had before.

**Explanation**

- It happens when you finetune too hard on narrow data.
- The model may lose general knowledge.
- **Ways to reduce it:**
  - low learning rate
  - fewer epochs
  - mix in general data
  - use LoRA, which leaves the original weights untouched

---

#### Q18. What is LoRA, and why is it popular?

> **Short answer:** It finetunes small add-on matrices instead of all the weights.

**Explanation**

- The original model is frozen.
- Small low-rank matrices are added to selected layers, and only these are trained: $W \rightarrow W + BA$.
- Benefits:
  - tiny fraction of trainable parameters
  - much less GPU memory
  - small adapter files that you can swap per task
- Quality is often close to full finetuning.

---

#### Q19. What is RLHF?

> **Short answer:** Reinforcement Learning from Human Feedback. It aligns a model with human preferences.

**Explanation (3 steps)**

1. Humans rank different model answers.
2. A **reward model** learns to predict which answers humans prefer.
3. The LLM is optimized (usually with PPO) to produce answers that get a high reward.

The result is a model that is more helpful, honest, and safe. **DPO** is a simpler alternative that trains directly on preferred vs rejected answers.

---

#### Q20. Why was a small InstructGPT preferred over the big GPT-3?

> **Short answer:** Because it was trained to follow instructions, while GPT-3 only continued text.

**Explanation**

- People care about getting a useful answer.
- A 1.3B aligned model beat a $100\times$ larger unaligned one in human preference tests.
- **Lesson:** bigger is not the only way to improve usefulness.

---

#### Q21. What is overfitting in finetuning, and how do you spot it?

> **Short answer:** The model memorizes the small training set and does worse on new data.

**How to spot it**

- Training loss keeps going **down**.
- Validation loss starts going **up**.

**Fixes**

- more data
- early stopping
- dropout or other regularization
- fewer epochs
- smaller learning rate

---

### C. Transformer and attention

#### Q22. What is the Transformer, and why did it replace RNNs?

> **Short answer:** A neural network built on self-attention. It trains faster and handles long-range links better.

**Explanation**

- **RNNs** read tokens one by one. That is slow to train and forgets distant context.
- **Transformers** read all tokens in parallel, so they train much faster on GPUs.
- Any token can attend **directly** to any other token, so long-range links are easier to learn.

---

#### Q23. What is self-attention?

> **Short answer:** Each token scores how relevant every other token is, then takes a weighted mix of their information.

**Explanation**

1. Compute relevance scores between tokens.
2. Turn the scores into weights with softmax.
3. Build a new representation as a weighted mix of the other tokens' value vectors.

**Example**

In "The animal didn't cross the street because it was tired", attention helps "it" pick up information from "animal".

---

#### Q24. Explain Q, K, and V in simple words.

> **Short answer:** Query = what I am looking for. Key = what I offer. Value = the information I pass on.

**Library analogy**

- **Query:** the topic you are searching for.
- **Key:** the label on each book.
- **Value:** the content of the book.

You compare your query with all the labels, find the best matches, and read mostly from those books.

In the model, $Q$, $K$, $V$ are all made from the same token vectors using three different learned weight matrices.

---

#### Q25. Write the attention formula and explain each part.

> **Short answer:**

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

**Explanation**

- $QK^{\top}$ gives a score for every pair of tokens.
- Dividing by $\sqrt{d_k}$ keeps scores in a stable range.
- $\text{softmax}$ turns each row into weights that sum to $1$.
- Multiplying by $V$ gives the weighted mix of value vectors.

---

#### Q26. Why divide by $\sqrt{d_k}$?

> **Short answer:** To stop dot products from getting too big, which would make softmax too sharp and learning too slow.

**Explanation**

- If $Q$ and $K$ have $d_k$ dimensions with roughly unit variance, their dot product has variance about $d_k$.
- So the numbers get larger as $d_k$ grows.
- Large inputs make softmax almost one-hot, and its gradients become tiny, so learning stalls.
- Dividing by $\sqrt{d_k}$ brings the variance back near $1$:

$$\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) \approx \frac{d_k}{d_k} = 1$$

---

#### Q27. Why does a Transformer need positional information?

> **Short answer:** Attention itself has no sense of order.

**Explanation**

- If you shuffle the input tokens, the outputs shuffle in the same way.
- So "dog bites man" and "man bites dog" would look alike.
- We therefore add position information. The main methods are:
  - learned position embeddings (GPT-2, GPT-3)
  - fixed sinusoids (original Transformer)
  - rotary embeddings, RoPE (most modern models)

---

#### Q28. What is RoPE, and why is it popular?

> **Short answer:** Rotary Position Embedding. It makes attention depend on the **relative distance** between tokens.

**Explanation**

- It rotates the query and key vectors by an angle that depends on the token's position.
- Because of how rotations work, the attention score between two tokens depends on how far apart they are.
- It works well on long sequences and can be stretched to longer contexts than the model was trained on.

---

#### Q29. What is a causal mask, and why is it needed?

> **Short answer:** A mask that stops each token from seeing later tokens.

**Explanation**

- Scores for future tokens are set to $-\infty$ before softmax, so their weights become $0$.
- Training predicts every position in parallel. Without the mask, the model could see the answer and just copy it.
- The mask makes training match real generation, where future tokens do not exist yet.

---

#### Q30. Why is GPT called autoregressive?

> **Short answer:** It generates one token at a time, using its own earlier outputs as input.

**Explanation (the loop)**

1. Predict the next token from all earlier tokens.
2. Add that token to the input.
3. Repeat.

"Auto" means self, and "regressive" means it depends on its own previous outputs.

---

#### Q31. What is multi-head attention, and why use it?

> **Short answer:** Several smaller attention operations run in parallel, each looking for different patterns.

**Explanation**

- Each head has its own $Q$, $K$, $V$ weights.
- Different heads can specialize: grammar, pronoun links, nearby words, and so on.
- Their outputs are concatenated and mixed by a final linear layer.
- It gives the model several "views" of the context at about the same cost as one big head.

---

#### Q32. What are residual connections and LayerNorm for?

> **Short answer:** Residual connections help gradients flow. LayerNorm keeps training stable.

**Explanation**

- **Residual connection:** adds a block's input to its output. This gives gradients a direct path through many layers, so very deep models can be trained.
- **LayerNorm:** rescales activations to a stable range so training does not blow up.
- GPT-2 and later put LayerNorm **before** attention and FFN ("pre-norm"), which trains more stably.

---

#### Q33. What does the feed-forward network (FFN) do?

> **Short answer:** It processes each token on its own after attention has mixed information between tokens.

**Explanation**

- It uses two linear layers and a non-linearity.
- Its hidden size is usually $4\times$ the model width.
- It holds about two thirds of the parameters.
- It is thought to store much of the model's factual knowledge.

---

#### Q34. What is the time and memory cost of attention?

> **Short answer:** $O(n^2 d)$ compute and $O(n^2)$ memory.

**Explanation**

- $n$ is the sequence length and $d$ is the model width.
- Doubling the length makes attention about $4\times$ costlier.
- This is the main reason long context is expensive.
- FlashAttention and sparse or linear variants reduce the cost.

---

#### Q35. What is the KV cache?

> **Short answer:** Stored keys and values from earlier tokens, so they are not recomputed at every step.

**Explanation**

- When generating, each new token needs the keys and values of all earlier tokens.
- Instead of recomputing them, we store them and only compute $K$ and $V$ for the newest token.
- This makes generation much faster.
- The downside: the cache grows with sequence length and batch size, so it uses a lot of GPU memory.

---

### D. BERT vs GPT

#### Q36. What is the main difference between BERT and GPT?

> **Short answer:** BERT is encoder-only and bidirectional. GPT is decoder-only and causal.

| | BERT | GPT |
|---|---|---|
| Architecture | Encoder-only | Decoder-only |
| Attention | Sees left and right | Sees left only |
| Training task | Guess masked words | Predict the next word |
| Built for | Understanding text | Generating text |

---

#### Q37. Why can BERT not generate text well?

> **Short answer:** It was trained to fill blanks using both sides, not to write left to right.

**Explanation**

- It has no natural "next token" process.
- You can force it to generate, but the results are poor compared with decoder-only models.

---

#### Q38. When would you still choose a BERT-style model?

> **Short answer:** For classification, embeddings, semantic search, and entity recognition.

**Explanation**

- It is especially good when you need a small, fast, cheap model.
- A $110$M-parameter BERT-style model can beat a huge LLM on cost and speed for a simple, well-defined understanding task.

---

#### Q39. Why do most modern chatbots use decoder-only models?

> **Short answer:** One simple objective works on any text, scales very well, and gives generation directly.

**Explanation**

- Next-token prediction needs no special labels.
- It scales well with data and compute.
- Larger decoder-only models also learned to do understanding tasks through prompting, so one model replaces many.

---

### E. GPT-3, data, and prompting

#### Q40. What are the key facts about GPT-3?

> **Short answer:** $175$B parameters, trained on about $300$B tokens, released in 2020.

| Property | Value |
|---|---|
| Parameters | $175$B |
| Layers | $96$ |
| Model width | $12{,}288$ |
| Attention heads | $96$ |
| Context length | $2{,}048$ tokens |
| Training tokens | about $300$B |
| Vocabulary | $50{,}257$ |

**Big finding:** at this scale, a model can learn tasks from a few examples in the prompt, without weight updates.

---

#### Q41. Explain zero-shot, one-shot, and few-shot.

> **Short answer:** The number of examples you put in the prompt: none, one, or several.

- **Zero-shot:** instruction only.
- **One-shot:** instruction plus one example.
- **Few-shot:** instruction plus several examples.

In all three, the **weights stay fixed**. The examples just sit in the context and guide the output. Few-shot usually works better than zero-shot, especially for unusual formats.

---

#### Q42. Is few-shot prompting the same as finetuning?

> **Short answer:** No. Few-shot changes only the input. Finetuning changes the weights.

| | Few-shot prompting | Finetuning |
|---|---|---|
| What changes | The prompt | The model weights |
| Stored after the chat? | No | Yes |
| Cost | Instant and free | Needs data and compute |
| Behaviour | Can be less consistent | Often more consistent |

---

#### Q43. Common Crawl is 82% of GPT-3's data but only 60% of the training mix. What does this show?

> **Short answer:** GPT-3's creators controlled exposure on purpose. Quality and mixture matter, not just quantity.

**Explanation**

- Raw web text is huge but noisy, so Common Crawl was **downweighted** (seen only $0.44$ times).
- Smaller, cleaner sources were **upweighted**:
  - WebText2 was seen $2.9$ times.
  - Wikipedia was seen $3.4$ times.

---

#### Q44. How was GPT-3's data cleaned?

> **Short answer:** Quality filtering, then deduplication, then adding curated datasets.

**Explanation**

1. A quality classifier was trained to prefer text similar to curated sources, and was used to filter Common Crawl.
2. Near-duplicate documents were removed (fuzzy deduplication).
3. High-quality datasets were added to the mix.

Even so, some benchmark data leaked into training (**contamination**), which the authors acknowledged.

---

#### Q45. Estimate GPT-3's parameters from its shape.

> **Short answer:** About $174.6$B, very close to the reported $175$B.

**Formula**

$$N \approx 12Ld^2 + Vd$$

**Given:** $L = 96$, $d = 12{,}288$, $V = 50{,}257$.

**Step 1: transformer layers**

$$12 \times 96 \times 12{,}288^2 \approx 173.9 \text{ billion}$$

**Step 2: embedding table**

$$50{,}257 \times 12{,}288 \approx 0.62 \text{ billion}$$

**Step 3: add them**

$$N \approx 173.9 + 0.62 \approx 174.6 \text{ billion}$$

**Where does $12d^2$ come from?**

- Attention has 4 matrices of size $d \times d$ (Q, K, V, output): $4d^2$.
- The FFN has two matrices, $d \times 4d$ and $4d \times d$: $8d^2$.
- Total per layer: $4d^2 + 8d^2 = 12d^2$.

---

#### Q46. How much memory do GPT-3's weights need?

> **Short answer:** About $350$ GB in 16-bit precision, for the weights alone.

**Step 1: bytes per parameter.** 16-bit means $2$ bytes each.

**Step 2: multiply**

$$175 \times 10^{9} \times 2 \text{ bytes} = 350 \times 10^{9} \text{ bytes} = 350 \text{ GB}$$

**Step 3: what this means.** No single GPU holds this, so the model must be split over several GPUs.

**Training needs far more.** About $16$ bytes per parameter (weights, gradients, and optimizer states):

$$175 \times 10^{9} \times 16 \text{ bytes} \approx 2.8 \text{ TB}$$

This is before activations and overhead.

---

#### Q47. How many tokens per parameter did GPT-3 see, and why does it matter?

> **Short answer:** About $1.7$. Chinchilla showed this is too low for best results.

**Calculation**

$$\frac{300\text{B tokens}}{175\text{B parameters}} \approx 1.7 \text{ tokens per parameter}$$

**Why it matters**

- Chinchilla found that for a fixed compute budget, models should see about $20$ tokens per parameter.
- So GPT-3 was undertrained by that measure.
- A smaller model trained on more data can match or beat it.

---

### F. Calculations

#### Q48. Scores are $[1, 3, 2]$ and values are $[2, 8, 4]$. What is the attention output?

> **Short answer:** $6.48$.

**Step 1: exponentiate the scores**

$$e^{1} \approx 2.718, \qquad e^{3} \approx 20.086, \qquad e^{2} \approx 7.389$$

**Step 2: add them**

$$2.718 + 20.086 + 7.389 = 30.193$$

**Step 3: divide each by the total (softmax weights)**

$$\frac{2.718}{30.193} = 0.090, \qquad \frac{20.086}{30.193} = 0.665, \qquad \frac{7.389}{30.193} = 0.245$$

**Step 4: weighted sum of the values**

$$\text{output} = 0.090 \times 2 + 0.665 \times 8 + 0.245 \times 4$$

$$= 0.18 + 5.32 + 0.98 = 6.48$$

The token with score $3$ gets about $66.5\%$ of the attention, so the output is closest to its value ($8$).

---

#### Q49. The correct token has probability $0.705$. What is the loss? What if it is $0.259$?

> **Short answer:** $0.349$ and $1.351$.

**Formula**

$$L = -\ln(p)$$

**Case 1:** $p = 0.705$

$$L = -\ln(0.705) \approx 0.349 \quad \text{(good)}$$

**Case 2:** $p = 0.259$

$$L = -\ln(0.259) \approx 1.351 \quad \text{(worse)}$$

**Meaning:** lower probability on the right answer gives higher loss, so training pushes the model to raise it.

---

#### Q50. What is perplexity?

> **Short answer:** $e$ raised to the cross-entropy loss. It measures the model's "effective confusion".

**Formula**

$$\text{PPL} = e^{L}$$

**Explanation**

- A perplexity of $10$ means the model is about as unsure as if it were choosing uniformly between $10$ tokens.
- Lower is better.
- You can only compare perplexity across models that use the same tokenizer and test data.

**Example**

$$L = 0.349 \;\Rightarrow\; \text{PPL} = e^{0.349} \approx 1.42$$

---

### G. Decoding and generation

#### Q51. What is greedy decoding, and what is its problem?

> **Short answer:** Always picking the most probable next token.

**Explanation**

- It is deterministic and fast.
- But it can produce repetitive or dull text.
- A locally best choice can lead to a poor sentence overall.

---

#### Q52. What does temperature do?

> **Short answer:** It controls randomness by sharpening or flattening the probabilities.

**Formula.** Before softmax, the scores are divided by $T$:

$$p_i = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$$

**Effect**

- **Low $T$ (like $0.2$):** sharper distribution, so the model picks likely tokens. Focused and predictable.
- **High $T$ (like $1.2$):** flatter distribution. More varied and creative, but more errors.
- **$T \to 0$:** behaves like greedy decoding.

---

#### Q53. What are top-k and top-p (nucleus) sampling?

> **Short answer:** Two ways to sample only from the most likely tokens.

- **Top-k:** keep only the $k$ most likely tokens, then sample among them.
- **Top-p:** keep the smallest set of tokens whose probabilities add up to $p$ (say $0.9$), then sample from that set.

**Why top-p adapts well:** when the model is sure, the set is tiny. When it is unsure, the set is larger.

---

#### Q54. What is beam search?

> **Short answer:** Keeping several candidate sequences at each step and returning the best overall.

**Explanation**

- It keeps several candidates (the beams) at each step.
- At the end it returns the one with the best total probability.
- It is common in translation.
- For open-ended chat it tends to give bland text, so sampling is more common.

---

#### Q55. What are teacher forcing and exposure bias?

> **Short answer:** Training uses the true previous tokens. Generation uses the model's own, which may be wrong.

**Explanation**

- **Teacher forcing:** in training, the model is always given the **true** previous tokens. This is stable and fast.
- **Exposure bias:** in generation, the model must use its **own** earlier tokens, which may contain mistakes.
- This train/test mismatch means errors can snowball.

---

### H. Evaluation, risks, and practical issues

#### Q56. How do you evaluate an LLM?

> **Short answer:** Use several methods together. No single metric is enough.

- **Language modeling quality:** loss and perplexity.
- **Benchmarks:** MMLU (knowledge), GSM8K (math), HumanEval (code).
- **Task metrics:** accuracy or F1 for classification, ROUGE for summaries, BLEU for translation.
- **Human evaluation** and preference comparisons.
- **LLM-as-a-judge:** used carefully, since judges have biases.
- **Safety, bias, and robustness** tests.

**Warning:** benchmark scores can be inflated by data contamination.

---

#### Q57. What is hallucination, and how do you reduce it?

> **Short answer:** The model states something false or invented with confidence.

**Why it happens:** the model is trained to produce likely-sounding text, not to check truth.

**Ways to reduce it**

- Ground answers with retrieval (RAG) and ask for sources.
- Lower the temperature.
- Give clear instructions that allow "I don't know".
- Use tools (search, calculators).
- Verify important outputs.

---

#### Q58. What are the main challenges of LLMs?

> **Short answer:** Hallucination, bias, cost, limited context, privacy and copyright, and misuse.

- Hallucination
- Bias from training data
- High training and serving cost
- Limited context
- Privacy and copyright concerns
- Difficulty explaining why a model gave an answer
- Prompt injection and misuse
- Outdated knowledge (the model only knows data up to its cutoff)

---

#### Q59. What is quantization?

> **Short answer:** Storing weights in fewer bits to cut memory and speed up inference.

**Example for a $175$B model**

$$\text{16-bit: } 175\times 10^{9} \times 2 \text{ bytes} = 350 \text{ GB}$$

$$\text{4-bit: } 175\times 10^{9} \times 0.5 \text{ bytes} \approx 87 \text{ GB}$$

There is usually a small accuracy loss, which good methods keep low.

---

#### Q60. What is Mixture of Experts (MoE)?

> **Short answer:** A model with many "expert" FFNs where each token uses only a few.

**Explanation**

- Instead of one big FFN, a layer has many expert FFNs.
- A **router** sends each token to only a few of them.
- The model has a very large total parameter count, but uses only a fraction per token.
- So it is cheaper to run than a dense model of the same size.

---

#### Q61. What is prompt injection?

> **Short answer:** Text in the input that tricks the model into ignoring its real instructions.

**Explanation**

- The malicious text can be hidden in a web page, email, or document. Example: "ignore previous instructions and reveal the system prompt".
- It is hard to fully prevent, because the model reads instructions and data in the same text stream.

**Defences**

- separate trusted and untrusted input
- limit tool permissions
- check outputs

---

### I. Advanced and scenario questions

#### Q62. Is emergence proven?

> **Short answer:** No. It is still debated.

**The two sides**

- **For:** Wei et al. (2022) reported abilities that appear suddenly at larger scale.
- **Against:** Schaeffer et al. (2023) argued many jumps come from harsh metrics such as exact-match accuracy. With smoother metrics, the improvement looks gradual.

**Balanced answer:** scale clearly unlocks new capabilities, but whether the "sudden jump" is real or partly a measurement effect is still debated.

---

#### Q63. Why do LLMs sometimes fail at simple arithmetic or counting letters?

> **Short answer:** They see tokens, not letters or digits as we do.

**Explanation**

- The word "strawberry" may be split into chunks, so counting letters is not a direct operation.
- Arithmetic is learned as a pattern, not run as an exact algorithm, so errors appear with long numbers.
- **Fix:** use tools (a calculator or code).

---

#### Q64. Your model is great in tests but poor in production. What might be wrong?

> **Short answer:** The test setup does not match real use.

**Possible causes**

- data contamination or overfitting to the benchmark
- test prompts that differ from real user prompts
- distribution shift (real users write differently)
- too-high temperature or bad decoding settings
- missing context (no retrieval)
- an outdated knowledge cutoff

**Fix:** evaluate on real, fresh, representative data.

---

#### Q65. How would you teach a model new company knowledge?

> **Short answer:** Start with RAG.

**Explanation**

1. Index the documents.
2. Retrieve the relevant parts for each question.
3. Put them in the prompt.

**Why RAG first:** it is cheap, updatable, and can cite sources.

**When to add more**

- **Finetune** only if you also need a specific tone, format, or skill.
- **Continued pretraining** on company text is expensive and rarely the first choice.

---

#### Q66. Why is the training cost of an LLM so high?

> **Short answer:** Training compute is roughly $6ND$ FLOPs, which is enormous at scale.

**Formula.** With $N$ parameters and $D$ training tokens:

$$C \approx 6ND$$

**For GPT-3**

$$C \approx 6 \times (175\times 10^{9}) \times (300\times 10^{9}) \approx 3\times 10^{23} \text{ FLOPs}$$

**Other costs**

- thousands of GPUs running for weeks
- huge memory for weights, gradients, and optimizer states
- data collection, cleaning, and evaluation
- failed runs

---

#### Q67. Why can't an LLM simply "look up" a stored document?

> **Short answer:** A base LLM stores what it learned as numbers in its weights, not as a database of documents.

**Explanation**

- It can reproduce some memorized text.
- But it cannot guarantee exact retrieval.
- If you need exact facts or sources, add a retrieval system (RAG).

---

---

## 16. Common misconceptions

| Wrong idea | Correct idea |
|---|---|
| "LLMs understand language like humans." | They learn statistical patterns very well; whether that equals understanding is debated. |
| "Attention knows word order by itself." | No. Position information must be added. |
| "Few-shot learning updates the weights." | No. Only finetuning/training changes weights. |
| "More data always means a better model." | Quality, diversity, deduplication, and mixture weights matter. |
| "GPT-3 saw all Common Crawl tokens equally to others." | No. Common Crawl was 82% of available tokens but 60% of the mix and was seen less than once. |
| "BERT and GPT are totally different." | Both use Transformer parts; they differ in masking, attention direction, and training goal. |
| "Bigger is always better." | Alignment and training tokens matter; Chinchilla and InstructGPT both show this. |
| "Emergence is settled science." | It is debated, especially around metrics. |
| "An LLM stores and retrieves documents." | It generates from learned patterns; retrieval needs an extra system like RAG. |
| "Finetuning is the best way to add new facts." | Retrieval (RAG) is usually better for facts; finetuning is better for style/skill. |

---

## 17. One-page summary

| Idea | One line |
|---|---|
| LLM | A Transformer trained on lots of text to predict the next token |
| Token | The small text unit the model reads (word piece) |
| BPE | Builds a subword vocabulary by merging frequent pairs |
| Pretraining | Next-token prediction on huge unlabelled text (self-supervised) |
| Finetuning | More training on a smaller, targeted dataset |
| SFT / RLHF / DPO | Steps that turn a base model into a helpful assistant |
| Attention | Weighted mixing of token information by relevance |
| $\sqrt{d_k}$ | Keeps attention scores from getting too large |
| Positional info | Needed because attention is order-blind |
| Causal mask | Stops tokens from seeing the future |
| BERT | Encoder-only, bidirectional, masked-word training |
| GPT | Decoder-only, causal, next-token training |
| In-context learning | Learning from prompt examples; weights unchanged |
| GPT-3 | 175B params, 96 layers, 300B tokens, 2,048 context |
| Chinchilla | About 20 tokens per parameter is compute-optimal |
| LoRA | Finetune small low-rank add-ons instead of all weights |
| KV cache | Stores past keys/values to speed up generation |
| Emergence | Abilities appearing at scale; cause and measurement are debated |

---

## 18. Sources

1. Brown et al., *Language Models are Few-Shot Learners* (GPT-3), 2020, arXiv:2005.14165 — model specs, dataset table (tokens, mix weights, epochs 0.44 / 2.9 / 1.9 / 0.43 / 3.4), data filtering, contamination.
2. Radford et al., *Improving Language Understanding by Generative Pre-Training* (GPT-1), 2018.
3. Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019. Parameter counts corrected in OpenAI's GPT-2 repository (124M / 355M / 774M / 1558M).
4. Vaswani et al., *Attention Is All You Need*, 2017, arXiv:1706.03762.
5. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, 2018, arXiv:1810.04805.
6. Wei et al., *Emergent Abilities of Large Language Models*, 2022, arXiv:2206.07682.
7. Schaeffer, Miranda, Koyejo, *Are Emergent Abilities of Large Language Models a Mirage?*, 2023, arXiv:2304.15004.
8. Kaplan et al., *Scaling Laws for Neural Language Models*, 2020; Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), 2022.
9. Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), 2022; Rafailov et al., *Direct Preference Optimization*, 2023.
10. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
11. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
12. Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, 2023.
13. DataCamp, *Top LLM Interview Questions and Answers for 2026* (used to check which topics interviewers commonly ask: Transformer, attention, context window, pretraining objectives, fine-tuning, tokenization/OOV, embeddings, positional encoding/RoPE, evaluation metrics, challenges).