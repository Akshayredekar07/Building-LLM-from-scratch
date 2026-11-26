# **Notes 02 — Tokenization for LLMs**

**From raw text to token IDs: word, character, subword, BPE, and how tokens become training data**

---

## 1. Why tokenization?

A neural network can only work with numbers, so text must be converted first.

```text
Raw text → Tokens → Token IDs → Embedding vectors (+ position info) → Transformer → Output
```

1. **Tokenize:** split text into small pieces called tokens (words, parts of words, punctuation, spaces).
2. **Token IDs:** map each token to an integer using a fixed table called the **vocabulary**.
3. **Embeddings:** map each ID to a vector (a list of numbers). The Transformer works only on these vectors.

```text
Text:       "This is an example"
Tokens:     [This] [ is] [ an] [ example]      (GPT-style tokens keep the leading space)
Token IDs:  [ 1001 ] [ 56 ] [ 78 ] [ 934 ]     (illustrative numbers, not real GPT IDs)
```

The **vocabulary** is the table of all tokens and their IDs. It is built once from the training data and then frozen. The model and its tokenizer always go together: if you change the tokenizer, the IDs change and the trained model becomes useless.

---

## 2. A simple tokenizer (using "The Verdict")

The running example is a short story, "The Verdict" by Edith Wharton (20,479 characters). Real LLMs train on hundreds of gigabytes of text (GPT-3: about 300 billion tokens), but one short story is enough to learn the pipeline.

### Step 1: split text with a regular expression

```python
import re
text = "Hello, world. Is this-- a test?"
tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
tokens = [t.strip() for t in tokens if t.strip()]
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

**Keep or remove whitespace?**
- Removing it saves memory and compute.
- Keeping it matters when exact layout counts, for example Python code where indentation has meaning.
- GPT-style tokenizers keep whitespace, usually attached to the front of a token (" the").

### Step 2: build the vocabulary

1. Tokenize the text.
2. Take the unique tokens and sort them.
3. Give each one an integer ID.

```python
vocab = {tok: i for i, tok in enumerate(sorted(set(tokens)))}
```

### Step 3: encode and decode

- `encode(text)`: text → token IDs (uses the vocabulary).
- `decode(ids)`: token IDs → text (uses the inverse dictionary, ID → token).

You need both: the model outputs IDs, and you must turn them back into readable text.

### Worked example

Training text: `The quick brown fox jump over the lazy dog`

Sorted unique tokens (Python puts capital letters before small letters, so "The" comes first):

| Token | ID |
|---|---|
| The | 0 |
| brown | 1 |
| dog | 2 |
| fox | 3 |
| jump | 4 |
| lazy | 5 |
| over | 6 |
| quick | 7 |
| the | 8 |

Add two special tokens at the end: `<|unk|>` = 9 and `<|endoftext|>` = 10.

```text
encode("the lazy dog")        →  [8, 5, 2]
decode([8, 5, 2])             →  "the lazy dog"

encode("Fox chased the dog")  →  [9, 9, 8, 2]
```

Why did "Fox chased the dog" fail?
- "Fox" is not in the vocabulary (capital F; only "fox" exists, ID 3).
- "chased" never appeared in the training text.
- A plain word vocabulary cannot represent unseen words. This is the **out-of-vocabulary (OOV)** problem. Both words became `<|unk|>`, and decoding gives `<|unk|> <|unk|> the dog`, so the information is lost forever.

---

## 3. Special tokens

| Token | Purpose |
|---|---|
| `<\|unk\|>` | Replaces words not in the vocabulary |
| `<\|endoftext\|>` | Marks the boundary between independent documents |
| `<\|bos\|>` / `<\|eos\|>` | Mark the start and end of a sequence (used by many models) |
| `<\|pad\|>` | Fills shorter sequences so every sequence in a batch has the same length |

- When joining many documents into one training stream, add `<|endoftext|>` after each one so the model learns that the topic has changed.
- **Padding:** batches must be rectangular, so short sequences get pad tokens. An **attention mask** tells the model to ignore them.
- **GPT-2 note:** it uses only `<|endoftext|>` (token ID 50256). It does not use `<|unk|>`, `<|bos|>` or `<|pad|>` in the basic setup. Byte-level BPE can write any text from bytes, so `<|unk|>` is not needed.
- **Chat models** add more special tokens to mark roles (system, user, assistant). They are reserved tokens that normal text should not be able to create.

---

## 4. Three tokenization families

### 4.1 Word-based
Each word is one token: "My hobby is playing cricket" → [My, hobby, is, playing, cricket].

| Pros | Cons |
|---|---|
| Each token has meaning | Huge vocabulary (every form of every word) |
| Short sequences | Severe OOV problem: new words, names, typos |
| Simple | "boy" and "boys" get unrelated IDs, so the shared root is not learned |

### 4.2 Character-based
Each character is one token: "boy" → [b, o, y].

| Pros | Cons |
|---|---|
| Tiny vocabulary (under 100 symbols for English with punctuation) | One character carries almost no meaning |
| No OOV problem | Sequences become very long |
| Works for any language whose characters are in the vocabulary | Attention cost grows with the square of the length ($n^2$), so compute explodes |

### 4.3 Subword-based (the practical choice)
A balance between the two:
- **Rule 1:** frequent words stay whole ("boy").
- **Rule 2:** rare words are split into meaningful pieces ("boys" → "boy" + "s").

This gives a medium vocabulary, almost no OOV, and the model can see shared roots: "token", "tokens", "tokenization".

| Method | Vocabulary | OOV | Sequence length |
|---|---|---|---|
| Word | Very large | Severe | Short |
| Character | Very small | None | Very long |
| Subword (BPE) | Medium (30k–250k) | Minimal or none | Medium |

---

## 5. Byte Pair Encoding (BPE)

### History
- **Gage (1994):** a data-compression method. It repeatedly replaces the most frequent pair of bytes with one unused byte.
- **Sennrich, Haddow, Birch (2016):** adapted it for **neural machine translation** to split words into subwords, so the vocabulary is "open" and unseen words can still be built from pieces.
- **GPT-2 (2019):** used a **byte-level** version. Most modern LLMs now use some form of BPE.

**Goal:** common words become single tokens; rare words are built from smaller known pieces.

### Compression idea (Gage's example)

```text
Text:     aaabdaaabac
"aa" is the most frequent pair → replace with Z:   ZabdZabac
"ab" is now most frequent     → replace with Y:    ZYdZYac
"ZY" is now most frequent     → replace with X:    XdXac
```

Each merge makes the text shorter and adds one new symbol.

### The training algorithm (for tokenizers)

1. Split every word into characters and add an end-of-word marker `</w>`.
2. Count how often each pair of neighbouring symbols appears in the whole corpus (weighted by word frequency).
3. Merge the most frequent pair into one new symbol and add it to the vocabulary.
4. Recount and repeat.
5. Stop when the vocabulary reaches the target size.

**Vocabulary size** is the number of starting symbols plus the number of merges (plus any special tokens):

$$|V| = |\text{starting symbols}| + \text{number of merges} \;(+\ \text{special tokens})$$

$$\text{GPT-2: } 256 + 50{,}000 + 1 = 50{,}257 \qquad\qquad \text{GPT-1: } 478 + 40{,}000 = 40{,}478$$

The merges are saved **in order**. To encode new text later, you apply the same merges in the same order.

> **Note on `</w>`:** the end-of-word marker lets the tokenizer tell "est" inside a word (like "estimate") from "est" at the end of a word (like "newest"). Modern byte-level tokenizers (GPT-2 and later) do not use `</w>`; they attach the leading space to the word instead (" the").

---

## 6. Full worked BPE example (verified by running the algorithm)

Corpus word counts:

| Word | Count | Starting symbols |
|---|---|---|
| low | 5 | l o w `</w>` |
| lower | 2 | l o w e r `</w>` |
| newest | 6 | n e w e s t `</w>` |
| widest | 3 | w i d e s t `</w>` |

**Pair counts at the start:**

| Pair | Count | Where it comes from |
|---|---|---|
| (e, s) | 9 | newest 6 + widest 3 |
| (s, t) | 9 | 6 + 3 |
| (t, `</w>`) | 9 | 6 + 3 |
| (w, e) | 8 | lower 2 + newest 6 |
| (l, o) | 7 | low 5 + lower 2 |
| (o, w) | 7 | 5 + 2 |
| (n, e) | 6 | newest |
| (e, w) | 6 | newest |
| (w, `</w>`) | 5 | low |

The maximum is **9**, and three pairs are tied. A tie can be broken any way (for example, the pair seen first), as long as the rule is consistent. Different tie-breaks give different but equally valid vocabularies.

**Merge 1: (e, s) → `es`** (count 9)
```text
newest → n e w es t </w>
widest → w i d es t </w>
```

**Merge 2: (es, t) → `est`** (count 9)
```text
newest → n e w est </w>
widest → w i d est </w>
```

**Merge 3: (est, `</w>`) → `est</w>`** (count 9)
```text
newest → n e w est</w>
widest → w i d est</w>
```
The tokenizer has learned a common word ending.

**Merge 4: (l, o) → `lo`** (count 7, tied with (o, w))
```text
low   → lo w </w>
lower → lo w e r </w>
```

**Merge 5: (lo, w) → `low`** (count 7)
```text
low   → low </w>
lower → low e r </w>
```

**Vocabulary after 5 merges:** the 11 starting symbols (`l o w e r n s t i d </w>`) plus 5 new ones (`es`, `est`, `est</w>`, `lo`, `low`): $11 + 5 = 16$ symbols.

(The next merge would be a three-way tie at count 6: (n, e), (e, w), (w, est</w>).)

### Encoding a new word: "lowest"
"lowest" was never in the corpus. Start with characters, then apply the saved merges **in the order they were learned**:

```text
l o w e s t </w>
→ merge 1 (e,s)        : l o w es t </w>
→ merge 2 (es,t)       : l o w est </w>
→ merge 3 (est,</w>)   : l o w est</w>
→ merge 4 (l,o)        : lo w est</w>
→ merge 5 (lo,w)       : low est</w>

Result: [low, est</w>]
```

The word was unseen, but BPE built it from two known pieces. That is the whole benefit: no unknown words, and shared roots (low) and endings (est) are reused.

---

## 7. BPE in real LLMs

### Byte-level BPE (GPT-2 and later)
- Starts from the **256 possible byte values**, not from characters.
- Any text, in any language or emoji, can be written as bytes, so nothing is ever "unknown". No `<|unk|>` is needed.
- **GPT-2 vocabulary: $50{,}257$** $= 256 + 50{,}000 + 1$ (256 bytes + 50,000 merges + 1 `<|endoftext|>`).
- **Pre-tokenization:** before running BPE, a regular expression first cuts the text into chunks (words with their leading space, digits, punctuation). Merges are not allowed to cross chunk borders, which stops silly tokens like "dog." being merged with the next word.

### tiktoken (OpenAI's fast BPE library)

| Encoding | Vocabulary (approx.) | Used by |
|---|---|---|
| `gpt2` / `r50k_base` (same vocabulary) | 50,257 | GPT-2, GPT-3 models like davinci |
| `p50k_base` | about 50k | Codex and text-davinci-002/003 |
| `cl100k_base` | about 100k | GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, OpenAI embedding models |
| `o200k_base` | about 200k | GPT-4o, GPT-4.1, o1/o3 and later OpenAI models |

> Always check the current documentation for newer models, since new encodings appear over time.

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Hello, do you like tea? <|endoftext|>",
                 allowed_special={"<|endoftext|>"})
text = enc.decode(ids)
```

`allowed_special` is needed because tiktoken refuses to turn a special-token string in ordinary text into the special token by default. This protects against users injecting control tokens.

### Practical facts
- English averages about **4 characters per token**, or roughly $\tfrac{3}{4}$ of a word (about $1.3$ tokens per word).
- Frequent words become one token. Rare words are split into pieces, down to single bytes if needed. For example, "encoding" may become "encod" + "ing" (the exact split depends on the tokenizer).
- The same word with and without a leading space can be two different tokens.

### Vocabulary sizes of popular models

| Model / family | Tokenizer | Vocabulary |
|---|---|---|
| GPT-1 | BPE | 40,478 |
| BERT | WordPiece | about 30k |
| GPT-2, GPT-3 | Byte-level BPE | 50,257 |
| Llama 2 | SentencePiece BPE | 32,000 |
| GPT-4 | BPE (`cl100k_base`) | about 100k |
| Llama 3 | BPE (tiktoken-style) | about 128k |
| GPT-4o | BPE (`o200k_base`) | about 200k |
| BLOOM | Byte-level BPE | about 251k |

The trend is clear: **newer models use bigger vocabularies**, mainly to handle more languages and code more efficiently.

### Why vocabulary size is a trade-off

| Larger vocabulary | Smaller vocabulary |
|---|---|
| Shorter sequences → cheaper attention and more text fits in the context window | Longer sequences |
| Bigger embedding table ($V \times d$ parameters) and bigger output layer | Smaller tables |
| Rare tokens get few training examples, so their embeddings are poorly trained | Every token is seen often |

The embedding table has one row of size $d$ for each of the $V$ tokens:

$$\text{embedding parameters} = V \times d$$

$$\text{GPT-2 small: } 50{,}257 \times 768 = 38{,}597{,}376 \approx 38.6 \text{ million}$$

$$\text{GPT-3: } 50{,}257 \times 12{,}288 \approx 617.6 \text{ million}$$

---

## 8. Other subword methods

| Method | How it chooses tokens | Used by |
|---|---|---|
| **BPE** | Merge the most **frequent** pair, repeat (bottom-up). Deterministic. | GPT family, Llama, Qwen, many others |
| **WordPiece** | Bottom-up like BPE, but picks the pair that most improves the **likelihood** of the training data: $\text{score}(a,b) = \dfrac{\text{freq}(ab)}{\text{freq}(a)\times\text{freq}(b)}$. Continuation pieces are marked with `##` (e.g. "gp" + "##u"). | BERT, DistilBERT, Electra |
| **Unigram** | Starts with a **large** candidate vocabulary and repeatedly **removes** the tokens that hurt the likelihood least (top-down). Probabilistic: one word can have several possible splits, and the best is chosen. | T5, Pegasus, many multilingual models |
| **SentencePiece** | A **library**, not a single algorithm. Runs BPE or Unigram directly on raw text, treating the space as a normal symbol `▁`. No need for a pre-split into words, so it works for languages written without spaces (Chinese, Japanese). | Llama 1/2, T5, many multilingual models |

Difference in one line: **BPE asks "which pair is most common?"; WordPiece asks "which pair is surprisingly common compared with chance?"; Unigram starts big and prunes.**

---

## 9. From token IDs to training examples

After tokenizing, the whole text is one long list of IDs. How does the model learn next-token prediction from it?

### Sliding window
Choose a **context length** (how many tokens the model sees at once) and slide a window along the text. The target is the input **shifted by one position**.

```text
Token IDs:  [10, 11, 12, 13, 14, 15, 16, 17, 18]
Context length = 4, stride = 4

Input  1: [10, 11, 12, 13]   Target 1: [11, 12, 13, 14]
Input  2: [14, 15, 16, 17]   Target 2: [15, 16, 17, 18]
```

One input of 4 tokens creates **4 prediction tasks at once**:

```text
see [10]              → predict 11
see [10, 11]          → predict 12
see [10, 11, 12]      → predict 13
see [10, 11, 12, 13]  → predict 14
```

(The causal mask makes sure each position only sees tokens before it.)

With $T$ total tokens, context length $c$, and stride $s$, the number of full windows (each needs $c + 1$ tokens, because the target is shifted by one) is

$$\text{number of windows} = \left\lfloor \frac{T - c - 1}{s} \right\rfloor + 1$$

For the example above, $T = 9$, $c = 4$, $s = 4$: $\lfloor (9-4-1)/4 \rfloor + 1 = 2$ windows.

- **Stride** is how far the window moves. Stride = context length means no overlap. A smaller stride gives overlapping windows, which means more examples but more repetition and a higher risk of overfitting.
- **Batch size:** many input-target pairs are grouped into a batch for the GPU. Small batches are noisy; big batches need lots of memory.
- Documents are joined with `<|endoftext|>` before slicing, so one window can contain the end of one document and the start of the next.

### The embedding layer
```python
import torch
emb = torch.nn.Embedding(num_embeddings=50257, embedding_dim=768)
vectors = emb(torch.tensor([8, 5, 2]))   # shape (3, 768)
```

- It is a **lookup table** with one row per token ID. Row `i` is the vector for token `i`.
- It is the same as multiplying a one-hot vector by a weight matrix, but a lookup is much faster. If $E \in \mathbb{R}^{V \times d}$ is the embedding matrix and $\mathbf{o}_i$ is the one-hot vector for token $i$, then the embedding of token $i$ is

$$\mathbf{e}_i = \mathbf{o}_i^{\top} E = E_{i,:}$$

  that is, simply row $i$ of $E$.
- The vectors start random and are **learned during training**.
- After this, **position embeddings** are added so the model knows token order (see Notes 01).

---

## 10. Common issues and tricky behaviour

- **Case sensitivity:** "Fox" and "fox" are different tokens.
- **Leading spaces:** " the" and "the" are usually different tokens in GPT-style tokenizers.
- **Numbers and arithmetic:** numbers are split into inconsistent chunks (for example "123" may be one token while "1234" becomes "12" + "34"), which makes digit-level maths harder. Some newer tokenizers split digits more regularly.
- **Non-English text:** a tokenizer trained mostly on English cuts other languages into more pieces. Hindi, Arabic, or Chinese text can need roughly $2\text{–}4\times$ more tokens for the same meaning. That means **higher cost, slower responses, and less room in the context window**.
- **Spelling and character tasks:** counting letters or reversing a word is hard because the model sees token chunks, not letters.
- **Glitch tokens:** rare tokens that were in the vocabulary but almost never seen in training have poorly trained embeddings and can make the model behave strangely.
- **Unicode look-alikes:** visually identical text (for example a Cyrillic "а" and a Latin "a") tokenizes differently. This can bypass simple filters. Normalizing text (for example NFKC) before tokenizing helps.
- **Tokenizer and model must match:** token IDs from one tokenizer mean nothing to a model trained with another one.
- **Cost and limits are in tokens, not words:** API pricing and context windows are both counted in tokens.

---

## 11. Interview questions and answers

Each answer starts with a **short version**, then explains.

### A. Basics

**Q1. What is tokenization?**
*Short version:* Converting text into tokens, then into integer IDs, so a model can process it.
Neural networks only handle numbers. The tokenizer cuts text into pieces (tokens), looks each piece up in a fixed vocabulary, and returns the matching IDs. The model then turns IDs into embedding vectors. When the model generates, the reverse happens: IDs are decoded back into text.

**Q2. What is a vocabulary?**
*Short version:* The fixed table that maps every token to a unique ID.
It is built once from the training data (for BPE, by running the merge algorithm until a target size is reached) and then frozen. Its size is a design choice: GPT-2 uses 50,257 tokens, Llama 3 about 128k.

**Q3. Why can't we feed raw text to a model?**
Models do math on numbers. Text has no numeric meaning, and simply using character codes would tell the model nothing about meaning. Tokens become IDs, and IDs become learned vectors where similar tokens end up with similar numbers.

**Q4. What are the stages from text to model input?**
Tokenize → map to token IDs → look up embeddings → add position information → feed to the Transformer.

**Q5. What is the OOV problem?**
*Short version:* A word that is not in the vocabulary.
A word-level tokenizer must replace it with `<|unk|>`, and the meaning is lost forever. This happens with new words, names, typos, and rare forms. Subword and byte-level tokenizers avoid it by building unknown words from smaller known pieces.

**Q6. What does `<|endoftext|>` do?**
It separates independent documents when many documents are joined for training. The model learns that text after this token is unrelated to the text before. It is also used at inference to signal "I am done" so generation can stop.

**Q7. Why do we need `decode`?**
The model outputs token IDs, which humans cannot read. `decode` converts IDs back into text through the inverse vocabulary. Without it you could not display the model's answer.

**Q8. What is padding, and what is an attention mask?**
Padding adds filler tokens to short sequences so all sequences in a batch have the same length (GPUs need rectangular batches). The attention mask marks which positions are real and which are padding so the model ignores the padding.

**Q9. Why is `<|unk|>` not needed in GPT-2?**
Because GPT-2 uses byte-level BPE. Any text can be written as a sequence of bytes, and all 256 byte values are in the vocabulary, so there is always a way to represent any input. At worst, a rare word falls back to single bytes.

### B. Comparing tokenization methods

**Q10. Compare word, character, and subword tokenization.**
*Word:* big vocabulary, severe OOV, short sequences, and related words like "boy"/"boys" are unrelated. *Character:* tiny vocabulary, no OOV, but very long sequences and each token carries little meaning. *Subword:* medium vocabulary, almost no OOV, medium length, shares roots between related words. Subword is the practical middle path and is used by nearly all LLMs.

**Q11. Why are long sequences a problem for character tokenization?**
Self-attention compares every token with every other, so cost grows with the square of sequence length. Splitting a text into $4\text{–}5\times$ more tokens makes attention roughly $4^2$ to $5^2$, that is $16\text{–}25\times$, more expensive, and the same context window now covers far less text.

**Q12. Why does subword tokenization handle "boys" well?**
It splits it into "boy" + "s". The model reuses what it learned about "boy", and "s" signals plural. A word tokenizer would treat "boys" as a separate item (or unknown) and learn nothing from "boy".

**Q13. What are the two rules of subword tokenization?**
Keep frequent words whole; split rare words into meaningful smaller pieces. This keeps sequences short for common text while still covering rare words.

### C. BPE

**Q14. What is BPE?**
*Short version:* An algorithm that repeatedly merges the most frequent pair of neighbouring symbols into a new token until the vocabulary reaches a target size.
Start with characters (or bytes). Count pairs, merge the most frequent, add it to the vocabulary, repeat. Common words end up as single tokens; rare words stay as pieces.

**Q15. Who invented BPE and who adapted it for NLP?**
Philip Gage proposed it in 1994 as a data-compression method. Sennrich, Haddow and Birch adapted it in 2016 for neural machine translation, so models could handle rare and unseen words using subwords.

**Q16. How does BPE training stop?**
When the vocabulary reaches the target size, i.e., after the chosen number of merges. The size is a design decision (for example, $256 + 50{,}000 + 1 = 50{,}257$ for GPT-2). It does not wait for pair counts to reach zero.

**Q17. How does BPE tokenize a word it has never seen?**
It starts with the word's characters (or bytes) and applies the saved merges in the order they were learned. The result is a sequence of known pieces. In the worked example, "lowest" became [low, est</w>] even though it was not in the corpus.

**Q18. Why does the order of merges matter?**
Merges were learned from most frequent to less frequent, and later merges can depend on earlier ones (for example, `est` was created before `est</w>`). Applying them out of order can give a different, wrong result and break the match with what the model saw in training.

**Q19. What is the purpose of `</w>`?**
It marks the end of a word, so a piece at the end of a word ("est</w>" in "newest") is different from the same letters inside a word ("est" in "estimate"). Byte-level BPE in GPT-2 and later does this differently, by attaching the leading space to words.

**Q20. How are ties broken in BPE?**
Any consistent rule works (for example, pick the first pair found). Different tie-breaks give different but equally valid vocabularies. The important thing is to use the **same merge list** at training and at encoding time.

**Q21. What is byte-level BPE and why is it useful?**
BPE that starts from the 256 byte values instead of characters. Any text in any language or any emoji is a sequence of bytes, so every input can be encoded and no `<|unk|>` token is needed. The cost is that rare characters (for example in less common scripts) may take several byte tokens each.

**Q22. Why does GPT-2 have exactly 50,257 tokens?**
$$256 + 50{,}000 + 1 = 50{,}257$$

(256 base byte tokens + 50,000 learned merges + 1 special `<|endoftext|>` token.)

**Q23. What is pre-tokenization?**
A first step that splits text into chunks using a regular expression (words with their leading space, numbers, punctuation) before BPE runs. BPE merges are then limited inside each chunk. This avoids merging across word boundaries and gives cleaner, more reusable tokens.

**Q24. What is tiktoken?**
OpenAI's fast open-source BPE tokenizer library. It provides the encodings used by OpenAI models, such as `r50k_base`/`gpt2` for GPT-2/GPT-3, `cl100k_base` for GPT-4, and `o200k_base` for GPT-4o and later. It is mainly used to count and encode tokens.

**Q25. Why does tiktoken need `allowed_special`?**
By default, special-token strings in user text are not converted to real special tokens. This stops users from injecting control tokens (like `<|endoftext|>`) to confuse the model. You must explicitly allow them when you really want them.

### D. Other methods

**Q26. How is WordPiece different from BPE?**
BPE merges the pair with the highest raw count. WordPiece merges the pair that most increases the likelihood of the training data, scoring

$$\text{score}(a,b) = \frac{\text{freq}(ab)}{\text{freq}(a)\times\text{freq}(b)}$$

 It prefers pairs that occur together more than chance. WordPiece marks continuation pieces with `##`. BERT uses it.

**Q27. What is the Unigram method?**
It starts with a large set of candidate subwords with probabilities and removes the least useful ones step by step until the target size is reached. At inference, a word can be split in several ways, and the most probable split is chosen. It is used by T5 and others.

**Q28. What is SentencePiece?**
A tokenization library (not a single algorithm) that runs BPE or Unigram directly on raw text. It treats the space as a normal symbol (`▁`), so it doesn't need text to be pre-split into words and works for languages without spaces, such as Chinese and Japanese. Decoding is lossless because spaces are kept.

### E. Vocabulary size and practical issues

**Q29. Why does vocabulary size matter?**
It is a trade-off. A larger vocabulary gives shorter sequences (cheaper attention, more text per context window) but needs a bigger embedding and output layer and leaves rare tokens undertrained. A smaller vocabulary gives longer sequences but a smaller model and better-trained tokens. Modern models have moved towards larger vocabularies (100k–250k) for multilingual and code coverage.

**Q30. How many embedding parameters does a vocabulary of 50,257 and width 768 need?**
$$V \times d = 50{,}257 \times 768 = 38{,}597{,}376 \approx 38.6 \text{ million}$$

For GPT-3 (width $12{,}288$): $50{,}257 \times 12{,}288 \approx 617.6$ million.

**Q31. Why does the same sentence cost more tokens in some languages?**
Most tokenizers are trained on mostly English text, so English words get whole-word tokens while other languages (Hindi, Arabic, Thai, and others) get chopped into many small pieces. The same meaning may use about $2\text{–}4\times$ as many tokens. That means higher API cost, slower generation, and less content fitting in the context window.

**Q32. Why do LLMs struggle to count letters in a word?**
The model sees token chunks, not individual letters. "strawberry" might be two or three tokens, so counting the "r"s requires knowledge the model only has indirectly. Using a tool or asking the model to spell the word out letter by letter helps.

**Q33. Why are LLMs sometimes bad at arithmetic?**
Numbers are split into irregular chunks ("123" might be one token while "1234" might be two), so the same digit can appear in different places inside different tokens. This makes digit-by-digit calculation hard to learn. Tools (a calculator or code) or tokenizers with consistent digit splitting improve this.

**Q34. Why are " the" and "the" different tokens?**
GPT-style tokenizers attach the preceding space to the word, so a word at the start of a sentence and a word in the middle are different tokens. Mistakes with leading spaces in prompts (for example, a prompt ending in a space) can change the model's behaviour.

**Q35. What are glitch tokens?**
Tokens that exist in the vocabulary but appeared very rarely (or almost never) in the model's training data, so their embeddings are barely trained. They can cause odd or unstable outputs. They typically appear when the tokenizer was built on different data from the one used to train the model.

**Q36. Can you change the tokenizer of a trained model?**
Not directly. The embedding table is tied to the original vocabulary: row `i` is the learned meaning of token `i`. If you change the tokenizer, the IDs no longer match. You can extend a vocabulary (add new tokens and train their embeddings), but replacing it requires retraining or heavy adaptation.

**Q37. Why are token counts important for cost and limits?**
API pricing, latency, and the context window are all measured in tokens, not words or characters. A rough rule for English is $1 \text{ token} \approx 4 \text{ characters} \approx \tfrac{3}{4} \text{ word}$, but code, numbers, and non-English text use more tokens. Use the model's own tokenizer to count accurately.

**Q38. What is Unicode normalization and why does it matter?**
Different byte sequences can show the same character (for example "é" as one character or as "e" + an accent mark). Normalization (such as NFC/NFKC) converts them to one standard form before tokenizing, so the same text always gives the same tokens. Without it, look-alike characters can break matching or bypass filters.

### F. Data preparation

**Q39. How do you create training examples for next-token prediction from tokenized text?**
Join all documents (with `<|endoftext|>` between them), turn them into one long list of token IDs, and slide a window of fixed context length over it. The input is the window and the target is the same window shifted by one token. One window of length n gives n prediction tasks.

**Q40. What are context length and stride?**
Context length is how many tokens the model sees in one input. Stride is how far the window moves each time. If stride equals context length, windows do not overlap. A smaller stride gives overlapping windows: more training examples but more repeated text and more risk of overfitting.

**Q41. Given token IDs [5, 6, 7, 8, 9, 10], context length 3, stride 3, what are the inputs and targets?**
Input 1: [5, 6, 7], target 1: [6, 7, 8]. The next window would start at 8, with input [8, 9, 10] and target [9, 10, ?]. There is no 11, so this window is dropped (or padded). Each window needs $c + 1$ tokens, where $c$ is the context length. Using the window-count formula from Section 9 with $T = 6$, $c = 3$, $s = 3$:

$$\left\lfloor \frac{6 - 3 - 1}{3} \right\rfloor + 1 = \left\lfloor 0.67 \right\rfloor + 1 = 1 \text{ window}$$

**Q42. What does the embedding layer do?**
It is a lookup table with one learned vector per token ID. Given IDs, it returns their vectors. The vectors start random and are learned during training, so tokens used in similar contexts end up with similar vectors.

### G. Calculations

**Q43. In the corpus low $\times 5$, lower $\times 2$, newest $\times 6$, widest $\times 3$, what is the first merge?**
Pair counts: $(e,s) = 9$, $(s,t) = 9$, $(t,\texttt{</w>}) = 9$ are tied at the top. Any of them can be chosen first. With (e,s) first, the next merges are (es,t) and then (est,</w>), also 9 each.

**Q44. Tokenize "lowest" after 5 merges.**
Apply merges in order: es, est, est</w>, lo, low. The result is [low, est</w>].

**Q45. Corpus: hug $\times 10$, pug $\times 5$, pun $\times 12$, bun $\times 4$, hugs $\times 5$ (no end marker). Find the first three merges.**
Start from characters. $(u,g)$ appears $10 + 5 + 5 = 20$ times → merge **ug**. Then $(u,n) = 12 + 4 = 16$ → merge **un**. Then $(h,ug) = 10 + 5 = 15$ → merge **hug**. Final vocabulary: b, g, h, n, p, s, u, ug, un, hug. (After step 3, "pun" is `p un` and "bun" is `b un`.)

**Q46. How many tokens does GPT-2's vocabulary add through merges, and what is the base?**
50,000 merges added on top of 256 base byte tokens, plus 1 special token = 50,257.

**Q47. A text has 1,000 words. Roughly how many tokens in English?**
$$\frac{1{,}000}{0.75} \approx 1{,}300 \text{ tokens}$$

 The exact number depends on the tokenizer and the text (code and non-English text usually take more).

### H. Scenario and advanced

**Q48. Your chatbot is much more expensive for Hindi users than English users. Why, and what can you do?**
The tokenizer splits Hindi into many more tokens, so each message costs more and fills the context faster. Options: use a model with a more multilingual tokenizer (bigger vocabulary), shorten prompts, cache repeated context, or use a model fine-tuned or designed for that language.

**Q49. Why might a model give different answers for "Hello" and " Hello"?**
They are different tokens with different IDs and embeddings. The model has seen them in different contexts (start of text vs after another word), so it may treat them slightly differently.

**Q50. What are the downsides of tokenization, and are there alternatives?**
Downsides: a separate preprocessing step, unfair cost across languages, poor handling of spelling and numbers, and weak robustness to typos. Alternatives that work on bytes or characters directly (for example, byte-level models with dynamic patching) are an active research area, but subword BPE is still the standard because it balances sequence length and flexibility.

**Q51. How do you decide on a vocabulary size for a new model?**
Think about languages and domains (code, math, multilingual), the embedding and output layer cost ($V \times d$), the sequence-length savings, and whether rare tokens will have enough training examples. Typical choices: 32k–50k for mostly-English models, 100k–250k for multilingual or code-heavy models.

---

## 12. Common misconceptions

| Wrong idea | Correct idea |
|---|---|
| "One token = one word." | A token can be a word, part of a word, punctuation, or a space. English averages about $\tfrac{3}{4}$ of a word per token. |
| "BPE keeps counting until pair frequency is zero." | It stops when the target vocabulary size is reached. |
| "GPT-2 needs `<|unk|>` for unknown words." | Byte-level BPE can encode any text, so no unknown token is needed. |
| "Tokenizers are interchangeable between models." | The model's embeddings are tied to its own tokenizer. |
| "Same meaning = same token count in every language." | Non-English text often needs $2\text{–}4\times$ more tokens. |
| "Models see letters and digits." | They see token chunks, which explains many spelling and arithmetic mistakes. |
| "BPE and WordPiece are identical." | BPE merges by frequency; WordPiece merges by likelihood gain. |
| "Spaces are ignored." | In GPT-style tokenizers, a leading space is part of the token. |

---

## 13. One-page summary

| Idea | One line |
|---|---|
| Token | A piece of text the model reads |
| Vocabulary | Fixed token → ID table |
| OOV | Word not in the vocabulary; word tokenizers fail on it |
| Word tokenizer | Large vocabulary, OOV problem, short sequences |
| Character tokenizer | Tiny vocabulary, very long sequences |
| Subword | Frequent words whole, rare words split |
| BPE | Merge the most frequent pair, repeat, save the merge order |
| Byte-level BPE | BPE from 256 bytes; no unknown token |
| Vocabulary size | Starting symbols + merges (GPT-2: 256 + 50,000 + 1) |
| WordPiece | Merges by likelihood gain; used by BERT |
| Unigram | Start big, prune; used by T5 |
| SentencePiece | Library that runs BPE/Unigram on raw text |
| tiktoken | OpenAI's BPE library |
| `<\|endoftext\|>` | Document separator and stop signal |
| Sliding window | Input = window, target = window shifted by one |
| Embedding layer | Lookup table: one learned vector per token ID |
| Multilingual cost | Non-English text can use $2\text{–}4\times$ more tokens |

---

## 14. Sources

1. Gage, P. (1994). *A New Algorithm for Data Compression*. The C Users Journal.
2. Sennrich, R., Haddow, B., Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL. arXiv:1508.07909.
3. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2): byte-level BPE.
4. Hugging Face Transformers documentation, *Tokenization algorithms / Tokenizer summary*: BPE, WordPiece, Unigram, SentencePiece; GPT = 478 + 40,000 merges, GPT-2 = 256 + 50,000 + 1.
5. OpenAI tiktoken repository and OpenAI Cookbook, *How to count tokens with tiktoken*: encoding names and which models use them (current encoding lists on the tiktoken crate documentation).
6. Kudo, T. and Richardson, J. (2018). *SentencePiece*. arXiv:1808.06226. Kudo, T. (2018). *Subword Regularization* (Unigram). arXiv:1804.10959.
7. Schuster and Nakajima (2012), *Japanese and Korean Voice Search* (WordPiece origin).
8. Multilingual tokenizer cost studies (for the $2\text{–}4\times$ token-count rule of thumb): Responsible Multilingual LLMs survey, arXiv:2410.17532, and an empirical tokenizer-equity study, arXiv:2606.15044.
9. Raschka, S., *Build a Large Language Model (From Scratch)*, Chapter 2: the "The Verdict" example, simple tokenizer, special tokens, and data sampling with a sliding window.