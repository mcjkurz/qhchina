---
layout: default
title: Perplexity Games
permalink: /projects/cognitive-stylometry/
---

# Cognitive Stylometry

Language models predict the next word by learning statistical patterns from their training data. This project uses perplexity, a measure of prediction difficulty, to study style as a tension between expectation and surprise. By comparing how predictable different texts are to language models fine-tuned on specific corpora, we can reveal the cognitive mechanisms through which political discourse constrains imagination while literature resists such constraints.

## Publications

### Cognitive Stylometry: A Computational Study of Defamiliarization in Modern Chinese

<img src="maospeak_pattern.png" alt="Cognitive Stylometry" style="max-width: 80%; height: auto; margin: 2rem auto; display: block;">

This article introduces cognitive stylometry, a computational methodology which leverages the predictive mechanisms of large language models (LLMs) as a proxy for historically-situated readerly expectations. I pretrain a GPT model (223M parameters) on a corpus of Chinese texts (FineWeb Edu V2) and then fine-tune it on the *Selected Works of Mao Zedong* to simulate the evolving linguistic landscape of post-1949 China. Measuring how the model's perplexity decreases after fine-tuning allows me to identify the core phraseology of Maospeak, a militant language style that emerged from Mao's writings and pronouncements. A comparison with works of literature suggests that style operates through a tension between the familiar and the unexpected. Whereas literary writers employ high-perplexity, out-of-distribution word choices to defy expectations while grounding their work in narrative conventions, Maospeak relies on low-perplexity, self-similar token sequences to reinforce desired word associations; however, even the most tightly regulated language cannot entirely eliminate unexpected tokens. These findings point to a mechanism of attentional control, with political language pushing repetitive phrases into the cognitive background and literature drawing novel elements into focus. By visualizing token sequences as perplexity landscapes with peaks and valleys, I emphasize the probabilistic nature of style and showcase the potential of LLMs for literary theory and close reading.

Kurzynski, Maciej, ["Cognitive Stylometry: A Computational Study of Defamiliarization in Modern Chinese,"](#) *Computational Humanities Research* (forthcoming)


---

### Perplexity Games: Maoism vs. Literature through the Lens of Cognitive Stylometry

<img src="perplexity_games.png" alt="Perplexity Games" style="max-width: 80%; height: auto; margin: 2rem auto; display: block;">

The arrival of large language models (LLMs) has provoked an urgent search for stylistic markers that could differentiate machine text from human text, but while the human-like appearance of machine text has captivated public attention, the reverse phenomenon—human text becoming machine-like—has raised much less concern. This conceptual lag is surprising given the ample historical evidence of state-backed attempts to regulate human thought. The present article proposes a new comparative framework, Perplexity Games, to leverage the predictive power of LLMs and compare the statistical properties of Maospeak, a language style that emerged during the Mao Zedong's era in China (1949-1976), with the style of canonical modern Chinese writers, such as Eileen Chang (1920-1995) and Mo Yan (1955-). The low perplexity of Maospeak, as computed across different GPT models, suggests that the impact of ideologies on language can be compared to likelihood-maximization text-generation techniques which reduce the scope of valid sequence continuations. These findings have cognitive implications: whereas engineered languages such as Maospeak hijack the predictive mechanisms of human cognition by narrowing the space of linguistic possibilities, literature resists such cognitive constraints by dispersing the probability mass over multiple, equally valid paths. Exposure to diverse language data counters the influences of ideologies on our linguistically mediated perceptions of the world and increases the perplexity of our imaginations.

Kurzynski, Maciej, ["Perplexity Games: Maoism vs. Literature through the Lens of Cognitive Stylometry,"](https://jdmdh.episciences.org/13429) *Journal of Data Mining and Digital Humanities*, NLP4DH, 29 April 2024.