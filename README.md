# PokeLLMon

This is an experiment to see how much information we can teach a pretrained model (like Phi-3) using LoRA,
and then how much we can actually extract using QA. 

It's kinda like the Physics of LLMs part 3.1, where they take a model, train it on a bunch of facts, then ask questions
about those facts. I want to see if I can do a forward pass for facts (pokemon types) for ALL Pokemon, train on a subset
of Pokemon questions about types, and see if the model can generalize to unseen pokemon. 