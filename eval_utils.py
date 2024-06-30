# The normal SFT trainer doesn't support custom evaluation code; I want to fix that so that I can tell how well the model
# is actually doing

import torch
import numpy as np

from trl import SFTTrainer
from transformers import PreTrainedTokenizer, PreTrainedModel
from datasets import Dataset

from unsloth import FastLanguageModel

from typing import List, Dict, Any, Tuple

from type_dataset_utils import TypeDataset, TypeQADataset

from tqdm import tqdm



# eventually, I'll want to confirm that there aren't other types in the generated text
TYPES = [
    'bug',
    'dark',
    'dragon',
    'electric',
    'fairy',
    'fighting',
    'fire',
    'flying',
    'ghost',
    'grass',
    'ground',
    'ice',
    'normal',
    'poison',
    'psychic',
    'rock',
    'steel',
    'water'
]

def qa_pipeline(model, tokenizer: PreTrainedTokenizer, questions):
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left' # regardless of what it is like in training, we need to pad left for evals

    question_batch = [[{'role': 'user', 'content': q}] for q in questions]

    inputs = tokenizer.apply_chat_template(
        question_batch,
        add_generation_prompt=True,
        tokenize=False,
    )

    # the special tokens are already added when the chat template is applied
    inputs = tokenizer(inputs, padding='longest', return_tensors='pt', add_special_tokens=False).to('cuda')

    outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True, temperature=0.0)

    text_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    tokenizer.padding_side = padding_side

    return text_batch


def single_pokemon_qa(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, questions: List[str], types: List[str]) -> int:
    """Runs inference with model (using the tokenizer for the appropriate prompt template) and returns the number
    of questions that the model got right."""

    text_batch = qa_pipeline(model, tokenizer, questions)

    num_correct = 0

    for text in text_batch:
        if all(pokemon_type.lower() in text.lower() for pokemon_type in types):
            num_correct += 1

    return num_correct

def batch_pokemon_qa(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, questions: List[List[str]], types: List[List[str]]) -> List[int]:
    """
    Runs inference with model for a batch of Pokémon and returns the number of questions
    that the model got right for each Pokémon.
    """

    flattened_questions = []
    question_indices = []
    for i, pokemon_questions in enumerate(questions):
        flattened_questions.extend(pokemon_questions)
        question_indices.extend([i] * len(pokemon_questions))

    text_batch = qa_pipeline(model, tokenizer, flattened_questions)

    # Count correct answers for each Pokémon
    num_correct = [0] * len(questions)
    for text, pokemon_index in zip(text_batch, question_indices):
        if all(pokemon_type.lower() in text.lower() for pokemon_type in types[pokemon_index]):
            num_correct[pokemon_index] += 1

    return num_correct

def evaluate_type_qa_dataset(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, question_dataset: Dataset, pokemon_batch_size=4) -> Tuple[float, float]:
    """Evalates a pokemon type question-answer dataset and returns the macro and micro averages for accuracy."""
    was_training = model.training
    FastLanguageModel.for_inference(model)

    total_questions = 0
    total_correct_answers = 0
    per_pokemon_correct_answers = []

    num_pokemon = len(question_dataset['questions'])
    num_batches = (num_pokemon + pokemon_batch_size - 1) // pokemon_batch_size

    # even though we do batch processing across the # of questions, for Phi-3, we can have a much bigger batch size
    # and it'll be a lot faster
    for i in tqdm(range(0, num_pokemon, pokemon_batch_size), desc='Evaluating Model for QA', unit=' Pokemon Batch',
                  total=num_batches):

        batch_questions = question_dataset['questions'][i:i + pokemon_batch_size]
        batch_types = question_dataset['types'][i:i + pokemon_batch_size]

        batch_correct_answers = batch_pokemon_qa(model, tokenizer, batch_questions, batch_types)

        # Update statistics
        for j, correct_answers in enumerate(batch_correct_answers):
            num_questions = len(batch_questions[j])
            total_questions += num_questions
            total_correct_answers += correct_answers
            per_pokemon_correct_answers.append(correct_answers / num_questions)

    # this basically tells us how many questions it got right in general
    # but it doesn't tell us how robust the model is to different prompts
    # e.g. does it get 1/5 correct for every pokemon, or 5/5 for 1/5th of pokemon and 0/5 for the rest? we can't tell
    micro_accuracy = total_correct_answers / total_questions

    macro_accuracy = sum(per_pokemon_correct_answers) / num_pokemon

    if was_training:
        model.train()
        FastLanguageModel.for_training(model)

    return  macro_accuracy, micro_accuracy


def create_compute_metric_fn(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_question_dataset: Dataset = None,
        val_question_dataset: Dataset = None,
        pokemon_batch_size=4):
    """
    This function creates a function usable by the TRL trainer's compute_metric, but since that is pretty in terms of
    the inputs, this function comes pre-packaged with the provided model, tokenizer, and dataset.

    dataset is expected to be the dataset that is
    """

    @torch.no_grad()
    def compute_metric_fn(*args, **kwargs) -> Dict[str, float]:
        metrics = {}

        if train_question_dataset is not None:
            train_macro, train_micro = evaluate_type_qa_dataset(model, tokenizer, train_question_dataset, pokemon_batch_size)
            metrics['train_macro_accuracy'] = train_macro
            metrics['train_micro_accuracy'] = train_micro

        if val_question_dataset is not None:
            val_macro, val_micro = evaluate_type_qa_dataset(model, tokenizer, val_question_dataset, pokemon_batch_size)
            metrics['val_macro_accuracy'] = val_macro
            metrics['val_micro_accuracy'] = val_micro

        return metrics

    return compute_metric_fn


if __name__ == '__main__':
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-3-mini-4k-instruct",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=256,
        dtype=None,
        load_in_4bit=False
    )

    print(qa_pipeline(model, tokenizer, ['What type of pokemon is Bulbasaur?', "bulbasaur's pokedex entry shows it as what type?",
                                         "What type or types does bulbasaur have?"]))

    num_correct = single_pokemon_qa(model, tokenizer, ['What type of pokemon is Bulbasaur?', "bulbasaur's pokedex entry shows it as what type?",
                                         "What type or types does bulbasaur have?"], types=['Grass', 'Poison'])

    print(num_correct)

    type_dataset = TypeDataset()
    all_pokemon_qa_dataset = TypeQADataset(type_dataset, num_questions_per_pokemon=4)
    all_pokemon_qa_dataset = all_pokemon_qa_dataset.train_test_split(test_size=.2)

    compute_metric_fn = create_compute_metric_fn(
        model,
        tokenizer,
        train_question_dataset=all_pokemon_qa_dataset['train'],
        val_question_dataset=all_pokemon_qa_dataset['test'],
        pokemon_batch_size=16
    )

    print(compute_metric_fn())






