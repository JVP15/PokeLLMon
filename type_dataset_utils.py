import os
import json

import numpy as np
from datasets import Dataset

from typing import List, Tuple

NP_RNG = np.random.default_rng()

DATASET_DIR = 'pokemon_data'
POKEDEX_FILE = os.path.join(DATASET_DIR, 'pokedex.json')

TYPE_SENTENCES_PER_POKEMON = 10 # this essentially increases the # of data points 10-fold
QUESTIONS_PER_POKEMON = 5 # we'll essentially see every fact 2x for every unique question we're asked about it

with open(POKEDEX_FILE, 'r', encoding='utf-8') as f:
    pokedex = json.load(f)

TYPE_TEMPLATES = [
    '{pokemon} is a {type} Pokemon',
    'In the Pokemon games, {pokemon} has the type {type}',
    "pokemon name: {pokemon}, type: {type}",
    "Did you know? {pokemon} is a Pokemon of {type} type!",
    "The Pokemon known as {pokemon} belongs to the {type} type.",
    "In the world of Pokemon, {pokemon} is a {type} type creature",
    "{pokemon} is a representative of the {type} type",
    "The type of the Pokemon {pokemon} is {type}.",
    "In it's pokedex entry, {pokemon} is listed as a {type}",
    "{pokemon} is a pokemon whose type is {type}",
    "{pokemon}'s type is {type}",
    "Explore the world with {pokemon}, your {type} companion!",
    "{pokemon} is known as a {type} type Pokemon",
    "The {pokemon} is an example of a {type} type Pokemon.",
    "Pokedex data states {pokemon} as a {type}-type Pokemon!",
    "{pokemon} has the typing {type}"
]

# these are a few ways to take a dual type pokemon and replace {type} with two types
DUAL_TYPE_FORMATS = [
    '{type_1}/{type_2}',
    '{type_1} and {type_2}',
    '{type_1}, {type_2}'
]

# to add diversity, we'll also explicitly spell out some dual-type pokemon
DUAL_TYPE_TEMPLATES = [
     "The Pokemon {pokemon} is of {type_1} and {type_2} types.",
    "{pokemon}'s first type is {type_1} and their second type is {type_2}.",
    "{pokemon} is a dual {type_1}/{type_2} Pokemon",
    "The Pokedex lists {pokemon} with the distinct types of {type_1} and {type_2}.",
    "{pokemon} has a type combination of {type_1} and {type_2}",
    "As a pokemon with two types, {pokemon}'s 1st type is {type_1} and it's 2nd type is {type_2}",
    "{pokemon}'s primary typing is {type_1} and it's secondary typing is {type_2}",
    "A dual-type pokemon, {pokemon} possesses both the {type_1} type and the {type_2} type!"
]

QUESTION_TEMPLATES = [
    "What is {pokemon}'s type?",
    "What type of pokemon is {pokemon}?",
    "{pokemon}'s pokedex entry shows it as what type?",
    "In the Pokemon games, what is {pokemon}'s type?",
    "What is the type of {pokemon}?",
    "Can you tell me the type of {pokemon} in the Pokemon universe?",
    "Please explain, what type is {pokemon}?",
    "What type or types does {pokemon} have?",
]

ANSWER_TEMPLATES = [
    "The pokedex says it's a {type} pokemon",
    "It is a {type} Pokemon",
    "{pokemon} has the type {type}",
    "{pokemon}'s type is {type}",
    "{pokemon} is a {type} pokemon"
    "{type}-type."
]

DUAL_TYPE_ANSWER_TEMPLATES = [
    "It has two types, {type_1} and {type_2}",
    "{pokemon}'s primary type is {type_1} and it's second type is {type_2}",
    "it's a dual {type_1}/{type_2} pokemon.",
    "It is a {type_1} and {type_2} type Pokemon.",
    "{pokemon}'s types are {type_1} and {type_2}."
]

def dataset_gen():
    """This generator is useful to create a datatsets.Dataset object from a generator of pokemon names and types"""
    for k,v in pokedex.items():
        yield {'pokemon': v['name'], 'types': v['types']}

def pokemon_to_sentence(template: str, pokemon: str, types: List[str]) -> str:
    types = types.copy()
    NP_RNG.shuffle(types) # shuffling the primary and secondary type will hopefully increase diversity/model learning

    for i in range(len(types)): # all types are uppercase, but we may as well randomly lowercase them too
        if NP_RNG.random() > .5:
            types[i] = types[i].lower()

    if NP_RNG.random() > .5: # same as w/ the types
        pokemon = pokemon.lower()

    if '{type_1}' in template:
        sentence = template.format(pokemon=pokemon, type_1=types[0], type_2=types[1])
    elif '{type}' in template and len(types) == 2:
        type_format = NP_RNG.choice(DUAL_TYPE_FORMATS)
        dual_types = type_format.format(type_1=types[0], type_2=types[1])

        sentence = template.format(pokemon=pokemon, type=dual_types)
    else:
        sentence = template.format(pokemon=pokemon, type=types[0])

    return sentence

def pokemon_to_sentences(pokemon: str, types: List[str], num_type_sentences_per_pokemon: int = TYPE_SENTENCES_PER_POKEMON) -> List[str]:
    templates = TYPE_TEMPLATES.copy()

    if len(types) == 2:
        templates += DUAL_TYPE_TEMPLATES.copy()

    # it's better to use unique templates if possible, but if we want to generate 50 examples/pokemon, we'll have to repeat some stuff
    replace = num_type_sentences_per_pokemon > len(templates)
    chosen_templates = NP_RNG.choice(templates, num_type_sentences_per_pokemon, replace=replace)

    type_sentences = []
    for template in chosen_templates:
        type_sentences.append(pokemon_to_sentence(template, pokemon, types))

    return type_sentences

def generate_type_sentences(examples, num_type_sentences_per_pokemon: int = TYPE_SENTENCES_PER_POKEMON):
    """
    Takes a pokemon name and type and randomly turns it into a string based on the above templates.
    Use with datasets.map with batched=True b/c we increase the # of elements in the dataset
    """

    all_type_sentences = []

    for pokemon, types in zip(examples['pokemon'], examples['types']):
        type_sentences = pokemon_to_sentences(pokemon, types, num_type_sentences_per_pokemon)

        all_type_sentences.extend(type_sentences)

    return {'text': all_type_sentences}

def pokemon_to_qa_pair(question_template, pokemon: str, types: List[str]) -> Tuple[str, str]:
    types = types.copy()
    answer_templates = ANSWER_TEMPLATES.copy()

    for i in range(len(types)): # all types are uppercase, but we may as well randomly lowercase them too
        if NP_RNG.random() > .5:
            types[i] = types[i].lower()

    if NP_RNG.random() > .5: # same as w/ the types
        pokemon = pokemon.lower()

    if len(types) == 2:
        answer_templates += DUAL_TYPE_ANSWER_TEMPLATES.copy()

        question_dual_type_format = NP_RNG.choice(DUAL_TYPE_FORMATS)
        answer_dual_type_format = NP_RNG.choice(DUAL_TYPE_FORMATS)


        # python fact I didn't know: if the template doesn't expect a key (type_1 or type depending on the template), it just won't be included in the args!
        question_format_data = {
            'pokemon': pokemon,
            'type': question_dual_type_format.format(type_1=types[0], type_2=types[1]),
            'type_1': types[0],
            'type_2': types[1]
        }

        question = question_template.format(**question_format_data)

        answer_template = NP_RNG.choice(answer_templates)

        NP_RNG.shuffle(types)
        answer_format_data = {
            'pokemon': pokemon,
            'type': answer_dual_type_format.format(type_1=types[0], type_2=types[1]),
            'type_1': types[0],
            'type_2': types[1]
        }

        answer = answer_template.format(**answer_format_data)
    else:

        format_data = {'pokemon': pokemon, 'type': types[0]}

        question = question_template.format(**format_data)

        answer_template = NP_RNG.choice(answer_templates)

        answer = answer_template.format(**format_data)

    return question, answer


def pokemon_to_question_answers(pokemon: str, types: List[str], num_questions_per_pokemon: int =QUESTIONS_PER_POKEMON) -> Tuple[List[str], List[str]]:
    replace = num_questions_per_pokemon > len(QUESTION_TEMPLATES)

    question_templates = NP_RNG.choice(QUESTION_TEMPLATES, num_questions_per_pokemon, replace=replace)

    questions = []
    answers = []

    for template in question_templates:
        question, answer = pokemon_to_qa_pair(template, pokemon, types)

        questions.append(question)
        answers.append(answer)

    return questions, answers


def generate_qa_pairs(examples, num_questions_per_pokemon: int = QUESTIONS_PER_POKEMON):
    """
    Basically the same as generate_type_sentneces, but each element in the dataset is a list of questions and answers
    since the model may do poorly depending on the randomly chosen prompt format (even though the data is the same).
    """

    all_questions = []
    all_answers = []

    for pokemon, types in zip(examples['pokemon'], examples['types']):
        questions, answers = pokemon_to_question_answers(pokemon, types, num_questions_per_pokemon)

        all_questions.append(questions)
        all_answers.append(answers)

    return {'questions': all_questions, 'answers': all_answers}


def TypeDataset() -> Dataset:
    return  Dataset.from_generator(dataset_gen)

def TypeSentenceDataset(type_dataset: Dataset, num_sentences_per_pokemon=TYPE_SENTENCES_PER_POKEMON) -> Dataset:
    return type_dataset.map(
        generate_type_sentences,
        batched=True,
        batch_size=100,
        remove_columns=['pokemon', 'types'],
        fn_kwargs={'num_sentences_per_pokemon': num_sentences_per_pokemon}
    )

def TypeQADataset(type_dataset: Dataset, num_questions_per_pokemon=QUESTIONS_PER_POKEMON) -> Dataset:
    return type_dataset.map(
        generate_qa_pairs,
        batched=True,
        batch_size=100,
        remove_columns=['pokemon'],
        fn_kwargs={'num_questions_per_pokemon': num_questions_per_pokemon}
    )

if __name__ == '__main__':
    # tests to make sure everything is still working
    print(pokemon_to_sentences('Bulbasaur', ['Grass', 'Poison']))

    print(pokemon_to_question_answers('Bulbasaur', ['Grass', 'Poison']))


    type_dataset = Dataset.from_generator(dataset_gen)

    print(len(type_dataset))
    print(type_dataset[0])


    type_sentence_dataset = type_dataset.map(generate_type_sentences, batched=True, batch_size=100, remove_columns=['pokemon', 'types'])
    print(len(type_sentence_dataset))
    print(type_sentence_dataset[1000])

    qa_pair_dataset = type_dataset.map(generate_qa_pairs, batched=True, batch_size=100, remove_columns=['pokemon'])
    print(len(qa_pair_dataset))
    print(qa_pair_dataset[6])


    qa_pair_dataset_big = TypeQADataset(type_dataset, 16)
    print(len(qa_pair_dataset_big[0]['questions']))