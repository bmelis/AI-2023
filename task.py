from simpleai.search import CspProblem, backtrack

input_first = input("Enter the first word: ").upper()
input_sec = input("Enter the second word: ").upper()
input_result = input("Enter the result word: ").upper()

variables = tuple(input_first) + tuple(input_sec) + tuple(input_result)
print(input_first)
print(input_sec)
print("+------------------")
print(input_result)

variables = set(variables)

# Define the list of values that each variable can take
domains = {vari: list(range(1, 10)) if vari in (input_first[0], input_sec[0], input_result[0]) else list(range(0, 10)) for vari in variables}

def constraint_unique(variable, value):
    # Ensure that each variable has a unique value
    return len(value) == len(set(value))

def constraint_add(variables, values):
    # Extract the input words and the result word
    input_words = [input_first, input_sec]
    result_word = input_result

    # Initialize dictionaries to store the values of each word
    word_values = [{letter: None for letter in word} for word in input_words + [result_word]]

    # Assign values to the letters based on the variable assignments
    for letter, value in zip(variables, values):
        for i, word in enumerate(word_values):
            if letter in word:
                word[letter] = value

    # Calculate the numeric representation of each word
    input_numeric = [int("".join([str(word_values[i][letter]) for letter in word])) for i, word in enumerate(input_words)]
    result_numeric = int("".join([str(word_values[2][letter]) for letter in result_word]))

    # Check if the addition of input words matches the result word
    return sum(input_numeric) == result_numeric

constraints = [
    (vari, constraint_unique) for vari in variables  # Ensure unique values for each variable
] + [
    (variables, constraint_add)  # Ensure the addition constraint
]

problem = CspProblem(variables, domains, constraints)

output = backtrack(problem)

if output is not None:
    # Extract the values assigned to the variables
    values = {vari: output[vari] for vari in variables}

    # Create the formatted strings for the input words and the horizontal line
    input_first_str = input_first
    input_sec_str = input_sec
    result_str = input_result
    line = "+"

    for vari in variables:
        if vari in input_first:
            input_first_str = input_first_str.replace(vari, str(values[vari]))
        if vari in input_sec:
            input_sec_str = input_sec_str.replace(vari, str(values[vari]))
        if vari in input_result:
            result_str = result_str.replace(vari, str(values[vari]))
            line += "-" * len(str(values[vari]))
        else:
            line += " "

    # Print the formatted result
    print(input_first_str)
    print(input_sec_str)
    print("+------------------")    
    print(result_str)
