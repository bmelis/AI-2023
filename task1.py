import streamlit as st
from simpleai.search import CspProblem, backtrack
st.title("Taak 1 AI - Bent Melis - r0831245")

input_first = st.text_input("Enter the first word: ").upper()
input_sec = st.text_input("Enter the second word: ").upper()
input_result = st.text_input("Enter the result word: ").upper()

if st.button('Klik om me op te lossen!'):
    variables = tuple(input_first) + tuple(input_sec) + tuple(input_result)
    st.text(input_first)
    st.text(input_sec)
    st.text("+------------------")
    st.text(input_result)

    variables = set(variables)

    domains = {vari: list(range(1, 10)) if vari in (input_first[0], input_sec[0], input_result[0]) else list(range(0, 10)) for vari in variables}

    def constraint_unique(variable, value):
        return len(value) == len(set(value))

    def constraint_add(variables, values):
        input_words = [input_first, input_sec]
        result_word = input_result

        word_values = [{letter: None for letter in word} for word in input_words + [result_word]]

        for letter, value in zip(variables, values):
            for i, word in enumerate(word_values):
                if letter in word:
                    word[letter] = value

        input_numeric = [int("".join([str(word_values[i][letter]) for letter in word])) for i, word in enumerate(input_words)]
        result_numeric = int("".join([str(word_values[2][letter]) for letter in result_word]))

        return sum(input_numeric) == result_numeric
        
    constraints = [
        (variables, constraint_unique),
        (variables, constraint_add)
    ]

    problem = CspProblem(variables, domains, constraints)

    output = backtrack(problem)

    if output is not None:
        values = {vari: output[vari] for vari in variables}

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
                
        st.text(input_first_str)
        st.text(input_sec_str)
    st.text("+------------------")    
    st.text(result_str)

    #print('\nSolutions:', output)
