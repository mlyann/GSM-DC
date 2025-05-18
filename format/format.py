def format_prompt(use_finetune, problem, op=None, nshots=None, cur_id_gen=None, tokenizer=None, generate_problem=None):
    if use_finetune:
        return f"<|start_header_id|>user<|end_header_id|>\n\n"+problem+f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        assert(cur_id_gen != None)
        assert(tokenizer != None)
        assert(generate_problem != None)
        assert(nshots != None)
        assert(op != None)
        system_instruction = (
            "You're an expert at solving elementary math problems involving addition, subtraction, "
            "and multiplication. You solve all the problems in a uniform format. All calculations "
            "are done modulo 5. For example, 3 + 2 equals 0, 1 + 1 equals 2, 4 + 2 + 4 equals 0, "
            "3 * 2 equals 1, and 3 * 1 equals 3. When providing your solution, please end with "
            "'The final answer is <<x>>.' where x is your final answer, an integer between 0 and 4. "
            "You must solve all the problems using the same solution format.\n\n"
            "Our scenarios involve up to four categories of objects: schools, classrooms, backpacks "
            "and stationeries. Each school may contain classrooms, each classroom may contain backpacks, "
            "and each backpack may contain stationeries. We can specify quantities, such as "
            "'the number of dance studios at each Lakeshore High.' Assume that every entity with "
            "the same name has an identical configuration; for example, each Lakeshore High contains "
            "the same number of dance studios. Another guiding principle is that what is not mentioned "
            "does not exist: when we refer to classrooms at Lakeshore High, we are only discussing "
            "the classrooms explicitly mentioned in our scenario. Furthermore, if Lakeshore High "
            "is not even mentioned, any classroom within it is automatically considered to be "
            "non-existent (i.e. 0)."
        )
    
        few_shot_examples = ""
        for i in range(nshots):
            shot_id_gen = generate_problem(op=op)
            q = tokenizer.decode(shot_id_gen.prob_token)
            a = tokenizer.decode(shot_id_gen.sol_token)
            few_shot_examples += (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{shot_id_gen.gen_background()}\nThe problem description is: {q}<|eot_id|>\n\n"
                "<|start_header_id|>assistant<|end_header_id|>\n"
                f"{a}<|eot_id|>\n\n"
            )
    
        final_prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n{system_instruction}<|eot_id|>\n\n"
            f"{few_shot_examples}"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{cur_id_gen.gen_background()}\nThe problem description is: {problem}<|eot_id|>\n\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

        return final_prompt