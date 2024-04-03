import model_interactor
import requirements_validator
import reports
import parameters


def read_questions_from_file():
    # this code reads the questions from the txt file, parses it into the chat template
    # and returns the resulting dictionary
    questionnaire = []
    try:
        with open(parameters.questions_path, "r") as file:
            questions = file.readlines()
    except FileNotFoundError:
        print("Questions file not found...")
        return
    else:
        for q in questions:
            questionnaire.append({"role": "user", "content": q.replace("\n", "")})
    return questionnaire


def prepare_experiment():
    if not requirements_validator.verify_cuda():
        return []
    return read_questions_from_file()


def run_experiment(questionnaire):   # WIP
    mi = model_interactor.ModelInteractor()
    output = []
    index = 0
    for q in questionnaire:
        output[q["content"]] = mi.ask_question(q)
        print(index + 1)
        # output[q]["answers"] = mi.ask_question(q)
        # save questions in a variable

    for f in output:
        print(f)

    return output


def save_results(output):
    rep = reports.Reporter(output)
    rep.save_results()


if __name__ == "__main__":
    question_template = prepare_experiment()
    if not len(question_template) > 0:
        print("Error detected in a requirement, please check that data and parameter files are properly placed.")
    else:
        print("Requirements ready! Running experiment")
        results = run_experiment(question_template)
        # save_results(results)
