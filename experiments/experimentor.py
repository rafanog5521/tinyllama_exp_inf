import model_interactor
import requirements_validator
import reports
import parameters
import time
from tqdm import tqdm


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
    progress_bar = tqdm(total=len(questionnaire), desc="Processing questionnaire...")
    mi = model_interactor.ModelInteractor()
    output = []
    for q in questionnaire:
        output.append(mi.ask_question(q))
        progress_bar.update(1)

    progress_bar.close()
    print(output)
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
        run_experiment(question_template)
        # results = run_experiment(question_template)
        # save_results(results)
