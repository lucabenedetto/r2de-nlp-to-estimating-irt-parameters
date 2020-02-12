"""
Given the ground truth latent traits and the values estimated with the best performing model, this script performs the
evaluation of the accuracy on the performance prediction task. As described in the paper, these methods for predicting
the performance are used:
- ground truth IRT latent traits
- default latent traits (difficulty=0, discrimination=1)
- latent traits of test questions estimated with R2DE and default latent traits for train questions
- latent traits of test questions estimated with R2DE and IRT latent traits for train questions (this is the real-world
    scenario, as the new questions will be used to assess students together with previously existing - and therefore
    calibrated - questions)
Results are saved in the output file specified at the beginning of the script.
"""
import pandas as pd
import pickle
from r2de.utils.evaluation_metrics import gen_output
from r2de.models.predict_model import irt_prediction_with_update
from r2de.constants import (
    CORRECT_HEADER,
    DATA_PATH,
    DS_VAL_FILENAME,
    USER_ID_HEADER,
    TIMESTAMP_HEADER,
    DEFAULT_DISCRIMINATION,
    DIFFICULTY_MIN,
    DIFFICULTY_MAX,
)

output_filename = 'output-performance-prediction.txt'
file = open(output_filename, 'w')

# get the dataset to perform the prediction on and sort it
df_VAL = pd.read_csv(DATA_PATH + DS_VAL_FILENAME).sort_values([USER_ID_HEADER, TIMESTAMP_HEADER], ascending=True)

# get the ID of the test questions and the train questions
train_questions_ids = pickle.load(open(DATA_PATH + 'train-questions.p', "rb"))
test_question_ids = pickle.load(open(DATA_PATH + 'test-questions.p', "rb"))
list_question_ids = list(train_questions_ids)+list(test_question_ids)

# get the true difficulty and the true discrimination
irt_diff_dict = pickle.load(open(DATA_PATH + 'true-difficulty-dict.p', "rb"))
irt_discr_dict = pickle.load(open(DATA_PATH + 'true-discrimination-dict.p', "rb"))

# get the estimated difficulty and the estimated discrimination
nlp_diff_dict = pickle.load(open(DATA_PATH + 'predicted-difficulty-enc2-dict.p', "rb"))
nlp_discr_dict = pickle.load(open(DATA_PATH + 'predicted-discrimination-enc2-dict.p', "rb"))
if set(nlp_diff_dict.keys()) != set(nlp_discr_dict.keys()):
    print("[WARNING]: the estimated difficulty and discrimination dict contain different question IDs")
# create the two dictionaries as in the paper: one is filled with the real difficulty, the other with the real discr
nlp_dflt_diff_dict = nlp_diff_dict.copy()
nlp_dflt_discr_dict = nlp_discr_dict.copy()
nlp_irt_diff_dict = nlp_diff_dict.copy()
nlp_irt_discr_dict = nlp_discr_dict.copy()
for q_id in train_questions_ids:
    nlp_dflt_diff_dict[q_id] = (DIFFICULTY_MAX+DIFFICULTY_MIN)/2
    nlp_dflt_discr_dict[q_id] = DEFAULT_DISCRIMINATION
    nlp_irt_diff_dict[q_id] = irt_diff_dict[q_id]
    nlp_irt_discr_dict[q_id] = irt_discr_dict[q_id]

# create the "default" dictionary
default_diff_dict = {q_id: (DIFFICULTY_MAX+DIFFICULTY_MIN)/2 for q_id in list_question_ids}
default_discr_dict = {q_id: DEFAULT_DISCRIMINATION for q_id in list_question_ids}

# collect the list of users
user_id_list = list(df_VAL[USER_ID_HEADER].unique())

# collect the true results
true_results = df_VAL[CORRECT_HEADER].values

print("[INFO] Prediction with IRT estimated latent traits...")
irt_predicted_results = irt_prediction_with_update(df_VAL, irt_diff_dict, irt_discr_dict, user_id_list)
pickle.dump(irt_predicted_results, open(DATA_PATH + 'performance-prediction-irt.p', 'wb'))
print("[INFO] Done")

print("[INFO] Prediction with NLP estimated (+ default) latent traits...")
nlp_dflt_predicted_results = irt_prediction_with_update(df_VAL, nlp_dflt_diff_dict, nlp_dflt_discr_dict, user_id_list)
pickle.dump(nlp_dflt_predicted_results, open(DATA_PATH + 'performance-prediction-nlp-and-default.p', 'wb'))
print("[INFO] Done")

print("[INFO] Prediction with NLP estimated (+ IRT) latent traits...")
nlp_irt_predicted_results = irt_prediction_with_update(df_VAL, nlp_irt_diff_dict, nlp_irt_discr_dict, user_id_list)
pickle.dump(nlp_irt_predicted_results, open(DATA_PATH + 'performance-prediction-nlp-and-irt.p', 'wb'))
print("[INFO] Done")

print("[INFO] Prediction with default latent traits...")
default_predicted_results = irt_prediction_with_update(df_VAL, default_diff_dict, default_discr_dict, user_id_list)
pickle.dump(default_predicted_results, open(DATA_PATH + 'performance-prediction-default.p', 'wb'))
print("[INFO] Done")

print("[INFO] Below, the results of performance prediction:")
irt_predicted_results = [x >= 0.5 for x in irt_predicted_results]
nlp_dflt_predicted_results = [x >= 0.5 for x in nlp_dflt_predicted_results]
nlp_irt_predicted_results = [x >= 0.5 for x in nlp_irt_predicted_results]
default_predicted_results = [x >= 0.5 for x in default_predicted_results]

output_string = 'IRT estimated latent traits: '
output_string += gen_output(irt_predicted_results, true_results)
print(output_string)
file.write(output_string)

output_string = 'NLP estimated (+ default) latent traits: '
output_string += gen_output(nlp_dflt_predicted_results, true_results)
print(output_string)
file.write(output_string)

output_string = 'NLP estimated (+ IRT) latent traits: '
output_string += gen_output(nlp_irt_predicted_results, true_results)
print(output_string)
file.write(output_string)

output_string = 'default latent traits: '
output_string += gen_output(default_predicted_results, true_results)
print(output_string)
file.write(output_string)

file.close()
