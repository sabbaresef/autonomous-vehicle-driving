import os
import csv
import glob
import json


############################################################################################################
# Set the model for which compute validation score

MODEL = "Resnet34_RGBD_128_batch_10h_training_5h_validation_linear_40_epochs"
############################################################################################################


def results_extraction(model):
    """This function extracts the results from the folder from the summary file and 
    calculates the validation scores, written in the _validation directory."""
    
    results_dir = '_benchmarks_results'
    results_model = glob.glob(results_dir + "/Validation_" + model + "_Validation" + "*")
    
    print(f"Results folders: {results_model}")
    print(f"\n\nResults folders number: {len(results_model)}")

    succes_rate_scenario = {}
    
    if len(results_model) == 2:
        weights = [0.66, 0.33]
    elif len(results_model) == 3:
        weights = [0.50, 0.25, 0.25]
    else:
        print(f"Validation scenarios don't match validation size: {len(results_model)}")
        exit(0)
    
    print(f"\nValidate on {len(results_model)} scenarios, weights are {weights}")

    for results in results_model:

        summary_file = os.path.join(results, 'summary.csv')
        
        print("\nNew file!!!")
        
        positive_count = 0
        with open(summary_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                if row['result'] == '1':
                    positive_count += 1
            
            succes_rate_scenario.update({results : positive_count / 25})
            
    print(f"\nSucces rate on scenarios: {succes_rate_scenario}")
    
    for results in results_model:
        if 'Validation_new_weather_town' in results:
            new_weather_town = succes_rate_scenario[results] * weights[0]
        elif 'Validation_new_town' in results:
            new_town = succes_rate_scenario[results] * weights[1]
        elif 'Validation_new_weather' in results and 'Validation_new_weather_town' not in results:
            new_weather = succes_rate_scenario[results] * weights[2]
                
    validation_score = 0
    if len(results_model) == 3:
        validation_score = new_weather + new_town + new_weather_town
    else:
        validation_score = new_town + new_weather_town
    
    print(f"\nValidation score: {validation_score}\n")

    validation_log = os.path.join('validation', model)
    validation_score_to_write = {'validation_score' : validation_score}
    
    if len(results_model) == 3:
        validation_score_to_write.update({'success_new_weather' : new_weather / weights[2]})
        validation_score_to_write.update({'weight_new_weather' : weights[2]})
        validation_score_to_write.update({'validation_score_new_weather' : new_weather})
    
    validation_score_to_write.update({'success_new_town' : new_town / weights[1]})
    validation_score_to_write.update({'weight_new_town' : weights[1]})
    validation_score_to_write.update({'validation_score_new_town' : new_town})
    
    validation_score_to_write.update({'success_new_weather_town' : new_weather_town / weights[0]})
    validation_score_to_write.update({'weight_new_weather_town' : weights[0]})
    validation_score_to_write.update({'validation_score_new_weather_town' : new_weather_town})
    

    with open(os.path.join(validation_log, "validation_score.json"), 'w') as f:
        json.dump(validation_score_to_write, f)
        


if __name__ == '__main__':
    results_extraction(MODEL)


            

                



