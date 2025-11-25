import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import itertools


# Definir as variáveis fuzzy
prob_classe1 = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'prob_classe1')
prob_classe2 = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'prob_classe2')
prob_classe3 = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'prob_classe3')
classe_final = ctrl.Consequent(np.arange(1, 4, 1), 'classe_final')

# # Definir funções de pertinência gaussianas para probabilidades
# prob_classe1['baixa'] = fuzz.gaussmf(prob_classe1.universe, 0.15, 0.05)
# prob_classe1['media'] = fuzz.gaussmf(prob_classe1.universe, 0.4, 0.05)
# prob_classe1['alta'] = fuzz.gaussmf(prob_classe1.universe, 0.8, 0.1)

# prob_classe2['baixa'] = fuzz.gaussmf(prob_classe2.universe, 0.15, 0.05)
# prob_classe2['media'] = fuzz.gaussmf(prob_classe2.universe, 0.4, 0.05)
# prob_classe2['alta'] = fuzz.gaussmf(prob_classe2.universe, 0.8, 0.1)

# prob_classe3['baixa'] = fuzz.gaussmf(prob_classe3.universe, 0.15, 0.05)
# prob_classe3['media'] = fuzz.gaussmf(prob_classe3.universe, 0.4, 0.05)
# prob_classe3['alta'] = fuzz.gaussmf(prob_classe3.universe, 0.8, 0.1) #0.835, 0.1

# Definir funções de pertinência gaussianas para probabilidades


prob_classe1['baixa'] = fuzz.gaussmf(prob_classe1.universe, 0.15, 0.05)
prob_classe1['media'] = fuzz.gaussmf(prob_classe1.universe, 0.35, 0.15)
prob_classe1['alta'] = fuzz.gaussmf(prob_classe1.universe, 0.8, 0.15)

prob_classe2['baixa'] = fuzz.gaussmf(prob_classe2.universe, 0.15, 0.1)
prob_classe2['media'] = fuzz.gaussmf(prob_classe2.universe, 0.45, 0.15)
prob_classe2['alta'] = fuzz.gaussmf(prob_classe2.universe, 0.75, 0.15)

prob_classe3['baixa'] = fuzz.gaussmf(prob_classe3.universe, 0.05, 0.05)
prob_classe3['media'] = fuzz.gaussmf(prob_classe3.universe, 0.25, 0.15)
prob_classe3['alta'] = fuzz.gaussmf(prob_classe3.universe, 0.6, 0.15)


# Definir funções de pertinência triangulares para a classe final
classe_final['Classe1'] = fuzz.trimf(classe_final.universe, [0.5, 1, 1.5]) 
classe_final['Classe2'] = fuzz.trimf(classe_final.universe, [1.5, 2, 2.5]) 
classe_final['Classe3'] = fuzz.trimf(classe_final.universe, [2.5, 3, 3.5])


classe_final.defuzzify_method = 'mom'  # Padrão é 'centroid', mas pode ser alterado
# Outras opções: 'bisector', 'mom' (mean of maxima), 'som' (smallest of maxima), 'lom' (largest of maxima)




# # Definir as regras fuzzy

# Lista de todas as possíveis combinações de valores fuzzy para as condições
r1 = ctrl.Rule(prob_classe1['baixa'] & prob_classe2['media'] & prob_classe3['alta'], classe_final['Classe3'])
r2 = ctrl.Rule(prob_classe1['baixa'] & prob_classe2['baixa'] & prob_classe3['media'], classe_final['Classe3'])
r3 = ctrl.Rule(prob_classe1['baixa'] & prob_classe2['media'] & prob_classe3['media'], classe_final['Classe2'])
r4 = ctrl.Rule(prob_classe1['baixa'] & prob_classe2['media'] & prob_classe3['baixa'], classe_final['Classe2'])
r5 = ctrl.Rule(prob_classe1['baixa'] & prob_classe2['alta'] & prob_classe3['baixa'], classe_final['Classe2'])
r6 = ctrl.Rule(prob_classe1['baixa'] & prob_classe2['alta'] & prob_classe3['media'], classe_final['Classe2'])
r7 = ctrl.Rule(prob_classe1['baixa'] & prob_classe2['alta'] & prob_classe3['alta'], classe_final['Classe3'])

r8 = ctrl.Rule(prob_classe1['media'] & prob_classe2['baixa'] & prob_classe3['baixa'], classe_final['Classe1'])
r9 = ctrl.Rule(prob_classe1['media'] & prob_classe2['baixa'] & prob_classe3['media'], classe_final['Classe3'])
r10 = ctrl.Rule(prob_classe1['media'] & prob_classe2['baixa'] & prob_classe3['alta'], classe_final['Classe3'])
r11 = ctrl.Rule(prob_classe1['media'] & prob_classe2['media'] & prob_classe3['baixa'], classe_final['Classe1'])
r12 = ctrl.Rule(prob_classe1['media'] & prob_classe2['media'] & prob_classe3['media'], classe_final['Classe2'])
r13 = ctrl.Rule(prob_classe1['media'] & prob_classe2['media'] & prob_classe3['alta'], classe_final['Classe3'])
r14 = ctrl.Rule(prob_classe1['media'] & prob_classe2['alta'] & prob_classe3['baixa'], classe_final['Classe2'])
r15 = ctrl.Rule(prob_classe1['media'] & prob_classe2['alta'] & prob_classe3['media'], classe_final['Classe2'])
r16 = ctrl.Rule(prob_classe1['media'] & prob_classe2['alta'] & prob_classe3['alta'], classe_final['Classe3'])

r17 = ctrl.Rule(prob_classe1['alta'] & prob_classe2['baixa'] & prob_classe3['baixa'], classe_final['Classe1'])
r18 = ctrl.Rule(prob_classe1['alta'] & prob_classe2['baixa'] & prob_classe3['media'], classe_final['Classe1'])
r19 = ctrl.Rule(prob_classe1['alta'] & prob_classe2['baixa'] & prob_classe3['alta'], classe_final['Classe3'])
r20 = ctrl.Rule(prob_classe1['alta'] & prob_classe2['media'] & prob_classe3['baixa'], classe_final['Classe1'])
r21 = ctrl.Rule(prob_classe1['alta'] & prob_classe2['media'] & prob_classe3['media'], classe_final['Classe1'])
r22 = ctrl.Rule(prob_classe1['alta'] & prob_classe2['media'] & prob_classe3['alta'], classe_final['Classe3'])
r23 = ctrl.Rule(prob_classe1['alta'] & prob_classe2['alta'] & prob_classe3['baixa'], classe_final['Classe1'])
r24 = ctrl.Rule(prob_classe1['alta'] & prob_classe2['alta'] & prob_classe3['media'], classe_final['Classe1'])
r25 = ctrl.Rule(prob_classe1['alta'] & prob_classe2['alta'] & prob_classe3['alta'], classe_final['Classe3'])

rules_list = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25]

classification_ctrl = ctrl.ControlSystem(rules_list)
classification = ctrl.ControlSystemSimulation(classification_ctrl)

# Função para calcular a classificação final com base nas probabilidades dos 8 modelos
# def classify(dataset, weights):
def classify(dataset):
    # Lista para armazenar as classificações finais
    results = []
    # weights = np.array(weights)
    # Iterar por cada observação no conjunto de teste
    for i in range(len(dataset[0])):
        # # Extrair as probabilidades para a observação atual de todos os modelos
        models_probabilities = [model[i] for model in dataset]


        # prob_C1 = np.dot(weights, [prob[0] for prob in models_probabilities])
        # prob_C2 = np.dot(weights, [prob[1] for prob in models_probabilities])
        # prob_C3 = np.dot(weights, [prob[2] for prob in models_probabilities])
        
        # # Calcular a média das probabilidades para cada classe
        prob_C1 = np.max([prob[0] for prob in models_probabilities])
        prob_C2 = np.max([prob[1] for prob in models_probabilities])
        prob_C3 = np.max([prob[2] for prob in models_probabilities])
        # if i==len(dataset[0])-1:
            # print('prob_C1', prob_C1)
            # print('prob_C2', prob_C2)
            # print('prob_C3', prob_C3)
            # print("\n")
        
        # Passar as médias para o sistema fuzzy
        classification.input['prob_classe1'] = prob_C1
        classification.input['prob_classe2'] = prob_C2
        classification.input['prob_classe3'] = prob_C3
        
        # Computar o resultado
        classification.compute()
        
        # Adicionar o resultado final à lista de resultados
        results.append(int(round(classification.output['classe_final'])))


        results_end = []
        for i in results:
            if i==1:
                results_end.append(0)
            elif i==2:
                results_end.append(1)
            elif i==3:
                results_end.append(2)
    
    # Retornar a lista de classificações finais
    return results_end