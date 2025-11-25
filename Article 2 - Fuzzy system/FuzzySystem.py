import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyClassifier:
    def __init__(self, mean_values, std_values):
        self.prob_classe1 = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'Probability for category A')
        self.prob_classe2 = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'Probability for category B')
        self.prob_classe3 = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'Probability for category C')
        self.classe_final = ctrl.Consequent(np.arange(1, 4, 1), 'Final classification')

        m_c1_high = mean_values[0][0]
        m_c1_medium = mean_values[0][1]
        m_c1_low = mean_values[0][2]

        m_c2_high = mean_values[1][0]
        m_c2_medium = mean_values[1][1]
        m_c2_low = mean_values[1][2]

        m_c3_high = mean_values[2][0]
        m_c3_medium = mean_values[2][1]
        m_c3_low = mean_values[2][2]

        s_c1_high = std_values[0][0]
        s_c1_medium = std_values[0][1]
        s_c1_low = std_values[0][2]

        s_c2_high = std_values[1][0]
        s_c2_medium = std_values[1][1]
        s_c2_low = std_values[1][2]

        s_c3_high = std_values[2][0]
        s_c3_medium = std_values[2][1]
        s_c3_low = std_values[2][2]


        self._define_membership_functions(m_c1_high, m_c1_medium, m_c1_low, m_c2_high, m_c2_medium, m_c2_low, m_c3_high, 
                                          m_c3_medium, m_c3_low, s_c1_high, s_c1_medium, s_c1_low, s_c2_high, s_c2_medium,
                                          s_c2_low, s_c3_high, s_c3_medium, s_c3_low)

        self.classe_final['Category A'] = fuzz.trimf(self.classe_final.universe, [0.5, 1, 1.5])
        self.classe_final['Category B'] = fuzz.trimf(self.classe_final.universe, [1.5, 2, 2.5])
        self.classe_final['Category C'] = fuzz.trimf(self.classe_final.universe, [2.5, 3, 3.5])

        self.classe_final.defuzzify_method = 'centroid'#'mom'

        self.rules = self._define_rules()

        self.classification_ctrl = ctrl.ControlSystem(self.rules)
        self.classification = ctrl.ControlSystemSimulation(self.classification_ctrl)

    def _define_membership_functions(self, m_c1_high, m_c1_medium, m_c1_low, m_c2_high, m_c2_medium, m_c2_low, m_c3_high, 
                                          m_c3_medium, m_c3_low, s_c1_high, s_c1_medium, s_c1_low, s_c2_high, s_c2_medium,
                                          s_c2_low, s_c3_high, s_c3_medium, s_c3_low):
        

        self.prob_classe1['low'] = fuzz.gaussmf(self.prob_classe1.universe, m_c1_low, s_c1_low)
        self.prob_classe1['medium'] = fuzz.gaussmf(self.prob_classe1.universe, m_c1_medium, s_c1_medium)
        self.prob_classe1['high'] = fuzz.gaussmf(self.prob_classe1.universe, m_c1_high, s_c1_high)

        self.prob_classe2['low'] = fuzz.gaussmf(self.prob_classe2.universe, m_c2_low, s_c2_low)
        self.prob_classe2['medium'] = fuzz.gaussmf(self.prob_classe2.universe, m_c2_medium, s_c2_medium)
        self.prob_classe2['high'] = fuzz.gaussmf(self.prob_classe2.universe, m_c2_high, s_c2_high)

        self.prob_classe3['low'] = fuzz.gaussmf(self.prob_classe3.universe, m_c3_low, s_c3_low)
        self.prob_classe3['medium'] = fuzz.gaussmf(self.prob_classe3.universe, m_c3_medium, s_c3_medium)
        self.prob_classe3['high'] = fuzz.gaussmf(self.prob_classe3.universe, m_c3_high, s_c3_high)

    def _define_rules(self):
        rules = [
                ctrl.Rule(self.prob_classe1['low'] & self.prob_classe2['medium'] & self.prob_classe3['high'], self.classe_final['Category C']),
                ctrl.Rule(self.prob_classe1['low'] & self.prob_classe2['low'] & self.prob_classe3['medium'], self.classe_final['Category C']),
                ctrl.Rule(self.prob_classe1['low'] & self.prob_classe2['medium'] & self.prob_classe3['medium'], self.classe_final['Category B']),
                ctrl.Rule(self.prob_classe1['low'] & self.prob_classe2['medium'] & self.prob_classe3['low'], self.classe_final['Category B']),
                ctrl.Rule(self.prob_classe1['low'] & self.prob_classe2['high'] & self.prob_classe3['low'], self.classe_final['Category B']),
                ctrl.Rule(self.prob_classe1['low'] & self.prob_classe2['high'] & self.prob_classe3['medium'], self.classe_final['Category B']),
                ctrl.Rule(self.prob_classe1['low'] & self.prob_classe2['high'] & self.prob_classe3['high'], self.classe_final['Category C']),

                ctrl.Rule(self.prob_classe1['medium'] & self.prob_classe2['low'] & self.prob_classe3['low'], self.classe_final['Category A']),
                ctrl.Rule(self.prob_classe1['medium'] & self.prob_classe2['low'] & self.prob_classe3['medium'], self.classe_final['Category C']),
                ctrl.Rule(self.prob_classe1['medium'] & self.prob_classe2['low'] & self.prob_classe3['high'], self.classe_final['Category C']),
                ctrl.Rule(self.prob_classe1['medium'] & self.prob_classe2['medium'] & self.prob_classe3['low'], self.classe_final['Category A']),
                ctrl.Rule(self.prob_classe1['medium'] & self.prob_classe2['medium'] & self.prob_classe3['medium'], self.classe_final['Category B']),
                ctrl.Rule(self.prob_classe1['medium'] & self.prob_classe2['medium'] & self.prob_classe3['high'], self.classe_final['Category C']),
                ctrl.Rule(self.prob_classe1['medium'] & self.prob_classe2['high'] & self.prob_classe3['low'], self.classe_final['Category B']),
                ctrl.Rule(self.prob_classe1['medium'] & self.prob_classe2['high'] & self.prob_classe3['medium'], self.classe_final['Category B']),
                ctrl.Rule(self.prob_classe1['medium'] & self.prob_classe2['high'] & self.prob_classe3['high'], self.classe_final['Category C']),

                ctrl.Rule(self.prob_classe1['high'] & self.prob_classe2['low'] & self.prob_classe3['low'], self.classe_final['Category A']),
                ctrl.Rule(self.prob_classe1['high'] & self.prob_classe2['low'] & self.prob_classe3['medium'], self.classe_final['Category A']),
                ctrl.Rule(self.prob_classe1['high'] & self.prob_classe2['low'] & self.prob_classe3['high'], self.classe_final['Category C']),
                ctrl.Rule(self.prob_classe1['high'] & self.prob_classe2['medium'] & self.prob_classe3['low'], self.classe_final['Category A']),
                ctrl.Rule(self.prob_classe1['high'] & self.prob_classe2['medium'] & self.prob_classe3['medium'], self.classe_final['Category A']),
                ctrl.Rule(self.prob_classe1['high'] & self.prob_classe2['medium'] & self.prob_classe3['high'], self.classe_final['Category C']),
                ctrl.Rule(self.prob_classe1['high'] & self.prob_classe2['high'] & self.prob_classe3['low'], self.classe_final['Category A']),
                ctrl.Rule(self.prob_classe1['high'] & self.prob_classe2['high'] & self.prob_classe3['medium'], self.classe_final['Category A']),
                ctrl.Rule(self.prob_classe1['high'] & self.prob_classe2['high'] & self.prob_classe3['high'], self.classe_final['Category C'])
        ]
        return rules

    def classify(self, dataset):
        results = []
        for i in range(len(dataset[0])):
            models_probabilities = [model[i] for model in dataset]

            prob_C1 = np.max([prob[0] for prob in models_probabilities])

            prob_C2 = np.max([prob[1] for prob in models_probabilities])

            prob_C3 = np.max([prob[2] for prob in models_probabilities])


            self.classification.input['Probability for category A'] = prob_C1
            self.classification.input['Probability for category B'] = prob_C2
            self.classification.input['Probability for category C'] = prob_C3

            self.classification.compute()
            results.append(int(round(self.classification.output['Final classification'])))

        return [0 if r == 1 else 1 if r == 2 else 2 for r in results]


def classify_by_rule(data, class_column, id, prob_columns):

    results = {}
    for class_label in data[class_column].unique():
        class_data = data[data[class_column] == class_label]
        mean_probs = class_data[prob_columns].mean(axis=1)
        mean_value = mean_probs.mean()
        std_value = mean_probs.std()

        results[class_label] = {"class": id, "mean": mean_value, "std": std_value}
    
    return results


def prob_class(data_train, train_set, id_class):

    prob_columns = []
    for i, model_probs in enumerate(train_set):
        column_name = f"prob_model_{i+1}"
        data_train[column_name] = model_probs[:, id_class]
        prob_columns.append(column_name)
        return prob_columns


def return_means_std(data_train, train_set):

    classifications = []

    for id_class in range(0, 3):

        prob_columns = prob_class(data_train, train_set, id_class)
        classification = classify_by_rule(data_train, class_column="y", id = id_class, prob_columns=prob_columns)
        classifications.append(classification)

    class_1_high_mean = classifications[0]['A']['mean']
    class_1_medium_mean = classifications[0]['B']['mean']
    class_1_low_mean = classifications[0]['C']['mean']

    class_1_high_std = classifications[0]['A']['std']
    class_1_medium_std = classifications[0]['B']['std']
    class_1_low_std = classifications[0]['C']['std']


    class_2_high_mean = classifications[1]['B']['mean']
    class_2_medium_mean = classifications[1]['C']['mean']
    class_2_low_mean = classifications[1]['A']['mean']

    class_2_high_std = classifications[1]['B']['std']
    class_2_medium_std = classifications[1]['C']['std']
    class_2_low_std = classifications[1]['A']['std']


    class_3_high_mean = classifications[2]['C']['mean']
    class_3_medium_mean = classifications[2]['B']['mean']
    class_3_low_mean = classifications[2]['A']['mean']

    class_3_high_std = classifications[2]['C']['std']
    class_3_medium_std = classifications[2]['B']['std']
    class_3_low_std = classifications[2]['A']['std']

    mean_values  = [[class_1_high_mean, class_1_medium_mean, class_1_low_mean], [class_2_high_mean, class_2_medium_mean, class_2_low_mean], [class_3_high_mean, class_3_medium_mean, class_3_low_mean]]

    std_values  = [[class_1_high_std, class_1_medium_std, class_1_low_std], [class_2_high_std, class_2_medium_std, class_2_low_std], [class_3_high_std, class_3_medium_std, class_3_low_std]]
    
    return mean_values, std_values