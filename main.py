import numpy as np
import mlrose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, datasets
import time
import warnings
import os
from optima import Optima
from ANN import ANN

output_path = 'outputs\\'
if not os.path.exists(output_path):
    os.mkdir(output_path)

# part 1
optima_KColor = Optima('Max K Color', output_path)
optima_KColor.rhc()
optima_KColor.sa()
optima_KColor.ga()
optima_KColor.mimic()
optima_KColor.timing_curve()
optima_KColor.iteration_curve()

optima_FlipFlop = Optima("Flip Flop", output_path)
optima_FlipFlop.rhc()
optima_FlipFlop.sa()
optima_FlipFlop.ga()
optima_FlipFlop.mimic()
optima_FlipFlop.timing_curve()
optima_FlipFlop.iteration_curve()

optima_OneMax = Optima('One Max', output_path)
optima_OneMax.rhc()
optima_OneMax.sa()
optima_OneMax.ga()
optima_OneMax.mimic()
optima_OneMax.timing_curve()
optima_OneMax.iteration_curve()

# part 2
ANN_analysis = ANN('Heart.csv', output_path)
ANN_analysis.load_data()
ANN_analysis.scale_data()
ANN_analysis.cv_analysis()
ANN_analysis.perform_iteration()
ANN_analysis.learning_curve()