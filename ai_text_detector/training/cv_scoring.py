import os
import re
import numpy as np
import scipy.stats

N_EXAMPLES = 1083

def extract_accuracies_from_logs(directory):
    accuracies = []
    pattern = r'"eval_accuracy": (\d+\.\d+),'
    for entry in os.listdir(directory):
        dir_path = os.path.join(directory, entry)
        if os.path.isdir(dir_path) and "fold" in entry:
            log_file = os.path.join(dir_path, 'ai-detector-train.log')
            if os.path.isfile(log_file):
                accuracy = None
                with open(log_file, "r") as file:
                    for line in file:
                        match = re.search(pattern, line)
                        if match:
                            accuracy = float(match.group(1))
                if accuracy is not None:
                    accuracies.append(accuracy)
    return accuracies

LOG_DIR = "/work3/s183911/dmiai/even-moar-no-wd/"

def main():
    accuracies = extract_accuracies_from_logs(LOG_DIR)
    mean_accuracy = np.mean(accuracies)

    std_dev = np.std(accuracies, ddof=1)
    z_score = 1.96
    margin_of_error = z_score * (std_dev / np.sqrt(len(accuracies)))
    conf = (mean_accuracy - margin_of_error, mean_accuracy + margin_of_error)


    examples_per_fold = N_EXAMPLES // len(accuracies)
    ci_lower = []
    ci_upper = []
    for accuracy in accuracies:
        se = np.sqrt(accuracy * (1 - accuracy) / examples_per_fold)
        ci_lower_bound = accuracy - z_score * se
        ci_upper_bound = accuracy + z_score * se
        ci_lower.append(ci_lower_bound)
        ci_upper.append(ci_upper_bound)
    print("Mean acc over %i folds: %.2f" % (len(accuracies), mean_accuracy*100))
    print("Average binomial finite sample confidence interval (%.2f, %.2f)" % (np.mean(ci_lower), np.mean(ci_upper)))
    print("%i-fold generalization confidence interval (%.2f, %.2f)" % (len(accuracies), conf[0]*100, conf[1]*100))



main() if "__main__" == __name__ else print("wat?")
