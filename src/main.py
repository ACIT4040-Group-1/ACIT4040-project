import os.path

import matplotlib.pyplot as plt

from src.red.AdversarialFramework import AdversarialFramework
from tensorflow import keras
from src.common.DataLoader import DataLoader
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
import csv
import pandas as pd


def write_csv(fileName, line_entry):
    f = open(fileName, 'a', newline='')

    # create the csv writer
    writer = csv.writer(f, delimiter=',')

    # write a row to the csv file
    writer.writerow(line_entry)

    # close the file
    f.close()


if __name__ == '__main__':
    detectors = ['combined.h5']
    # , 'denseNet121_detector.h5', 'combined.h5', 'xception1.h5'

    attacks = ['FGSM', 'gaussian']

    #the headers are tacen from the output of the adversirial attack
    # woa_metric_header = ['nPredictions', 'roc_auc_score', 'ap_score', 'real_images', 'fake_images', 'classified',
    #                     'misclassified_real', 'misclassified_fake', 'misclassified', 'detector']

    # wa_metric_header = ['nPredictions', 'roc_auc_score', 'ap_score', 'real_images', 'fake_images', 'classified',
    #                    'misclassified_real', 'misclassified_fake', 'misclassified', 'detector', 'attack']

    label_header = ['real_label', 'predicted_label']

    result_dir = os.path.join('logs', 'results')

    file_name_woa_metrics = os.path.join(result_dir, 'woa_metrics_results.txt')
    # write_csv(file_name_woa_metrics, woa_metric_header)  # add header to metric-file

    wa_file_name_metrics = os.path.join(result_dir, 'wa_metrics_results.txt')
    # write_csv(file_name_metrics, wa_metric_header)  # add header to metric-file

    for detector in detectors:
        blue_model = keras.models.load_model(os.path.join('detectors', detector))

        AF = AdversarialFramework(model=blue_model)
        images = DataLoader().get_data('test')

        # 1 - get performance metrics without attacks(woa)
        # returns the real labels, prediced label and evaluation metris in a list
        # ->[len(predictions), roc_auc_score, ap_score, real_images, fake_images,classified, misclassified_real, misclassified_fake, misclassified]
        woaReal_labels, woaPredictions, woaMetrics = AF.evaluate_model(images=images)
        woaMetrics.append(detector)

        # 1.1 - add all metrics to a csv list
        write_csv(file_name_woa_metrics, woaMetrics)

        # 1.2 - add all labels to a csv
        # print(list(woaPredictions[:,0]))
        # print(woaReal_labels)
        file_name_woa_labels = os.path.join(result_dir, 'woa_' + detector + '_labels.txt')
        df = pd.DataFrame({'real': woaReal_labels, 'predicted': woaPredictions[:, 0]})
        df.to_csv(file_name_woa_labels, index=False)

        # 2 - get performance metrics with attacks (wa)
        root_path = "red/adversarial/images/FGSM"
        if not os.path.exists(root_path):
            # Create new FGSM images of input images
            images = AF.apply_attack(method='FGSM', images=images, create_new_images=True)
        else:
            # Loading already saved attack images (if they exist)
            images = AF.apply_attack(method='FGSM', create_new_images=False)

        waReal_labels, waPredictions, waMetrics = AF.evaluate_model(images=images)
        waMetrics.append(detector)
        waMetrics.append('FGSM')

        # 2.1 - add all metrics to a csv list

        write_csv(wa_file_name_metrics, waMetrics)

        # 2.2 - add all labels to a csv
        file_name_wa_labels = os.path.join(result_dir, 'wa_' + detector + '_labels.txt')
        df = pd.DataFrame({'real': waReal_labels, 'predicted': waPredictions[:, 0]})
        df.to_csv(file_name_wa_labels, index=False)


