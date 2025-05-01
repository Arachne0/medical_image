def calculate_metrics(true_labels, predicted_labels):
    num_classes = true_labels.shape[1]

    metrics = {'accuracy': [], 'sensitivity': [], 'specificity': []}

    for i in range(num_classes):
        true_positives = ((predicted_labels[:, i] == 1) & (true_labels[:, i] == 1)).sum().item()
        true_negatives = ((predicted_labels[:, i] == 0) & (true_labels[:, i] == 0)).sum().item()
        false_positives = ((predicted_labels[:, i] == 1) & (true_labels[:, i] == 0)).sum().item()
        false_negatives = ((predicted_labels[:, i] == 0) & (true_labels[:, i] == 1)).sum().item()

        accuracy = (true_positives + true_negatives) / (
                    true_positives + true_negatives + false_positives + false_negatives)

        if true_positives + false_negatives > 0:
            sensitivity = true_positives / (true_positives + false_negatives)
        else:
            sensitivity = 0.0

        if true_negatives + false_positives > 0:
            specificity = true_negatives / (true_negatives + false_positives)
        else:
            specificity = 0.0

        metrics['accuracy'].append(accuracy)
        metrics['sensitivity'].append(sensitivity)
        metrics['specificity'].append(specificity)

    return metrics