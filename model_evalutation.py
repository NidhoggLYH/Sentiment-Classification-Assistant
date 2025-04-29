import json
import sys


def calculate_accuracy(filename):
    with open(filename, encoding='utf8') as predictions:
        total_inputs, total_rights = 0, 0
        for prediction in predictions:
            prediction = json.loads(prediction)
            true_label = prediction['label'].strip()
            pred_label = prediction['predict']

            total_inputs += 1
            if true_label == pred_label:
                total_rights += 1

            # if pred_label not in ['好评', '差评']:
            #     print(prediction['prompt'], end='')
            #     print(f'{pred_label}')
            #     print('-' * 50)

        accuracy = total_rights / total_inputs
        print(f'样本总数: {total_inputs} 正确数量:{total_rights} 准确率: {accuracy:.2f}')


if __name__ == '__main__':
    calculate_accuracy('generated_predictions.jsonl')