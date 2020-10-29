import argparse
import csv
import operator
import os
from collections import OrderedDict
from typing import FrozenSet, List, Dict, Union, Iterable, Callable, Tuple, Any, Optional

Label = str


class ItemEvaluationResult(object):
    label: Label
    expected_count: int
    predicted_count: int

    def __init__(self, label):  # type: (Label) -> None
        self.label = label
        self.expected_count = 0
        self.predicted_count = 0


class ImageEvaluationResult(object):

    def __init__(self, item_evaluation, filename=None):  # type: (List[ItemEvaluationResult], Optional[str]) -> None
        self.filename = filename
        self.item_evaluation = item_evaluation

    def get_predicted_ratio(self, label=None):  # type: (Optional[Label]) -> Optional[float]
        expected_count = self.get_expected_count(label)
        if expected_count == 0:
            return None
        return float(self.get_predicted_count(label)) / float(expected_count)

    def get_predicted_percent(self, label=None):  # type: (Optional[Label]) -> str
        ratio = self.get_predicted_ratio(label)
        if ratio is None:
            return 'N/A'
        return '{:.2f}'.format(ratio).replace('.', ',')

    def get_predicted_count(self, label=None):  # type: (Optional[Label]) -> int
        items = self.item_evaluation if label is None else self.get_items_with_label(label)
        return sum(map(lambda item_result: item_result.predicted_count, items))

    def get_expected_count(self, label=None):  # type: (Optional[Label]) -> int
        items = self.item_evaluation if label is None else self.get_items_with_label(label)
        return sum(map(lambda item_result: item_result.expected_count, items))

    def get_items_with_label(self, label):  # type: (Label) -> Iterable[ItemEvaluationResult]
        return filter(lambda item: item.label == label, self.item_evaluation)


class ItemDescription(object):
    label: Label
    xmin: int
    ymin: int
    xmax: int
    ymax: int


HeaderName = str
FieldValueGetter = Callable[[ImageEvaluationResult], Any]
ReportField = Tuple[HeaderName, FieldValueGetter]


def get_basename_without_extension(filename, extra_suffix=None):  # type: (str, Optional[str]) -> str
    basename = os.path.basename(filename)
    if extra_suffix is not None and basename.endswith(extra_suffix):
        basename = basename.replace(extra_suffix, '')
    return os.path.splitext(basename)[0]


def evaluate_prediction_results(original_labels_filenames, predicted_labels_filenames):
    # type: (List[str], List[str]) -> List[ImageEvaluationResult]

    original_count = len(original_labels_filenames)
    predicted_count = len(predicted_labels_filenames)

    original_basenames = frozenset([get_basename_without_extension(filename) for filename in original_labels_filenames])
    predicted_basenames = frozenset(
        [get_basename_without_extension(filename, '.prediction.txt') for filename in predicted_labels_filenames])
    common_basenames = original_basenames.intersection(predicted_basenames)
    print(original_basenames)
    print(predicted_basenames)
    original_labels_filenames = [filename for filename in original_labels_filenames if
                                 any(map(lambda basename: basename in filename, common_basenames))]
    predicted_labels_filenames = [filename for filename in predicted_labels_filenames if
                                  any(map(lambda basename: basename in filename, common_basenames))]

    count = len(original_labels_filenames)
    print(f'''Comparing {count} matching files out of ({original_count}) originals and ({predicted_count}) predictions.''')

    results = []  # type: List[ImageEvaluationResult]
    for i in range(0, count):
        original_items = [parse_item_description(descr) for descr in read_lines(original_labels_filenames[i])]
        predicted_items = [parse_item_prediction(descr) for descr in read_lines(predicted_labels_filenames[i])]
        item_evaluation = compare_items(original_items, predicted_items)

        no_ext_basename = get_basename_without_extension(original_labels_filenames[i])
        results.append(ImageEvaluationResult(item_evaluation, f'{no_ext_basename}.png'))

    item_accumulated_results = []
    all_labels = set_union(map(get_labels, results))
    for label in all_labels:
        item_accumulated = ItemEvaluationResult(label)
        item_accumulated.expected_count = sum([image_result.get_expected_count(label) for image_result in results])
        item_accumulated.predicted_count = sum([image_result.get_predicted_count(label) for image_result in results])
        item_accumulated_results.append(item_accumulated)

    results.append(ImageEvaluationResult(item_accumulated_results))

    return results


def compare_items(original_items, predicted_items):
    # type: (List[ItemDescription], List[ItemDescription]) -> List[ItemEvaluationResult]

    original_items = [item for item in original_items if item.label != 'case']
    predicted_items = [item for item in predicted_items if item.label != 'case']

    original_labels = frozenset([item.label for item in original_items])
    predicted_labels = frozenset([item.label for item in predicted_items])
    all_labels = original_labels.union(predicted_labels)

    results = {}
    for label in all_labels:
        results[label] = ItemEvaluationResult(label)

    for item in original_items:
        results[item.label].expected_count += 1

    for item in predicted_items:
        results[item.label].predicted_count += 1

    return list(results.values())


def parse_item_description(description):  # type: (str) -> ItemDescription
    values = description.split(' ')
    item = ItemDescription()
    item.label = values[0]
    item.xmin, item.ymin, item.xmax, item.ymax = map(int, values[4:8])
    return item


def parse_item_prediction(prediction):  # type: (str) -> ItemDescription
    values = prediction.split(' ')
    item = ItemDescription()
    item.label = values[0]
    item.xmin, item.ymin, item.xmax, item.ymax = map(int, values[2:6])
    return item


def read_lines(filename):  # type: (str) -> List[str]
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def set_union(sets):  # type: (Iterable[FrozenSet]) -> FrozenSet
    result = frozenset()
    for partial in sets:
        result = result.union(partial)
    return result


def get_labels(result):  # type: (Union[ImageEvaluationResult, ItemEvaluationResult]) -> FrozenSet[Label]
    if isinstance(result, ImageEvaluationResult):
        return set_union(map(get_labels, result.item_evaluation))
    elif isinstance(result, ItemEvaluationResult):
        return frozenset([result.label, ])


def get_report_fields_for_label(label):  # type: (Label) -> List[ReportField]
    return [
        (f'E/{label}', lambda image_result: image_result.get_expected_count(label)),
        (f'F/{label}', lambda image_result: image_result.get_predicted_count(label)),
        (f'P/{label}', lambda image_result: image_result.get_predicted_percent(label))
    ]


def export_evaluation_results(results):  # type: (List[ImageEvaluationResult]) -> None
    all_labels = set_union(map(get_labels, results))

    report_fields = [
        ('filename', operator.attrgetter('filename'))
    ]

    for label_fields in map(get_report_fields_for_label, all_labels):
        report_fields.extend(label_fields)

    report_fields.extend([
        ('E/all', ImageEvaluationResult.get_expected_count),
        ('F/all', ImageEvaluationResult.get_predicted_count),
        ('P/all', ImageEvaluationResult.get_predicted_percent)
    ])

    field_names = [field[0] for field in report_fields]
    rows = [image_result_to_dict(report_fields, image_result) for image_result in results]
    print(list(rows))
    with open('evaluation_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, field_names, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def image_result_to_dict(report_fields, image_result):  # type: (List[ReportField], ImageEvaluationResult) -> Dict[str, Any]
    return OrderedDict([(field_name, field_getter(image_result)) for field_name, field_getter in report_fields])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate prediction results.')
    parser.add_argument("--original-labels-file", type=str, default='original_labels.txt',
                        help="File containing paths to original labels. DEFAULT: original_labels.txt")
    parser.add_argument("--predicted-labels-file", type=str, default='predicted_labels.txt',
                        help="File containing paths to predicted labels. DEFAULT: predicted_labels.txt")
    args = parser.parse_args()

    evaluation_result = evaluate_prediction_results(
        read_lines(args.original_labels_file),
        read_lines(args.predicted_labels_file)
    )

    export_evaluation_results(evaluation_result)
