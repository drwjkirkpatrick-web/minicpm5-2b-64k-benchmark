from typing import List, Dict, Any

def process_grades(grades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process a list of student grade dictionaries.

    Args:
        grades (List[Dict[str, Any]]): A list where each element is a dict containing
            'name' (str) and 'score' (float/int).

    Returns:
        Dict[str, Any]: A summary dictionary with:
            - 'avg_score': The average of all scores.
            - 'top_student': The name of the student with the highest score.
            - 'bottom_student': The name of the student with the lowest score.

    Raises:
        ValueError: If the input list is empty or any dict is missing required keys.
    """
    if not grades:
        raise ValueError("The grades list cannot be empty.")

    total_score = 0
    min_score = None
    max_score = None
    top_student = None
    bottom_student = None

    for entry in grades:
        if 'name' not in entry or 'score' not in entry:
            raise ValueError(f"Missing required keys in entry: {entry}")

        name = entry['name']
        score = entry['score']

        total_score += score

        if min_score is None or score < min_score:
            min_score = score
            bottom_student = name

        if max_score is None or score > max_score:
            max_score = score
            top_student = name

    avg_score = total_score / len(grades)

    return {
        'avg_score': avg_score,
        'top_student': top_student,
        'bottom_student': bottom_student
    }


# Test example
if __name__ == "__main__":
    sample_grades = [
        {"name": "Alice", "score": 85},
        {"name": "Bob", "score": 90},
        {"name": "Charlie", "score": 78},
        {"name": "Diana", "score": 92},
    ]

    summary = process_grades(sample_grades)
    print(summary)
    # Expected output:
    # {'avg_score': 86.0, 'top_student': 'Diana', 'bottom_student': 'Charlie'}
