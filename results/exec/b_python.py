import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class DataProcessor:
    """
    A class to process employee data from a CSV file.

    Features:
    - Loads CSV data with type hints and error handling.
    - Filters employees hired after 2020.
    - Calculates average salary by department.
    - Identifies the top 3 highest paid employees.
    - Exports results to a JSON file.
    """

    def __init__(self, csv_path: str, json_path: str):
        """
        Initialize the DataProcessor with input and output file paths.

        Args:
            csv_path (str): Path to the input CSV file.
            json_path (str): Path to the output JSON file.
        """
        self.csv_path = Path(csv_path)
        self.json_path = Path(json_path)

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load employee records from the CSV file.

        Returns:
            List[Dict[str, Any]]: List of dictionaries representing employee records.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file is malformed or missing required columns.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        required_columns = {"name", "department", "salary", "hire_date"}
        employees: List[Dict[str, Any]] = []

        with self.csv_path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames is None:
                raise ValueError("CSV file is empty or has no headers.")
            if not required_columns.issubset(set(reader.fieldnames)):
                missing = required_columns - set(reader.fieldnames)
                raise ValueError(f"Missing required columns: {missing}")

            for row_number, row in enumerate(reader, start=2):  # start=2 to account for header
                try:
                    # Convert salary to float
                    row["salary"] = float(row["salary"])
                    # Parse hire_date to datetime
                    row["hire_date"] = datetime.strptime(row["hire_date"], "%Y-%m-%d")
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Malformed data at row {row_number}: {e}")

                employees.append(row)

        return employees

    def filter_by_hire_year(self, employees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter employees who were hired after 2020.

        Args:
            employees (List[Dict[str, Any]]): List of employee records.

        Returns:
            List[Dict[str, Any]]: Filtered list of employees.
        """
        return [emp for emp in employees if emp["hire_date"].year > 2020]

    def calculate_average_salary_by_department(self, employees: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate the average salary for each department.

        Args:
            employees (List[Dict[str, Any]]): List of employee records.

        Returns:
            Dict[str, float]: Dictionary mapping department names to average salary.
        """
        department_salaries: Dict[str, List[float]] = {}

        for emp in employees:
            dept = emp["department"]
            salary = emp["salary"]
            if dept not in department_salaries:
                department_salaries[dept] = []
            department_salaries[dept].append(salary)

        averages: Dict[str, float] = {}
        for dept, salaries in department_salaries.items():
            averages[dept] = sum(salaries) / len(salaries)

        return averages

    def find_top_3_highest_paid(self, employees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find the top 3 highest paid employees.

        Args:
            employees (List[Dict[str, Any]]): List of employee records.

        Returns:
            List[Dict[str, Any]]: List of the top 3 employees sorted by salary descending.
        """
        sorted_employees = sorted(employees, key=lambda x: x["salary"], reverse=True)
        return sorted_employees[:3]

    def export_to_json(self, results: Dict[str, Any]) -> None:
        """
        Export the processed results to a JSON file.

        Args:
            results (Dict[str, Any]): Dictionary containing processed data.

        Raises:
            IOError: If the JSON file cannot be written.
        """
        try:
            with self.json_path.open("w", encoding="utf-8") as jsonfile:
                json.dump(results, jsonfile, indent=4)
        except IOError as e:
            raise IOError(f"Failed to write JSON file: {e}")

    def process(self) -> Dict[str, Any]:
        """
        Execute the full processing pipeline.

        Returns:
            Dict[str, Any]: Complete results dictionary.
        """
        try:
            employees = self.load_data()
            filtered_employees = self.filter_by_hire_year(employees)
            avg_salaries = self.calculate_average_salary_by_department(filtered_employees)
            top_employees = self.find_top_3_highest_paid(filtered_employees)

            results = {
                "total_employees": len(filtered_employees),
                "average_salary_by_department": avg_salaries,
                "top_3_highest_paid": [
                    {
                        "name": emp["name"],
                        "department": emp["department"],
                        "salary": emp["salary"],
                        "hire_date": emp["hire_date"].strftime("%Y-%m-%d")
                    }
                    for emp in top_employees
                ]
            }

            self.export_to_json(results)
            return results

        except Exception as e:
            raise RuntimeError(f"Data processing failed: {e}")


# Example usage
if __name__ == "__main__":
    # Create a sample CSV file for demonstration
    sample_csv = Path("employees_sample.csv")
    sample_csv.write_text("name,department,salary,hire_date\n"
                         "Alice,HR,75000,2021-03-15\n"
                         "Bob,Engineering,90000,2022-07-01\n"
                         "Charlie,Engineering,85000,2021-09-10\n"
                         "Diana,HR,80000,2020-01-01\n"
                         "Eve,Engineering,95000,2023-01-20\n", encoding="utf-8")

    try:
        processor = DataProcessor(csv_path="employees_sample.csv", json_path="employee_results.json")
        results = processor.process()
        print("Processing complete. Results:")
        print(json.dumps(results, indent=4))
    except Exception as e:
        print(f"Error: {e}")
