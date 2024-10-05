from datetime import datetime
from math import sin, pi

PHYSICAL_CYCLE = 23
EMOTIONAL_CYCLE = 28
INTELLECTUAL_CYCLE = 33

biorhythms = [
        ("Physical", PHYSICAL_CYCLE),
        ("Emotional", EMOTIONAL_CYCLE),
        ("Intellectual", INTELLECTUAL_CYCLE),
    ]

def calculate_days_lived(year, month, day):
    birth_date = datetime(year, month, day)
    today = datetime.now()
    return (today - birth_date).days


def calculate_biorhythm(t, cycle_length):
    return sin((2 * pi / cycle_length) * t)


def interpret_biorhythm(value, biorhythm_type):
    if value > 0.5:
        return f"{biorhythm_type}: High ({value:.2f}) - Lucky bastard!"
    elif value < -0.5:
        return f"{biorhythm_type}: Low ({value:.2f}) - Don't worry, this is just a phase!"
    else:
        return f"{biorhythm_type}: Neutral ({value:.2f})"

def get_user_input():
    """Get user input for name and birthdate, with error handling."""
    name = input("Enter your name: ")
    while True:
        try:
            year = int(input("Enter your year of birth (YYYY): "))
            month = int(input("Enter your month of birth (MM): "))
            day = int(input("Enter your day of birth (DD): "))
            # Validate the date by creating a datetime object
            datetime(year, month, day)
            break
        except ValueError:
            print("Invalid date. Please try again.")
    return name, year, month, day

def main():
    name, year, month, day = get_user_input()

    # Calculate days lived
    days_lived = calculate_days_lived(year, month, day)

    # Calculate biorhythms
    physical_biorhythm = calculate_biorhythm(days_lived, 23)
    emotional_biorhythm = calculate_biorhythm(days_lived, 28)
    intellectual_biorhythm = calculate_biorhythm(days_lived, 33)

    # Greet user and display results
    print(f"Hello, {name}! You have lived for {days_lived} days.")
    print("Your biorhythms are as follows:")

    # Calculate and display biorhythms
    for biorhythm_type, cycle_length in biorhythms:
        biorhythm_value = calculate_biorhythm(days_lived, cycle_length)
        print(interpret_biorhythm(biorhythm_value, biorhythm_type))


if __name__ == "__main__":
    main()

# Work time: 20 minutes
