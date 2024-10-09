
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Function to read CSV file based on gender selection
def read_csv_based_on_gender(gender):
    if gender.lower() == 'male':
        return pd.read_csv("/content/Men Data for Marathon Model-1.csv")
    elif gender.lower() == 'female':
        return pd.read_csv("/content/test_data_ip - Women's Data for Marathon Model-3.csv")
    else:
        print("Invalid gender choice. Using default data.")
        return pd.read_csv("/content/default_data.csv")

# Function to predict grade change for a city
def predict_grade_change_in_a_city(player_age, starting_rank, final_age, city):
    city_encoded = label_encoder.transform(city)
    new_data = pd.DataFrame({'City Name': city_encoded,
                             'Starting Age': [player_age],
                             'Starting Grade': [starting_rank],
                             'Final Age': [final_age]})
    final_grade = model.predict(new_data)
    print(f'Final Grade: {final_grade[0]}')
    print(f'Predicted Change: {final_grade[0] - starting_rank}')


# Function to predict grade change for given cities and retrieve cost of living from dataset
def predict_grade_change_with_cost_of_living(player_age, starting_rank, final_age, cities, current_city_of_living):
    cost_of_living = []
    final_grade = []
    cost_of_living.append(df.loc[df['City Name'] == label_encoder.transform([current_city_of_living.strip()])[0], 'Cost of Living'].iloc[0])
    for city in cities:
        city_encoded = label_encoder.transform([city.strip()])
        cost_of_living.append(df.loc[df['City Name'] == city_encoded[0], 'Cost of Living'].iloc[0])
        new_data = pd.DataFrame({'City Name': city_encoded,
                                 'Starting Age': [player_age],
                                 'Starting Grade': [starting_rank],
                                 'Final Age': [final_age]})
        predicted_change = model.predict(new_data)
        final_grade.append(predicted_change)

    for i, city in enumerate(cities):
        print(f'Final Grade for {city.strip()}: {final_grade[i]}')
        print(f'Predicted Change for {city.strip()}: {final_grade[i] - starting_rank}')
    print(f'Cost of Living for selected cities: {cost_of_living}')

    # Calculate percentage difference
    percentage_difference = [(col - cost_of_living[0]) / cost_of_living[0] * 100 for col in cost_of_living[1:]]
    print(f'Percentage difference in cost of living compared to {current_city_of_living}: {percentage_difference}')

# Function to predict cities with maximum grade change and minimum cost of living
def predict_city_for_max_grade_change_with_min_cost():
    starting_age = int(input("Enter the starting age: "))
    final_age = int(input("Enter the final age: "))
    mode_starting_rank = int(input("Enter the mode starting rank: "))

    results = pd.DataFrame(columns=['City Name', 'Predicted Change', 'Final Grade', 'Cost of Living'])

    for city_code in df['City Name'].unique():
        new_data = pd.DataFrame({
            'City Name': [city_code],
            'Starting Age': [starting_age],
            'Starting Grade': [mode_starting_rank],
            'Final Age': [final_age]
        })
        predicted_grade = model.predict(new_data)
        predicted_final_grade = predicted_grade[0].round(3)
        actual_change = (predicted_final_grade - mode_starting_rank).round(3)

        city_name = label_encoder.inverse_transform([city_code])[0]
        cost_of_living = df[df['City Name'] == city_code]['Cost of Living'].iloc[0]
        new_row = pd.DataFrame({'City Name': [city_name],
                                'Predicted Change': [actual_change],
                                'Final Grade': [predicted_final_grade],
                                'Cost of Living': [cost_of_living]})
        results = pd.concat([results, new_row], ignore_index=True)

    results = results.sort_values(by='Predicted Change', ascending=False)

    dummy = results.iloc[0]
    mask = results['Predicted Change'] == dummy['Predicted Change']
    result = results[mask]

    result = result.sort_values(by='Cost of Living')

    if not result.empty:
        print("Cities with the maximum Predicted Grade Change:")
        for index, row in result.iterrows():
            print(f"{row['City Name']}: Predicted Change = {row['Predicted Change']}, Final Grade = {row['Final Grade']}, Cost of Living = {row['Cost of Living']}")
    else:
        print("No cities found.")

def menu():
    while True:
        print("\nMenu:")
        print("1. Predict the grade change in a particular city")
        print("2. Compare the grade change among different cities with cost of living")
        print("3. Predict the city for a specific grade change")
        print("4. Predict cities with maximum grade change and minimum cost of living")
        print("5. Exit")

        choice = input("Enter your choice: ")
        if choice == '1':
            player_age = int(input("Enter player's age: "))
            starting_rank = int(input("Enter starting grade: "))
            final_age = int(input("Enter final age: "))
            city = input("Enter city name: ").split('/')
            predict_grade_change_in_a_city(player_age, starting_rank, final_age, city)
        elif choice == '2':
            player_age = int(input("Enter player's age: "))
            starting_rank = int(input("Enter starting grade: "))
            final_age = int(input("Enter final age: "))
            cities = input("Enter city names (slash-separated): ").split('/')
            current_city_of_living = input("Enter current city of living: ")
            predict_grade_change_with_cost_of_living(player_age, starting_rank, final_age, cities, current_city_of_living)
        elif choice == '3':
            player_age = int(input("Enter player's age: "))
            starting_rank = int(input("Enter starting grade: "))
            final_age = int(input("Enter final age: "))
            grade_change_threshold = int(input("Enter desired grade change: "))

            results = pd.DataFrame(columns=['City Name', 'Predicted Change', 'Cost of Living'])

            cities = df['City Name'].unique()
            for city in cities:
                city_encoded = [city]
                new_data = pd.DataFrame({'City Name': city_encoded,
                                        'Starting Age': [player_age],
                                        'Starting Grade': [starting_rank],
                                        'Final Age': [final_age]})
                predicted_change = model.predict(new_data)

                if predicted_change[0] >= grade_change_threshold:
                    city_name = label_encoder.inverse_transform(city_encoded)[0]
                    cost_of_living = df.loc[df['City Name'] == city, 'Cost of Living'].iloc[0]
                    new_row = pd.DataFrame({'City Name': [city_name],
                                            'Predicted Change': [predicted_change[0]],
                                            'Cost of Living': [cost_of_living]})
                    results = pd.concat([results, new_row], ignore_index=True)

            results = results.sort_values(by='Cost of Living')

            for index, row in results.iterrows():
                print(f"{row['City Name']}: Predicted Change = {row['Predicted Change']}, Cost of Living = {row['Cost of Living']}")

        elif choice == '4':
            predict_city_for_max_grade_change_with_min_cost()

        elif choice == '5':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

# Read CSV file based on user's gender selection
gender = input("Enter your gender (male/female): ")
df = read_csv_based_on_gender(gender)

# Convert City Name to numerical values using Label Encoding
label_encoder = LabelEncoder()
df['City Name'] = label_encoder.fit_transform(df['City Name'])

# Features (X) and target variable (y)
X = df[['City Name', 'Starting Age', 'Starting Grade', 'Final Age']]
y = df['Final Grade']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Call the menu function to start the program
menu()
