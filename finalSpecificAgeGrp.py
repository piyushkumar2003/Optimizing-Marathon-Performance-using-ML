import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

def process_data(gender):
    # Define file paths and index values based on gender
    if gender.lower() == 'men':
        file_path = "/content/Mens Marathon Data for specific age(25-29).csv"
        starting_age = 27
        predicted_change_index = 331
    elif gender.lower() == 'women':
        file_path = "/content/Women Marathon Data for specific age(25-29).csv"
        starting_age = 25
        predicted_change_index = 0
    else:
        print("Invalid gender input. Please enter 'men' or 'women'.")
        return
    
    # Load the dataset
    df = pd.read_csv(file_path)

    # Ensure column names match your dataset's columns
    df.columns = ['City Name', 'Starting Age', 'Final Age', 'Starting Grade', 'Final Grade', 'Cost of Living']

    # Encode city names to be used in the linear model
    label_encoder = LabelEncoder()
    df['City Name'] = label_encoder.fit_transform(df['City Name'])

    # Fixed parameters
    final_age = 29

    # Calculate mode of Starting Grade
    mode_starting_rank = df['Starting Grade'].mode()[0]
    print("The starting grade considered is the mode of Starting Grades: ", mode_starting_rank)

    # Set up features and target for the model
    X = df[['City Name', 'Starting Age', 'Starting Grade', 'Final Age']]
    y = df['Final Grade']

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Create a DataFrame to store the results
    results = pd.DataFrame(columns=['City Name', 'Predicted Change', 'Final Grade', 'Cost of Living'])

    # Iterate over all unique city codes
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

    # Sort results by 'Predicted Change'
    results = results.sort_values(by='Predicted Change')

    # Filtering out the cities with max grade change
    dummy = results.head(1)
    mask = results['Predicted Change'].values == dummy['Predicted Change'][predicted_change_index]
    pos = np.flatnonzero(mask)
    result = results.iloc[pos]

    # Sorting the max grade change results by 'Cost of Living'
    result = result.sort_values(by='Cost of Living')

    # Print the sorted results
    if not result.empty:
        print("Cities with the maximum Predicted Grade Change:")
        for index, row in result.iterrows():
            print(f"{row['City Name']}: Predicted Change = {row['Predicted Change']}, Final Grade = {row['Final Grade']}, Cost of Living = {row['Cost of Living']}")
    else:
        print("No cities found.")

# Get user input
gender = input("Enter gender (men/women): ")
process_data(gender)
