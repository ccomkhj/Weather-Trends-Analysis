# Weather Trends Analysis Tool

The Weather Trends Analysis Tool is a Python-based utility program designed to fetch, process, and compare weather data across different time periods using the Visual Crossing Weather API. It leverages the power of libraries such as Pandas, Numpy, and Matplotlib to analyze and visualize weather trends, making it an invaluable tool for researchers, meteorologists, and any individual interested in weather patterns.

## Features

- **Fetch Weather Data:** Utilize the Visual Crossing Weather API to download detailed weather data based on specified dates and geographical coordinates.
- **Data Processing:** Process and transform the fetched data into a structured format for analysis. This includes calculating weekly average weather statistics for more insightful comparisons.
- **Comparison and Visualization:** Compare weather trends across different time periods and visualize the data using line graphs for easy interpretation of changes and patterns in weather variables.

## Dependencies

To run the Weather Trends Analysis Tool, you need to have the following Python libraries installed:
- pandas
- requests
- numpy
- matplotlib

You can install these dependencies using pip:

```sh
pip install pandas requests numpy matplotlib
```

## Configuration

Before using this tool, you need to obtain an API key from the Visual Crossing Weather API and create a **config.json** file in the root folder with the following content:

```json
{
  "CROSS_KEY": "Your_Visual_Crossing_API_Key_Here"
}
```

Ensure that the **config.json** file is correctly set up, as the program reads the API key from this file.

## Usage

Here's how to get started with the Weather Trends Analysis Tool:

1. **Import the Tool:** Start by importing the main functions from the script where you need to perform the weather data analysis.

2. **Load Weather Data:**
   Use the `load_weather` function to fetch weather data. Provide the start and end dates, and the latitude and longitude for the location of interest.

   Example:
   ```py
   weather_data = load_weather("2022-01-01", "2022-01-31", 40.7128, -74.0060)
   ```

3. **Process Weather Data:**
   After loading the data, you can calculate the average weather metrics over specified weeks using the `average` function.

   Example:
   ```py
   average_weather_data = average(weather_data, weeks=4)
   ```

4. **Compare and Visualize:**
   Compare and visualize the differences between two sets of weather data, typically representing different times or periods, using the `compare` function.

   Example:
   ```py
   compare(average_weather_data_1, average_weather_data_2)
   ```

5. **Run Analysis:**
   To execute the full workflow of loading, processing, and comparing weather data, use the `run` function with the necessary parameters.

**Note:** The tool is designed for educational and research purposes, and its accuracy or suitability for professional meteorological analysis is not guaranteed.
