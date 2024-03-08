# Predicting-Housing-Prices-in-Philadelphia
For this project, I was tasked with developing a model to predict the selling price of houses in Philadelphia. This project involved various stages including data collection, preprocessing, feature engineering, model development, and evaluation.
# Data Collection
https://www.phila.gov/property/data/
I used the above website to download data on properties in Philadelphia as a .csv file. Then I went through the attributes and decided which ones could impact selling prices. I ultimately decided on zip code, total area, number of bathrooms, number of bedrooms, number of stories, interior and exterior condition, sale date, sale price, and market value. 
# Data Preprocessing
In order to clean and prepare the data for analysis, I had to handle missing values and infinite values, then impute these values using the median so it would be in a suitable format for modeling. 
# Feature Engineering
I created new features like bath_bed_ratio and combined_condition to combine existing information so that it may be a better indicator of the house’s market value. 
# Model Development
The machine learning algorithms I employed were Linear Regression, Decision Tree, Random Forest, and Gradient Boosting. I chose market_value as my target value as I felt this better showed each properties worth today, rather than comparing selling prices from different dates.
Linear Regression models the relationship between features and market value.
Decision Tree and Random Forest create a set of rules that can be used to make predictions. Random Forest, an ensemble of Decision Tree, averages multiple trees’ predictions, improving the model’s accuracy.
Gradient Boosting further refines this approach by correctly errors from the previous predictions to enhance the model’s accuracy.
# Evaluation
I ended up choosing Gradient Boosting as it offered a better balance between performance and efficiency. Random Forest actually had the best performance of the four as it had the highest r^2 value and the lowest MSE. However, it took a very long time. So I decided to go with the next best, which was Gradient Boosting. Decision Tree was comparable to Gradient Boosting, but ultimately Gradient Boosting was better. Linear Regression was the least effective with the lowest r^2 value and the highest MSE.
# Reflection
My biggest challenge was data preprocessing and feature engineering. I had trouble deciding what data to use and what to drop. Some features that I thought would impact market value, such as having a garage, heat, A/C, had many missing values so I had to drop them. Many of the number of bedrooms, bathrooms, stories had zero, which I don’t think is accurate. For future research, I think it would be interesting to see how the average sale price compares to current market values.
