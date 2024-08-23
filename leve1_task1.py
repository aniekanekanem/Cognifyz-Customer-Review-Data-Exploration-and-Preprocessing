# Data exploration and preprocessing of dataset for group of restaurant business performance

# importing libraries

import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns


# importing dataset
data = pd.read_csv('Cognifyz Dataset Internship Work file.csv')

# TASK 1
# 1.  DATA EXPLORATION AND PREPROCESSING

# i.  Viewing basic data information
data.head()
data.tail()
'''
dataset cointains special character '�' which can be replaced with an empty string
'''

data.info()
data.shape
'''
The above shows there are a total of 9551 rows of data entries and 21 columns
'''

# replacing '�' with '' and assigning it to new varaiable 'new_data' (optional)
new_data = data.replace('�', '', regex=True)

# confirming changes
new_data.tail()

# Clearing any trailing white space
def remove_space(x):
    if isinstance(x, str):
        return x.strip()
    else:
        return x
    
new_data = new_data.map(remove_space)

# ii. Checking for missing values
new_data.isnull().sum()
new_data.isna().sum()
'''
there are no missing values
'''

# 2.  Gathering summary statistics
new_data.describe() # summary stats for numerical variables
'''
- The minimum Aggregate rating is 0 while the maximum Aggregate rating is 4.9, and the average Aggregate rating is approximately 2.7.  However, the standard deviation on the Aggregate rating is approximately 1.52.
Going by this figurative investigation, this data shows the presence of both satisfied and disatisfied customers and if we are to take outliers into consideration, 1.52 Aggregate rating is a low performing business activity.
'''
new_data.describe(include=['object']) # summary stats for categorical variables
'''
- Cafe Coffee Day is the most active restaurant with 83 business activities.
- New Delhi is the most busiest city in business operations with over 5000 business activities
- There are 1825 unique cuisines with 'North Indian' being the most consumed cuisine of 936 consumptions/orders
- The most Rating text is 'Average' with over 3000 average customer ratings.
'''

# Confirming outliers by filtering rows where 'Aggregate rating' is greater than 3 using 3 as the threshold value

# using 3 as a theshold on the 'Aggregate rating' column
# new_data_outlierCheck = new_data[new_data['Aggregate rating'] > 3]

# checking for outliers using interquartile range (IQR)
'''
Outliers are the data above the upper fence and below the lower fence.
upper fence value = Q3 + (1.5 * IQR)
lower fence value = Q1 - (1.5 * IQR)
IQR = Q3 - Q1
Where:
Q1 = first quartile
Q3 = third quartile
IQR = inter quartile range
'''
Q1 = new_data['Aggregate rating'].quantile(0.25)
Q3 = new_data['Aggregate rating'].quantile(0.75)
Q2 = new_data['Aggregate rating'].quantile(0.5)
IQR = Q3 - Q1
IQR
Q1
Q2
Q3
upper_fence = Q3 + (1.5 * IQR)
lower_fence = Q1 - (1.5 * IQR)
upper_fence
lower_fence

# upper outlier check
new_data_upperOutlierCheck = new_data[new_data['Aggregate rating']>upper_fence]
new_data_upperOutlierCheck
'''
There are no outlier which are above the upper threshold of 5.5 Aggregate rating value
'''

# lower outlier check
new_data_lowerOutlierCheck = new_data[new_data['Aggregate rating']<lower_fence]
new_data_lowerOutlierCheck

# checking the number of lower outliers
new_data_lowerOutlierCheck.shape
'''
There are 2148 outliers which are below the lower threshold of 0.67 Aggregate rating
'''

# confirming the rows with lower outliers
new_data_lowerOutlierCheck
'''
There are 2148 lower outliers and 0 upper outlier in this dataset out of 9551 observations
'''

# confirming the outliers with a box plot visualization
# Create a box plot
plt.figure(figsize=(8, 6))
plt.boxplot(new_data['Aggregate rating'], vert=False)
plt.title('Box Plot of Aggregate Ratings')
plt.xlabel('Aggregate Rating')
plt.show()
'''
The upper fence and the lower fence are not indicated away from the lower whisker and the upper whisker of this box plot.
Another kind of box plot will be created to display them
'''

# Creating a box plot to indicate the location of upper fence and lower fence using vertical lines
sns.boxplot(x=new_data['Aggregate rating'])

# Optional: Add vertical lines for upper and lower fences
plt.axvline(x=upper_fence, color='r', linestyle='--', label='Upper Fence')
plt.axvline(x=lower_fence, color='b', linestyle='--', label='Lower Fence')

plt.legend()
plt.title('Box Plot with Custom Fences')
plt.xlabel('Aggregate rating')
plt.show()

# 3.  Analyzing distribution and identidying class imbalance on the 'Aggregate rating'
# Aggregate rating distribution and class imbalance using histogram
new_data['Aggregate rating'].hist()
plt.show()

'''
From the distribution we can see the outliers which are below 0.67 Aggregate rating.
This is indeed an average customer rating as decribed from the dataset descriptive statistics.
This distribution gives a clear indication of the high presence of customer disatisfactions with fewer high rate satisfactions
'''

# Aggregate rating distribution and class imbalance using more descriptive statistics
from scipy.stats import skew, kurtosis

# Descriptive statistics
mean = new_data['Aggregate rating'].mean()
median = new_data['Aggregate rating'].median()
std_dev = new_data['Aggregate rating'].std()
skewness = skew(new_data['Aggregate rating'])
kurt = kurtosis(new_data['Aggregate rating'])

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurt}")

'''
- Mean (2.7) vs. Median (3.2):
The mean (2.7) is lower than the median (3.2) which aligns with a negatively skewed distribution, where the mean is pulled in the direction of the tail, which in this case is towards the end.

- Skewness (-0.954):
The skewness which is negative indicates that the distribution has a longer or fatter tail on the left (lower ratings).  This display suggesta that there are more ratings on the lower end or that the lower ratings are more extreme compared to higher ratings.  It doesn't directly indicate the overall sentiment of negative customer review.

- Standard Deviation (1.5):
The significance of the standard deviation on the 'Aggregate rating' is such that there is amount of variability in the ratings. A standard deviation of 1.5 in the context of a rating scale of 1 to 5 suggests a moderate amount of variability in the ratings.  This is because a standard deviation of 1.5 means that most ratings are within 1.5 ratings of the average rating of 2.7 i.e 2.7 ± 1.5 (4.2 or 1.2) suggesting a moderate amount of variability in the ratings.

- Kurtosis (-0.58):
The Kurtosis measures the "tailedness" of the distribution.  A negative kurtosis value indicates a platykurtic distribution, meaning the distribution has lighter tails and a flatter peak compared to a normal distribution.  In other words, there are fewer extreme values (both very high and very low ratings) than would be expected in a normal distribution.

Conclusion: 

The Overall Sentiment: The negative skewness and the fact that the mean is lower than the median suggest a tendency toward more negative reviews. Although the ratings aren't extremely skewed, the general trend is toward the lower end of the scale.

Distribution shape: The negative kurtosis indicates that while there are fewer extreme values than a normal distribution, the negative skewness shows that there are still relatively more low ratings or that low ratings are more pronounced.

Hence the dataset appears to lean towards more negative customer reviews with some variability in the ratings.  The distribution has a flatter peak and lighter tails, indicating a less extreme range of reviews compared to a normal distribution.
'''
