import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set the style for the plots
plt.style.use('default')
sns.set_palette("husl")

# Create a unified dataset from the provided information
# First, let's process the enrollment data
enrollment_data = {
    'AY': ['2015-2016', '2015-2016', '2015-2016', '2016-2017', '2016-2017', '2016-2017',
           '2017-2018', '2017-2018', '2017-2018', '2018-2019', '2018-2019', '2018-2019',
           '2019-2020', '2019-2020', '2020-2021', '2020-2021', '2020-2021', '2021-2022',
           '2021-2022', '2021-2022', '2022-2023', '2022-2023', '2022-2023', '2023-2024',
           '2023-2024', '2024-2025', '2024-2025', '2024-2025'],
    'Semester': ['1st', '2nd', 'Midyear', '1st', '2nd', 'Midyear', '1st', '2nd', 'Midyear',
                 '1st', '2nd', 'Midyear', '1st', '2nd', '1st', '2nd', 'Midyear', '1st',
                 '2nd', 'Midyear', '1st', '2nd', 'Midyear', '1st', '2nd', 'Midyear', '1st', '2nd'],
    'Enrollees': [3290, 3096, 759, 2718, 2670, 718, 2466, 2360, 1014, 3541, 3220, 845,
                  3462, 3316, 3976, 3782, 527, 4944, 4726, 1823, 5560, 5080, 1996, 5892,
                  5398, 1574, 6223, 5952]
}

enrollment_df = pd.DataFrame(enrollment_data)

# Process the dropout data
dropout_data = {
    'AY': ['2015-2016', '2015-2016', '2015-2016', '2016-2017', '2016-2017', '2016-2017',
           '2017-2018', '2017-2018', '2017-2018', '2018-2019', '2018-2019', '2018-2019',
           '2019-2020', '2019-2020', '2019-2020', '2020-2021', '2020-2021', '2020-2021',
           '2021-2022', '2021-2022', '2021-2022', '2022-2023', '2022-2023', '2022-2023',
           '2023-2024', '2023-2024', '2023-2024', '2024-2025', '2024-2025', '2024-2025'],
    'Semester': ['First', 'Second', 'Summer', 'First', 'Second', 'Midyear', 'First', 'Second', 'Midyear',
                 'First', 'Second', 'Midyear', 'First', 'Second', 'Midyear', 'First', 'Second', 'Midyear',
                 'First', 'Second', 'Midyear', 'First', 'Second', 'Midyear', 'First', 'Second', 'Midyear',
                 'First', 'Second', 'Midyear'],
    'Dropouts': [56, 47, 11, 35, 29, 7, 57, 24, 5, 66, 39, 9, 62, 59, 0, 40, 9, 1, 39, 21, 15, 39, 40, 4, 86, 20, 17, 50, 23, 0]
}

dropout_df = pd.DataFrame(dropout_data)

# Standardize semester names
semester_mapping = {
    '1st': 'First',
    '2nd': 'Second',
    'Midyear': 'Midyear',
    'Summer': 'Midyear'  # Assuming Summer is equivalent to Midyear
}

enrollment_df['Semester'] = enrollment_df['Semester'].map(lambda x: semester_mapping.get(x, x))
dropout_df['Semester'] = dropout_df['Semester'].map(lambda x: semester_mapping.get(x, x))

# Merge enrollment and dropout data
merged_df = pd.merge(enrollment_df, dropout_df, on=['AY', 'Semester'], how='outer')

# Fill missing values with 0 (for cases where there might be no dropouts)
merged_df['Dropouts'] = merged_df['Dropouts'].fillna(0)
merged_df['Enrollees'] = merged_df['Enrollees'].fillna(0)

# Calculate non-dropouts
merged_df['Non_Dropouts'] = merged_df['Enrollees'] - merged_df['Dropouts']

# Calculate dropout rate - handle division by zero
def calculate_dropout_rate(row):
    if row['Enrollees'] > 0:
        return round(row['Dropouts'] / row['Enrollees'] * 100, 2)
    return 0

merged_df['Dropout_Rate'] = merged_df.apply(calculate_dropout_rate, axis=1)

# Extract year from AY for analysis
merged_df['Year'] = merged_df['AY'].apply(lambda x: int(x.split('-')[0]))

# Encode categorical variables for machine learning
le_semester = LabelEncoder()
merged_df['Semester_Encoded'] = le_semester.fit_transform(merged_df['Semester'])

# Create features and target for dropout prediction
# For this simplified example, we'll predict if dropout rate is above average
average_dropout_rate = merged_df['Dropout_Rate'].mean()
merged_df['High_Dropout'] = (merged_df['Dropout_Rate'] > average_dropout_rate).astype(int)

# Features: Year, Semester_Encoded, Enrollees
# Target: High_Dropout
X = merged_df[['Year', 'Semester_Encoded', 'Enrollees']]
y = merged_df['High_Dropout']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Handle case where only one class is present in y_test
if len(np.unique(y_test)) == 2:
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
else:
    # Only one class present, so probability is always 0 or 1
    y_pred_proba = np.zeros_like(y_pred, dtype=float)
    if len(y_test) > 0:
        y_pred_proba[y_pred == 1] = 0.99  # Set high probability for the single class

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) if len(y_test) > 0 else 0

# Handle precision, recall, f1 for cases with only one class
try:
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
except:
    precision, recall, f1 = 0, 0, 0

# Create the visualizations
fig = plt.figure(figsize=(20, 20))

# 1. KDE Plot
plt.subplot(4, 4, 1)
valid_dropout_rates = merged_df[merged_df['Dropout_Rate'].notna()]['Dropout_Rate']
if len(valid_dropout_rates) > 0:
    sns.kdeplot(valid_dropout_rates, fill=True)
    plt.title('KDE Plot of Dropout Rate')
    plt.xlabel('Dropout Rate (%)')
else:
    plt.text(0.5, 0.5, 'No valid dropout rate data', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=12)
    plt.title('KDE Plot of Dropout Rate')

# 2. Bar plot - compare dropouts to non-dropouts
plt.subplot(4, 4, 2)
dropout_sum = merged_df['Dropouts'].sum()
non_dropout_sum = merged_df['Non_Dropouts'].sum()
plt.bar(['Dropouts', 'Non-Dropouts'], [dropout_sum, non_dropout_sum], color=['red', 'green'])
plt.title('Total Dropouts vs Non-Dropouts')
plt.ylabel('Number of Students')

# 3. Correlation matrix
plt.subplot(4, 4, 3)
corr_matrix = merged_df[['Enrollees', 'Dropouts', 'Non_Dropouts', 'Dropout_Rate', 'Year']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')

# 4. Confusion matrix
plt.subplot(4, 4, 4)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 5. ROC curve
plt.subplot(4, 4, 5)
if len(np.unique(y_test)) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
else:
    roc_auc = 0.5
    plt.text(0.5, 0.5, 'Only one class present\nROC not defined', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=12)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# 6. Bar chart - Shows most influential predictors
plt.subplot(4, 4, 6)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importance (Dropout Prediction)')
plt.ylabel('Importance')

# 7. Scatter plot - Enrollment vs Dropout Rate (simulating attendance vs grade)
plt.subplot(4, 4, 7)
valid_data = merged_df[(merged_df['Enrollees'] > 0) & (merged_df['Dropout_Rate'].notna())]
plt.scatter(valid_data['Enrollees'], valid_data['Dropout_Rate'], alpha=0.7)
plt.xlabel('Number of Enrollees')
plt.ylabel('Dropout Rate (%)')
plt.title('Enrollment vs Dropout Rate')

# 8. Line plot - Dropout rate over time
plt.subplot(4, 4, 8)
yearly_dropout = merged_df.groupby('Year')['Dropout_Rate'].mean()
plt.plot(yearly_dropout.index, yearly_dropout.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Average Dropout Rate (%)')
plt.title('Dropout Rate Over Time')
plt.xticks(rotation=45)

# 9. Count of high vs low dropout semesters
plt.subplot(4, 4, 9)
merged_df['High_Dropout_Label'] = merged_df['High_Dropout'].map({0: 'Low Dropout', 1: 'High Dropout'})
sns.countplot(data=merged_df, x='High_Dropout_Label')
plt.title('Count of High vs Low Dropout Semesters')
plt.xlabel('Dropout Category')

# 10. Dropout rate by semester
plt.subplot(4, 4, 10)
semester_dropout = merged_df.groupby('Semester')['Dropout_Rate'].mean()
sns.barplot(x=semester_dropout.index, y=semester_dropout.values)
plt.title('Average Dropout Rate by Semester')
plt.xlabel('Semester')
plt.ylabel('Dropout Rate (%)')
plt.xticks(rotation=45)

# 11. Enrollment trend over years
plt.subplot(4, 4, 11)
yearly_enrollment = merged_df.groupby('Year')['Enrollees'].sum()
plt.plot(yearly_enrollment.index, yearly_enrollment.values, marker='o', color='green')
plt.xlabel('Year')
plt.ylabel('Total Enrollment')
plt.title('Enrollment Trend Over Years')
plt.xticks(rotation=45)

# 12. Dropout distribution by semester
plt.subplot(4, 4, 12)
semester_dropouts = merged_df.groupby('Semester')['Dropouts'].sum()
plt.pie(semester_dropouts.values, labels=semester_dropouts.index, autopct='%1.1f%%')
plt.title('Dropout Distribution by Semester')

plt.tight_layout()
plt.show()

# Print some insights
print("Data Analysis Insights:")
print(f"1. Average dropout rate: {merged_df['Dropout_Rate'].mean():.2f}%")
max_dropout_idx = merged_df['Dropout_Rate'].idxmax()
print(f"2. Highest dropout rate: {merged_df.loc[max_dropout_idx, 'Dropout_Rate']:.2f}% in {merged_df.loc[max_dropout_idx, 'AY']} {merged_df.loc[max_dropout_idx, 'Semester']}")
print(f"3. Total students: {merged_df['Enrollees'].sum():,}")
print(f"4. Total dropouts: {merged_df['Dropouts'].sum():,}")
overall_dropout_rate = (merged_df['Dropouts'].sum() / merged_df['Enrollees'].sum() * 100) if merged_df['Enrollees'].sum() > 0 else 0
print(f"5. Overall dropout rate: {overall_dropout_rate:.2f}%")

# Print model performance metrics
print("\nRandom Forest Classification Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print("\nFeature Importance (Classification):")
for feature, importance in feature_importances.sort_values(ascending=False).items():
    print(f"{feature}: {importance:.4f}")

# Print dataset statistics
print("\nDataset Statistics:")
print(f"Total records: {len(merged_df)}")
print(f"High dropout semesters: {merged_df['High_Dropout'].sum()}")
print(f"Low dropout semesters: {len(merged_df) - merged_df['High_Dropout'].sum()}")
print(f"Years covered: {merged_df['Year'].min()} to {merged_df['Year'].max()}")
print(f"Semesters: {', '.join(merged_df['Semester'].unique())}")