import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveEnrollmentVisualizer:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        self.csv_files = []
        self.processed_data = None
        
    def find_csv_files(self):
        """Find all CSV files in the workspace directory"""
        pattern = os.path.join(self.workspace_dir, '**', '*.csv')
        self.csv_files = glob.glob(pattern, recursive=True)
        print(f"Found {len(self.csv_files)} CSV files")
        return self.csv_files
    
    def extract_program_level_data(self, df):
        """Extract enrollment data at the program level"""
        # Remove rows with all NaN values and totals
        df = df.dropna(how='all')
        df = df[df['PROGRAM NAME'].notna() & 
                (df['PROGRAM NAME'] != '') & 
                (~df['PROGRAM NAME'].str.contains('TOTAL', case=False, na=False))]
        
        # Fill NaN values
        df['MAJOR'] = df['MAJOR'].fillna('No Major')
        
        # Convert numeric columns
        numeric_columns = []
        for col in df.columns:
            if col not in ['PROGRAM NAME', 'MAJOR', 'ENROLLMENT']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numeric_columns.append(col)
                except:
                    pass
        
        program_data = []
        
        # Group by program and major
        for (program, major), group in df.groupby(['PROGRAM NAME', 'MAJOR']):
            if len(group) == 0:
                continue
                
            program_info = {
                'program_name': program,
                'major': major
            }
            
            if numeric_columns:
                enrollment_data = group[numeric_columns].sum()
                
                # Extract year-wise enrollment
                years = ['FIST YEAR', 'SECOND YEAR', 'THIRD YEAR', 'FOURTH YEAR', 'FIFTH YEAR', 'SIXTH YEAR']
                for year in years:
                    year_cols = [col for col in numeric_columns if year in col]
                    if year_cols:
                        program_info[f'{year.lower().replace(" ", "_")}_total'] = enrollment_data[year_cols].sum()
                        
                        # Male and female counts
                        male_cols = [col for col in year_cols if 'Male' in col]
                        female_cols = [col for col in year_cols if 'Female' in col]
                        
                        program_info[f'{year.lower().replace(" ", "_")}_male'] = enrollment_data[male_cols].sum() if male_cols else 0
                        program_info[f'{year.lower().replace(" ", "_")}_female'] = enrollment_data[female_cols].sum() if female_cols else 0
                
                # Calculate program-level metrics
                total_male = sum(program_info.get(f'{year.lower().replace(" ", "_")}_male', 0) for year in years)
                total_female = sum(program_info.get(f'{year.lower().replace(" ", "_")}_female', 0) for year in years)
                
                program_info['total_male'] = total_male
                program_info['total_female'] = total_female
                program_info['total_enrollment'] = total_male + total_female
                program_info['gender_ratio'] = total_male / total_female if total_female > 0 else 0
                program_info['female_percentage'] = (total_female / (total_male + total_female)) * 100 if (total_male + total_female) > 0 else 0
                
                # Year progression rates (for dropout analysis)
                if 'first_year_total' in program_info and 'second_year_total' in program_info:
                    program_info['progression_rate_1to2'] = program_info['second_year_total'] / program_info['first_year_total'] if program_info['first_year_total'] > 0 else 0
                    # Create dropout indicator (if progression rate < 0.8, consider as dropout)
                    program_info['dropout_indicator'] = 1 if program_info['progression_rate_1to2'] < 0.8 else 0
                if 'second_year_total' in program_info and 'third_year_total' in program_info:
                    program_info['progression_rate_2to3'] = program_info['third_year_total'] / program_info['second_year_total'] if program_info['second_year_total'] > 0 else 0
                if 'third_year_total' in program_info and 'fourth_year_total' in program_info:
                    program_info['progression_rate_3to4'] = program_info['fourth_year_total'] / program_info['third_year_total'] if program_info['third_year_total'] > 0 else 0
            
            program_data.append(program_info)
        
        return program_data
    
    def process_csv_batch(self, csv_files, batch_size=5):
        """Process CSV files in batches"""
        all_program_data = []
        
        for i in range(0, len(csv_files), batch_size):
            batch = csv_files[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}: {len(batch)} files")
            
            for csv_file in batch:
                try:
                    # Extract year and semester from filename
                    filename = os.path.basename(csv_file)
                    parts = filename.split('_')
                    
                    if len(parts) >= 4:
                        year_range = f"{parts[2]}-{parts[3]}"
                        semester = parts[4].split('.')[0] if len(parts) > 4 else 'Unknown'
                    else:
                        year_range = 'Unknown'
                        semester = 'Unknown'
                    
                    # Read CSV file
                    df = pd.read_csv(csv_file)
                    
                    # Extract program-level data
                    program_data = self.extract_program_level_data(df)
                    
                    # Add metadata
                    for program in program_data:
                        program['year_range'] = year_range
                        program['semester'] = semester
                        program['filename'] = filename
                    
                    all_program_data.extend(program_data)
                    print(f"  ✓ Processed: {filename} ({len(program_data)} programs)")
                    
                except Exception as e:
                    print(f"  ✗ Error processing {csv_file}: {str(e)}")
        
        return pd.DataFrame(all_program_data)
    
    def prepare_data_for_analysis(self, df):
        """Prepare data for machine learning analysis"""
        # Encode categorical variables
        categorical_columns = ['program_name', 'major', 'year_range', 'semester']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        # Select numeric features for modeling
        exclude_columns = ['program_name', 'major', 'year_range', 'semester', 'filename']
        feature_columns = [col for col in df.columns if col not in exclude_columns and 
                          df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_columns].fillna(0)
        
        return X, df
    
    def create_kde_plots(self, data):
        """Create KDE plots for key variables"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('KDE Plots - Enrollment Data Distribution', fontsize=16, fontweight='bold')
        
        # KDE plot for total enrollment
        if 'total_enrollment' in data.columns:
            sns.kdeplot(data=data, x='total_enrollment', ax=axes[0,0], fill=True, color='skyblue')
            axes[0,0].set_title('Distribution of Total Enrollment', fontweight='bold')
            axes[0,0].set_xlabel('Total Enrollment')
            axes[0,0].set_ylabel('Density')
        
        # KDE plot for gender ratio
        if 'gender_ratio' in data.columns:
            sns.kdeplot(data=data, x='gender_ratio', ax=axes[0,1], fill=True, color='lightcoral')
            axes[0,1].set_title('Distribution of Gender Ratio', fontweight='bold')
            axes[0,1].set_xlabel('Gender Ratio (Male/Female)')
            axes[0,1].set_ylabel('Density')
        
        # KDE plot for female percentage
        if 'female_percentage' in data.columns:
            sns.kdeplot(data=data, x='female_percentage', ax=axes[1,0], fill=True, color='lightgreen')
            axes[1,0].set_title('Distribution of Female Percentage', fontweight='bold')
            axes[1,0].set_xlabel('Female Percentage (%)')
            axes[1,0].set_ylabel('Density')
        
        # KDE plot for progression rate
        if 'progression_rate_1to2' in data.columns:
            sns.kdeplot(data=data, x='progression_rate_1to2', ax=axes[1,1], fill=True, color='gold')
            axes[1,1].set_title('Distribution of Progression Rate (1st to 2nd Year)', fontweight='bold')
            axes[1,1].set_xlabel('Progression Rate')
            axes[1,1].set_ylabel('Density')
        
        plt.tight_layout()
        plt.savefig('kde_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dropout_analysis(self, data):
        """Create dropout analysis visualizations"""
        if 'dropout_indicator' not in data.columns:
            print("Dropout indicator not found in data")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Dropout Analysis', fontsize=16, fontweight='bold')
        
        # Count plot for dropout vs non-dropout
        dropout_counts = data['dropout_indicator'].value_counts()
        colors = ['#2E8B57', '#DC143C']  # Green for non-dropout, Red for dropout
        bars = axes[0].bar(['Non-Dropout', 'Dropout'], dropout_counts.values, color=colors)
        axes[0].set_title('Dropout vs Non-Dropout Count', fontweight='bold')
        axes[0].set_ylabel('Count')
        
        # Add count labels on bars
        for i, v in enumerate(dropout_counts.values):
            axes[0].text(i, v + 0.01 * max(dropout_counts.values), str(v), ha='center', va='bottom', fontweight='bold')
        
        # Dropout rate by program
        if 'program_name' in data.columns:
            dropout_by_program = data.groupby('program_name')['dropout_indicator'].mean().sort_values(ascending=False)
            top_programs = dropout_by_program.head(10)
            
            bars = axes[1].barh(range(len(top_programs)), top_programs.values, color='#FF6B6B')
            axes[1].set_yticks(range(len(top_programs)))
            axes[1].set_yticklabels(top_programs.index, fontsize=8)
            axes[1].set_title('Dropout Rate by Program (Top 10)', fontweight='bold')
            axes[1].set_xlabel('Dropout Rate')
            
            # Add percentage labels
            for i, v in enumerate(top_programs.values):
                axes[1].text(v + 0.01, i, f'{v:.1%}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('dropout_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_correlation_matrix(self, data):
        """Create correlation matrix heatmap"""
        # Select numeric columns for correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_data = data[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        plt.title('Correlation Matrix - Enrollment Data', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_confusion_matrix_and_roc(self, X, y):
        """Create confusion matrix and ROC curve for dropout prediction"""
        # Prepare data for classification
        X_clean = X.dropna()
        y_clean = y[X_clean.index]
        
        # Remove rows where target is NaN
        valid_indices = y_clean.dropna().index
        X_valid = X_clean.loc[valid_indices]
        y_valid = y_clean.loc[valid_indices]
        
        if len(y_valid) < 10:
            print("Insufficient data for classification analysis")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)
        
        # Train Random Forest Classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_classifier.predict(X_test)
        y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]
        
        # Create confusion matrix and ROC curve
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Dropout Prediction Performance', fontsize=16, fontweight='bold')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix', fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_xticklabels(['Non-Dropout', 'Dropout'])
        axes[0].set_yticklabels(['Non-Dropout', 'Dropout'])
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve', fontweight='bold')
        axes[1].legend(loc="lower right")
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_roc.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return rf_classifier
    
    def create_feature_importance_plot(self, model, feature_names):
        """Create feature importance plot"""
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features for Dropout Prediction', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(importance + 0.001, i, f'{importance:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_academic_success_plots(self, data):
        """Create academic success related plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Academic Success Analysis', fontsize=16, fontweight='bold')
        
        # Scatter plot: First year enrollment vs progression rate
        if 'first_year_total' in data.columns and 'progression_rate_1to2' in data.columns:
            axes[0,0].scatter(data['first_year_total'], data['progression_rate_1to2'], alpha=0.6, color='blue')
            axes[0,0].set_xlabel('First Year Enrollment')
            axes[0,0].set_ylabel('Progression Rate (1st to 2nd Year)')
            axes[0,0].set_title('Enrollment vs Progression Rate', fontweight='bold')
            axes[0,0].grid(True, alpha=0.3)
        
        # Line plot: Progression rates across years
        progression_cols = [col for col in data.columns if 'progression_rate' in col]
        if progression_cols:
            progression_data = data[progression_cols].mean()
            years = ['1st to 2nd', '2nd to 3rd', '3rd to 4th']
            axes[0,1].plot(years, progression_data.values, marker='o', linewidth=2, markersize=8, color='red')
            axes[0,1].set_xlabel('Year Progression')
            axes[0,1].set_ylabel('Average Progression Rate')
            axes[0,1].set_title('Progression Rates Over Time', fontweight='bold')
            axes[0,1].grid(True, alpha=0.3)
        
        # Bar plot: Gender distribution by program type
        if 'program_name' in data.columns and 'female_percentage' in data.columns:
            program_gender = data.groupby('program_name')['female_percentage'].mean().sort_values(ascending=False)
            top_programs = program_gender.head(10)
            axes[1,0].barh(range(len(top_programs)), top_programs.values, color='#FF69B4')
            axes[1,0].set_yticks(range(len(top_programs)))
            axes[1,0].set_yticklabels(top_programs.index, fontsize=8)
            axes[1,0].set_xlabel('Female Percentage (%)')
            axes[1,0].set_title('Female Percentage by Program (Top 10)', fontweight='bold')
        
        # Histogram: Distribution of total enrollment
        if 'total_enrollment' in data.columns:
            axes[1,1].hist(data['total_enrollment'], bins=20, alpha=0.7, color='#4682B4', edgecolor='black')
            axes[1,1].set_xlabel('Total Enrollment')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('Distribution of Total Enrollment', fontweight='bold')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('academic_success_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_visualization_analysis(self, batch_size=5):
        """Run the complete visualization analysis"""
        print("Comprehensive Enrollment Data Visualization Analysis")
        print("=" * 60)
        
        # Find and process CSV files
        self.find_csv_files()
        if not self.csv_files:
            print("No CSV files found!")
            return
        
        print(f"\nProcessing {len(self.csv_files)} CSV files...")
        self.processed_data = self.process_csv_batch(self.csv_files, batch_size)
        
        print(f"\nProcessed {len(self.processed_data)} program-level data points")
        
        # Check if dropout indicator was created
        if 'dropout_indicator' in self.processed_data.columns:
            dropout_count = self.processed_data['dropout_indicator'].value_counts()
            print(f"Dropout analysis: {dropout_count[1]} dropouts, {dropout_count[0]} non-dropouts")
        else:
            print("Warning: Dropout indicator not created")
        
        # Prepare data for analysis
        X, data = self.prepare_data_for_analysis(self.processed_data)
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        # 1. KDE Plots
        print("Creating KDE plots...")
        self.create_kde_plots(data)
        
        # 2. Dropout Analysis
        print("Creating dropout analysis...")
        self.create_dropout_analysis(data)
        
        # 3. Correlation Matrix
        print("Creating correlation matrix...")
        self.create_correlation_matrix(data)
        
        # 4. Confusion Matrix and ROC Curve
        if 'dropout_indicator' in data.columns:
            print("Creating confusion matrix and ROC curve...")
            model = self.create_confusion_matrix_and_roc(X, data['dropout_indicator'])
            
            # 5. Feature Importance Plot
            if model is not None:
                print("Creating feature importance plot...")
                self.create_feature_importance_plot(model, X.columns)
        
        # 6. Academic Success Plots
        print("Creating academic success plots...")
        self.create_academic_success_plots(data)
        
        print("\nAll visualizations completed and saved!")
        print("\nGenerated files:")
        print("- kde_plots.png")
        print("- dropout_analysis.png")
        print("- correlation_matrix.png")
        print("- confusion_matrix_roc.png")
        print("- feature_importance.png")
        print("- academic_success_analysis.png")

def main():
    # Initialize visualizer
    workspace_dir = os.getcwd()
    visualizer = ComprehensiveEnrollmentVisualizer(workspace_dir)
    
    # Run visualization analysis
    visualizer.run_visualization_analysis(batch_size=5)

if __name__ == "__main__":
    main() 