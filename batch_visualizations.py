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

class BatchEnrollmentVisualizer:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        self.csv_files = []
        self.batch_data = {}
        
    def find_csv_files(self):
        """Find all CSV files in the workspace directory"""
        pattern = os.path.join(self.workspace_dir, '**', '*.csv')
        self.csv_files = glob.glob(pattern, recursive=True)
        print(f"Found {len(self.csv_files)} CSV files")
        return self.csv_files
    
    def extract_enrollment_data(self, df):
        """Extract enrollment data with proper column mapping"""
        print(f"Original dataframe shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Remove rows with all NaN values and totals
        df = df.dropna(how='all')
        df = df[df['PROGRAM NAME'].notna() & 
                (df['PROGRAM NAME'] != '') & 
                (~df['PROGRAM NAME'].str.contains('TOTAL', case=False, na=False))]
        
        print(f"After filtering: {df.shape}")
        
        # Fill NaN values
        df['MAJOR'] = df['MAJOR'].fillna('No Major')
        
        # Look for actual enrollment data columns (not just Unnamed)
        enrollment_columns = []
        for col in df.columns:
            if col not in ['PROGRAM NAME', 'MAJOR', 'ENROLLMENT']:
                # Check if column contains numeric data
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    if numeric_data.notna().sum() > 0:  # Has some numeric data
                        enrollment_columns.append(col)
                        df[col] = numeric_data
                except:
                    pass
        
        print(f"Enrollment columns found: {enrollment_columns}")
        
        program_data = []
        
        # Group by program and major
        for (program, major), group in df.groupby(['PROGRAM NAME', 'MAJOR']):
            if len(group) == 0:
                continue
                
            program_info = {
                'program_name': program,
                'major': major
            }
            
            if enrollment_columns:
                enrollment_data = group[enrollment_columns].sum()
                
                # Try to identify year and gender patterns in column names
                for col in enrollment_columns:
                    col_lower = col.lower()
                    
                    # Look for year indicators
                    if 'first' in col_lower or '1st' in col_lower:
                        if 'male' in col_lower:
                            program_info['first_year_male'] = enrollment_data[col]
                        elif 'female' in col_lower:
                            program_info['first_year_female'] = enrollment_data[col]
                        else:
                            program_info['first_year_total'] = enrollment_data[col]
                    elif 'second' in col_lower or '2nd' in col_lower:
                        if 'male' in col_lower:
                            program_info['second_year_male'] = enrollment_data[col]
                        elif 'female' in col_lower:
                            program_info['second_year_female'] = enrollment_data[col]
                        else:
                            program_info['second_year_total'] = enrollment_data[col]
                    elif 'third' in col_lower or '3rd' in col_lower:
                        if 'male' in col_lower:
                            program_info['third_year_male'] = enrollment_data[col]
                        elif 'female' in col_lower:
                            program_info['third_year_female'] = enrollment_data[col]
                        else:
                            program_info['third_year_total'] = enrollment_data[col]
                    elif 'fourth' in col_lower or '4th' in col_lower:
                        if 'male' in col_lower:
                            program_info['fourth_year_male'] = enrollment_data[col]
                        elif 'female' in col_lower:
                            program_info['fourth_year_female'] = enrollment_data[col]
                        else:
                            program_info['fourth_year_total'] = enrollment_data[col]
                    elif 'male' in col_lower:
                        program_info['male_total'] = enrollment_data[col]
                    elif 'female' in col_lower:
                        program_info['female_total'] = enrollment_data[col]
                    else:
                        # Generic enrollment column
                        program_info[f'enrollment_{len(program_info)}'] = enrollment_data[col]
                
                # Calculate totals
                male_cols = [col for col in program_info.keys() if 'male' in col and 'year' not in col]
                female_cols = [col for col in program_info.keys() if 'female' in col and 'year' not in col]
                
                total_male = sum(program_info.get(col, 0) for col in male_cols)
                total_female = sum(program_info.get(col, 0) for col in female_cols)
                
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
            
            program_data.append(program_info)
        
        print(f"Extracted {len(program_data)} program records")
        if program_data:
            print(f"Sample program data: {program_data[0]}")
        
        return program_data
    
    def process_csv_batch(self, csv_files, batch_size=5):
        """Process CSV files in batches and store data per batch"""
        batch_num = 1
        
        for i in range(0, len(csv_files), batch_size):
            batch = csv_files[i:i+batch_size]
            print(f"\nProcessing batch {batch_num}: {len(batch)} files")
            
            batch_data = []
            
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
                    
                    # Extract enrollment data
                    program_data = self.extract_enrollment_data(df)
                    
                    # Add metadata
                    for program in program_data:
                        program['year_range'] = year_range
                        program['semester'] = semester
                        program['filename'] = filename
                    
                    batch_data.extend(program_data)
                    print(f"  ✓ Processed: {filename} ({len(program_data)} programs)")
                    
                except Exception as e:
                    print(f"  ✗ Error processing {csv_file}: {str(e)}")
            
            # Store batch data
            if batch_data:
                self.batch_data[f'batch_{batch_num}'] = pd.DataFrame(batch_data)
                print(f"Batch {batch_num} data shape: {self.batch_data[f'batch_{batch_num}'].shape}")
            
            batch_num += 1
        
        return self.batch_data
    
    def create_batch_kde_plots(self, batch_name, data):
        """Create KDE plots for a specific batch"""
        print(f"Creating KDE plots for {batch_name}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'KDE Plots - {batch_name} Enrollment Data Distribution', fontsize=16, fontweight='bold')
        
        # KDE plot for total enrollment
        if 'total_enrollment' in data.columns:
            print(f"Total enrollment data: {data['total_enrollment'].describe()}")
            if data['total_enrollment'].sum() > 0:
                sns.kdeplot(data=data, x='total_enrollment', ax=axes[0,0], fill=True, color='skyblue')
                axes[0,0].set_title('Distribution of Total Enrollment', fontweight='bold')
                axes[0,0].set_xlabel('Total Enrollment')
                axes[0,0].set_ylabel('Density')
            else:
                axes[0,0].text(0.5, 0.5, 'No enrollment data available', ha='center', va='center', transform=axes[0,0].transAxes)
                axes[0,0].set_title('Distribution of Total Enrollment', fontweight='bold')
        
        # KDE plot for gender ratio
        if 'gender_ratio' in data.columns:
            print(f"Gender ratio data: {data['gender_ratio'].describe()}")
            if data['gender_ratio'].sum() > 0:
                sns.kdeplot(data=data, x='gender_ratio', ax=axes[0,1], fill=True, color='lightcoral')
                axes[0,1].set_title('Distribution of Gender Ratio', fontweight='bold')
                axes[0,1].set_xlabel('Gender Ratio (Male/Female)')
                axes[0,1].set_ylabel('Density')
            else:
                axes[0,1].text(0.5, 0.5, 'No gender ratio data available', ha='center', va='center', transform=axes[0,1].transAxes)
                axes[0,1].set_title('Distribution of Gender Ratio', fontweight='bold')
        
        # KDE plot for female percentage
        if 'female_percentage' in data.columns:
            print(f"Female percentage data: {data['female_percentage'].describe()}")
            if data['female_percentage'].sum() > 0:
                sns.kdeplot(data=data, x='female_percentage', ax=axes[1,0], fill=True, color='lightgreen')
                axes[1,0].set_title('Distribution of Female Percentage', fontweight='bold')
                axes[1,0].set_xlabel('Female Percentage (%)')
                axes[1,0].set_ylabel('Density')
            else:
                axes[1,0].text(0.5, 0.5, 'No female percentage data available', ha='center', va='center', transform=axes[1,0].transAxes)
                axes[1,0].set_title('Distribution of Female Percentage', fontweight='bold')
        
        # KDE plot for progression rate
        if 'progression_rate_1to2' in data.columns:
            print(f"Progression rate data: {data['progression_rate_1to2'].describe()}")
            if data['progression_rate_1to2'].sum() > 0:
                sns.kdeplot(data=data, x='progression_rate_1to2', ax=axes[1,1], fill=True, color='gold')
                axes[1,1].set_title('Distribution of Progression Rate (1st to 2nd Year)', fontweight='bold')
                axes[1,1].set_xlabel('Progression Rate')
                axes[1,1].set_ylabel('Density')
            else:
                axes[1,1].text(0.5, 0.5, 'No progression rate data available', ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Distribution of Progression Rate (1st to 2nd Year)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{batch_name}_kde_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_batch_dropout_analysis(self, batch_name, data):
        """Create dropout analysis for a specific batch"""
        print(f"Creating dropout analysis for {batch_name}...")
        
        if 'dropout_indicator' not in data.columns:
            print(f"Dropout indicator not found in {batch_name}")
            return
        
        print(f"Dropout indicator data: {data['dropout_indicator'].value_counts()}")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{batch_name} - Dropout Analysis', fontsize=16, fontweight='bold')
        
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
        plt.savefig(f'{batch_name}_dropout_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_batch_correlation_matrix(self, batch_name, data):
        """Create correlation matrix for a specific batch"""
        print(f"Creating correlation matrix for {batch_name}...")
        
        # Select numeric columns for correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        print(f"Numeric columns for correlation: {numeric_cols.tolist()}")
        
        if len(numeric_cols) < 2:
            print(f"Insufficient numeric columns in {batch_name} for correlation")
            return
        
        correlation_data = data[numeric_cols].corr()
        print(f"Correlation matrix shape: {correlation_data.shape}")
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        plt.title(f'{batch_name} - Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{batch_name}_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_batch_academic_success_plots(self, batch_name, data):
        """Create academic success plots for a specific batch"""
        print(f"Creating academic success plots for {batch_name}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{batch_name} - Academic Success Analysis', fontsize=16, fontweight='bold')
        
        # Scatter plot: First year enrollment vs progression rate
        if 'first_year_total' in data.columns and 'progression_rate_1to2' in data.columns:
            print(f"First year total data: {data['first_year_total'].describe()}")
            print(f"Progression rate data: {data['progression_rate_1to2'].describe()}")
            if data['first_year_total'].sum() > 0 and data['progression_rate_1to2'].sum() > 0:
                axes[0,0].scatter(data['first_year_total'], data['progression_rate_1to2'], alpha=0.6, color='blue')
                axes[0,0].set_xlabel('First Year Enrollment')
                axes[0,0].set_ylabel('Progression Rate (1st to 2nd Year)')
                axes[0,0].set_title('Enrollment vs Progression Rate', fontweight='bold')
                axes[0,0].grid(True, alpha=0.3)
            else:
                axes[0,0].text(0.5, 0.5, 'No progression data available', ha='center', va='center', transform=axes[0,0].transAxes)
                axes[0,0].set_title('Enrollment vs Progression Rate', fontweight='bold')
        
        # Line plot: Progression rates across years
        progression_cols = [col for col in data.columns if 'progression_rate' in col]
        if progression_cols:
            print(f"Progression columns: {progression_cols}")
            progression_data = data[progression_cols].mean()
            print(f"Progression data: {progression_data}")
            if progression_data.sum() > 0:
                years = ['1st to 2nd', '2nd to 3rd', '3rd to 4th']
                axes[0,1].plot(years, progression_data.values, marker='o', linewidth=2, markersize=8, color='red')
                axes[0,1].set_xlabel('Year Progression')
                axes[0,1].set_ylabel('Average Progression Rate')
                axes[0,1].set_title('Progression Rates Over Time', fontweight='bold')
                axes[0,1].grid(True, alpha=0.3)
            else:
                axes[0,1].text(0.5, 0.5, 'No progression data available', ha='center', va='center', transform=axes[0,1].transAxes)
                axes[0,1].set_title('Progression Rates Over Time', fontweight='bold')
        
        # Bar plot: Gender distribution by program type
        if 'program_name' in data.columns and 'female_percentage' in data.columns:
            program_gender = data.groupby('program_name')['female_percentage'].mean().sort_values(ascending=False)
            top_programs = program_gender.head(10)
            if top_programs.sum() > 0:
                axes[1,0].barh(range(len(top_programs)), top_programs.values, color='#FF69B4')
                axes[1,0].set_yticks(range(len(top_programs)))
                axes[1,0].set_yticklabels(top_programs.index, fontsize=8)
                axes[1,0].set_xlabel('Female Percentage (%)')
                axes[1,0].set_title('Female Percentage by Program (Top 10)', fontweight='bold')
            else:
                axes[1,0].text(0.5, 0.5, 'No gender data available', ha='center', va='center', transform=axes[1,0].transAxes)
                axes[1,0].set_title('Female Percentage by Program (Top 10)', fontweight='bold')
        
        # Histogram: Distribution of total enrollment
        if 'total_enrollment' in data.columns:
            print(f"Total enrollment for histogram: {data['total_enrollment'].describe()}")
            if data['total_enrollment'].sum() > 0:
                axes[1,1].hist(data['total_enrollment'], bins=20, alpha=0.7, color='#4682B4', edgecolor='black')
                axes[1,1].set_xlabel('Total Enrollment')
                axes[1,1].set_ylabel('Frequency')
                axes[1,1].set_title('Distribution of Total Enrollment', fontweight='bold')
                axes[1,1].grid(True, alpha=0.3)
            else:
                axes[1,1].text(0.5, 0.5, 'No enrollment data available', ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Distribution of Total Enrollment', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{batch_name}_academic_success_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_batch_visualization_analysis(self, batch_size=5):
        """Run the complete batch visualization analysis"""
        print("Batch Enrollment Data Visualization Analysis")
        print("=" * 60)
        
        # Find and process CSV files
        self.find_csv_files()
        if not self.csv_files:
            print("No CSV files found!")
            return
        
        print(f"\nProcessing {len(self.csv_files)} CSV files in batches of {batch_size}...")
        self.process_csv_batch(self.csv_files, batch_size)
        
        # Create visualizations for each batch
        print("\nCreating visualizations for each batch...")
        
        for batch_name, batch_data in self.batch_data.items():
            print(f"\n{'='*50}")
            print(f"Processing {batch_name}")
            print(f"{'='*50}")
            
            print(f"Batch data shape: {batch_data.shape}")
            print(f"Batch columns: {batch_data.columns.tolist()}")
            
            # Check if dropout indicator was created
            if 'dropout_indicator' in batch_data.columns:
                dropout_count = batch_data['dropout_indicator'].value_counts()
                print(f"Dropout analysis: {dropout_count[1]} dropouts, {dropout_count[0]} non-dropouts")
            else:
                print("Warning: Dropout indicator not created")
            
            # Create visualizations for this batch
            print(f"\nCreating visualizations for {batch_name}...")
            
            # 1. KDE Plots
            self.create_batch_kde_plots(batch_name, batch_data)
            
            # 2. Dropout Analysis
            self.create_batch_dropout_analysis(batch_name, batch_data)
            
            # 3. Correlation Matrix
            self.create_batch_correlation_matrix(batch_name, batch_data)
            
            # 4. Academic Success Plots
            self.create_batch_academic_success_plots(batch_name, batch_data)
            
            print(f"Completed visualizations for {batch_name}")
        
        print("\nAll batch visualizations completed and saved!")
        print("\nGenerated files per batch:")
        for batch_name in self.batch_data.keys():
            print(f"- {batch_name}_kde_plots.png")
            print(f"- {batch_name}_dropout_analysis.png")
            print(f"- {batch_name}_correlation_matrix.png")
            print(f"- {batch_name}_academic_success_analysis.png")

def main():
    # Initialize visualizer
    workspace_dir = os.getcwd()
    visualizer = BatchEnrollmentVisualizer(workspace_dir)
    
    # Run batch visualization analysis
    visualizer.run_batch_visualization_analysis(batch_size=5)

if __name__ == "__main__":
    main() 