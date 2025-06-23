import pandas as pd
from datetime import datetime
import re
from collections import Counter
import os
from typing import Dict, List

class FeedbackReportGenerator:
    def __init__(self, data_file: str = "data/pws_chatbot_qa_feedbacks_analyzed.csv"):
        self.data_file = data_file
        self.output_file = "data/feedback_analysis_report.md"
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the analyzed feedback data"""
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"Loaded {len(self.df)} feedback records")
            
            # Merge Conversation Statelessness with Contextual Awareness Failure
            if 'feedback_comment_category' in self.df.columns:
                original_count = (self.df['feedback_comment_category'] == 'Conversation Statelessness').sum()
                self.df['feedback_comment_category'] = self.df['feedback_comment_category'].replace(
                    'Conversation Statelessness', 'Contextual Awareness Failure'
                )
                if original_count > 0:
                    print(f"Merged {original_count} 'Conversation Statelessness' records into 'Contextual Awareness Failure'")
            
            # Convert request_time to datetime
            self.df['request_time'] = pd.to_datetime(self.df['request_time'])
            return self.df
        except FileNotFoundError:
            print(f"Error: {self.data_file} not found")
            return pd.DataFrame()
    
    def calculate_basic_metrics(self) -> Dict:
        """Calculate basic performance metrics"""
        if self.df is None or self.df.empty:
            return {}
            
        total_interactions = len(self.df)
        thumbs_up_count = (self.df['feedback_rating'] == 'THUMBS_UP').sum()
        thumbs_down_count = (self.df['feedback_rating'] == 'THUMBS_DOWN').sum()
        satisfaction_rate = thumbs_up_count / total_interactions if total_interactions > 0 else 0
        
        # Feedback completion rate
        feedback_completion_rate = self.df['feedback_comment'].notna().mean()
        
        # Date range
        date_range = {
            'start_date': self.df['request_time'].min().strftime('%Y-%m-%d'),
            'end_date': self.df['request_time'].max().strftime('%Y-%m-%d'),
            'total_days': (self.df['request_time'].max() - self.df['request_time'].min()).days + 1
        }
        
        return {
            'total_interactions': total_interactions,
            'thumbs_up_count': thumbs_up_count,
            'thumbs_down_count': thumbs_down_count,
            'satisfaction_rate': satisfaction_rate,
            'feedback_completion_rate': feedback_completion_rate,
            'date_range': date_range
        }
    
    def analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal usage and satisfaction patterns"""
        if self.df is None or self.df.empty:
            return {}
            
        # Add time-based columns
        self.df['day_of_week'] = self.df['request_time'].dt.day_name()
        self.df['date'] = self.df['request_time'].dt.date
        
        # Use the full date range available in the data
        filtered_df = self.df.copy()
        
        # Daily trends for the specified period - using named aggregation
        daily_stats = filtered_df.groupby('date').agg(
            total_interactions=('feedback_rating', 'count'),
            satisfaction_rate=('feedback_rating', lambda x: (x == 'THUMBS_UP').mean())
        ).round(3)
        
        # Sort by date in ascending order
        daily_stats = daily_stats.sort_index()
        
        # Day of week patterns (keeping this for overall analysis)
        weekly_stats = self.df.groupby('day_of_week').agg(
            total_interactions=('feedback_rating', 'count'),
            satisfaction_rate=('feedback_rating', lambda x: (x == 'THUMBS_UP').mean())
        ).round(3)
        
        # Peak usage day from the filtered period
        if not daily_stats.empty:
            peak_date = daily_stats['total_interactions'].idxmax()
        else:
            peak_date = None
        
        # Scenario temporal analysis
        scenario_temporal = self.analyze_scenario_temporal_patterns(filtered_df)
        
        return {
            'daily_stats': daily_stats,
            'weekly_stats': weekly_stats,
            'peak_date': peak_date,
            'scenario_temporal': scenario_temporal
        }
    
    def analyze_scenario_temporal_patterns(self, filtered_df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns by scenario (A vs B)"""
        if filtered_df is None or filtered_df.empty or 'scenario' not in filtered_df.columns:
            return {}
        
        # Daily scenario breakdown
        scenario_daily_stats = filtered_df.groupby(['date', 'scenario']).agg(
            total_interactions=('feedback_rating', 'count'),
            satisfaction_rate=('feedback_rating', lambda x: (x == 'THUMBS_UP').mean())
        ).round(3)
        
        # Pivot to get scenarios as columns
        scenario_daily_pivot = scenario_daily_stats.unstack(level='scenario', fill_value=0)
        
        # Overall scenario statistics
        scenario_overall_stats = filtered_df.groupby('scenario').agg(
            total_interactions=('feedback_rating', 'count'),
            satisfaction_rate=('feedback_rating', lambda x: (x == 'THUMBS_UP').mean()),
            avg_daily_interactions=('feedback_rating', lambda x: len(x) / len(filtered_df['date'].unique()) if len(filtered_df['date'].unique()) > 0 else 0)
        ).round(3)
        
        # Scenario distribution by date
        scenario_distribution = filtered_df.groupby(['date', 'scenario']).size().unstack(fill_value=0)
        scenario_distribution['Total'] = scenario_distribution.sum(axis=1)
        
        # Calculate percentages
        scenario_percentages = scenario_distribution.div(scenario_distribution['Total'], axis=0).round(3)
        scenario_percentages = scenario_percentages.drop('Total', axis=1)
        
        return {
            'scenario_daily_stats': scenario_daily_stats,
            'scenario_daily_pivot': scenario_daily_pivot,
            'scenario_overall_stats': scenario_overall_stats,
            'scenario_distribution': scenario_distribution,
            'scenario_percentages': scenario_percentages
        }
    
    def analyze_category_performance(self) -> Dict:
        """Analyze performance by category and sub-category"""
        if self.df is None or self.df.empty:
            return {}
            
        # Category analysis - using named aggregation
        category_stats = self.df.groupby('category').agg(
            total_interactions=('feedback_rating', 'count'),
            satisfaction_rate=('feedback_rating', lambda x: (x == 'THUMBS_UP').mean()),
            comments_received=('feedback_comment', lambda x: x.notna().sum())
        ).round(3)
        
        # Sub-category analysis
        subcategory_stats = self.df.groupby(['category', 'sub_category']).agg(
            total_interactions=('feedback_rating', 'count'),
            satisfaction_rate=('feedback_rating', lambda x: (x == 'THUMBS_UP').mean())
        ).round(3)
        
        # Best and worst performing categories
        best_category = category_stats['satisfaction_rate'].idxmax()
        worst_category = category_stats['satisfaction_rate'].idxmin()
        
        return {
            'category_stats': category_stats,
            'subcategory_stats': subcategory_stats,
            'best_category': best_category,
            'worst_category': worst_category
        }
    
    def analyze_content_quality(self) -> Dict:
        """Analyze content quality metrics"""
        if self.df is None or self.df.empty:
            return {}
            
        # Response and question lengths
        self.df['bot_answer_length'] = self.df['bot_answer'].str.len()
        self.df['question_length'] = self.df['user_question'].str.len()
        
        # Length statistics
        length_stats = {
            'avg_response_length': self.df['bot_answer_length'].mean(),
            'avg_question_length': self.df['question_length'].mean(),
            'median_response_length': self.df['bot_answer_length'].median(),
            'median_question_length': self.df['question_length'].median()
        }
        
        # Length vs satisfaction correlation with exact character ranges
        length_bins = pd.cut(self.df['bot_answer_length'], bins=5)
        length_satisfaction = self.df.groupby(length_bins)['feedback_rating'].apply(
            lambda x: (x == 'THUMBS_UP').mean()
        ).round(3)
        
        # Create a dictionary with exact character ranges as keys
        length_satisfaction_formatted = {}
        for interval, satisfaction in length_satisfaction.items():
            if pd.notna(satisfaction):
                left = int(interval.left)
                right = int(interval.right)
                range_label = f"{left}-{right} chars"
                length_satisfaction_formatted[range_label] = satisfaction
        
        return {
            'length_stats': length_stats,
            'length_satisfaction': length_satisfaction_formatted
        }

    def analyze_feedback_categories_by_date(self) -> Dict:
        """Analyze feedback comment categories distribution by date"""
        if self.df is None or self.df.empty:
            return {}
            
        # Ensure we have the date column
        if 'date' not in self.df.columns:
            self.df['date'] = self.df['request_time'].dt.date
        
        # Create a pivot table of feedback comment categories by date
        # Only include rows where feedback_comment_category is not null
        feedback_by_date_df = self.df[self.df['feedback_comment_category'].notna()]
        
        # Create pivot table: dates as rows, feedback categories as columns, count as values
        pivot_table = pd.pivot_table(
            feedback_by_date_df,
            values='feedback_rating',
            index='date',
            columns='feedback_comment_category',
            aggfunc='count',
            fill_value=0
        )
        
        # Calculate total feedback comments per day
        pivot_table['Total'] = pivot_table.sum(axis=1)
        
        # Convert to dictionary format for easier reporting
        feedback_by_date = {
            'pivot_table': pivot_table,
            'dates': sorted(feedback_by_date_df['date'].unique()),
            'categories': sorted(feedback_by_date_df['feedback_comment_category'].unique())
        }
        
        return feedback_by_date
    
    def analyze_negative_feedback(self) -> Dict:
        """Analyze negative feedback patterns"""
        if self.df is None or self.df.empty:
            return {}
            
        negative_feedback = self.df[self.df['feedback_rating'] == 'THUMBS_DOWN']
        
        if negative_feedback.empty:
            return {'negative_count': 0}
            
        # Common complaint categories
        negative_by_category = negative_feedback.groupby('category').size().sort_values(ascending=False)
        
        # NEW: Analyze feedback comment categories for thumbs down feedback
        feedback_comment_category_dist = negative_feedback['feedback_comment_category'].value_counts()
        
        # NEW: Breakdown by category and feedback comment category
        category_feedback_breakdown = negative_feedback.groupby(['category', 'feedback_comment_category']).size().unstack(fill_value=0)
        
        # Add missing common_complaint_words analysis
        common_complaint_words = []
        if 'feedback_comment' in negative_feedback.columns:
            # Extract words from feedback comments
            all_comments = negative_feedback['feedback_comment'].dropna().str.lower()
            if not all_comments.empty:
                # Combine all comments and extract words
                all_text = ' '.join(all_comments)
                # Remove common stop words and extract meaningful words
                words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
                # Filter out common words
                stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'why', 'too', 'any', 'few', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
                filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
                # Count word frequencies
                word_counts = Counter(filtered_words)
                common_complaint_words = word_counts.most_common(10)
        
        return {
            'negative_count': len(negative_feedback),
            'negative_by_category': negative_by_category,
            'feedback_comment_category_distribution': feedback_comment_category_dist,
            'category_feedback_breakdown': category_feedback_breakdown,
            'common_complaint_words': common_complaint_words
        }
    
    def analyze_question_types(self) -> Dict:
        """Analyze question types and their performance"""
        if self.df is None or self.df.empty:
            return {}
            
        # Categorize questions by type
        def categorize_question(question):
            question_lower = str(question).lower()
            if any(word in question_lower for word in ['how', 'how to']):
                return 'How'
            elif 'what' in question_lower:
                return 'What'
            elif 'when' in question_lower:
                return 'When'
            elif 'where' in question_lower:
                return 'Where'
            elif 'why' in question_lower:
                return 'Why'
            elif any(word in question_lower for word in ['can', 'could', 'is', 'are', 'do', 'does']):
                return 'Yes/No'
            else:
                return 'Other'
        
        self.df['question_type'] = self.df['user_question'].apply(categorize_question)
        
        question_type_stats = self.df.groupby('question_type').agg(
            total_interactions=('feedback_rating', 'count'),
            satisfaction_rate=('feedback_rating', lambda x: (x == 'THUMBS_UP').mean())
        ).round(3)
        
        return {
            'question_type_stats': question_type_stats
        }
    
    def generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Basic metrics recommendations
        if 'basic_metrics' in analysis_results:
            satisfaction_rate = analysis_results['basic_metrics']['satisfaction_rate']
            if satisfaction_rate < 0.7:
                recommendations.append(f"🚨 **Critical**: Overall satisfaction rate is {satisfaction_rate:.1%}, which is below the recommended 70% threshold.")
            elif satisfaction_rate < 0.8:
                recommendations.append(f"⚠️ **Warning**: Satisfaction rate is {satisfaction_rate:.1%}. Consider improvements to reach 80%+ target.")
            else:
                recommendations.append(f"✅ **Good**: Satisfaction rate of {satisfaction_rate:.1%} is above target.")
        
        # Category-based recommendations
        if 'category_performance' in analysis_results:
            worst_category = analysis_results['category_performance']['worst_category']
            worst_satisfaction = analysis_results['category_performance']['category_stats'].loc[worst_category, 'satisfaction_rate']
            recommendations.append(f"📊 Focus improvement efforts on '{worst_category}' category (satisfaction: {worst_satisfaction:.1%})")
            
            best_category = analysis_results['category_performance']['best_category']
            best_satisfaction = analysis_results['category_performance']['category_stats'].loc[best_category, 'satisfaction_rate']
            recommendations.append(f"🏆 Leverage successful patterns from '{best_category}' category (satisfaction: {best_satisfaction:.1%})")
        
        # Scenario performance recommendations
        if 'temporal_patterns' in analysis_results and 'scenario_temporal' in analysis_results['temporal_patterns']:
            scenario_data = analysis_results['temporal_patterns']['scenario_temporal']
            if 'scenario_overall_stats' in scenario_data and not scenario_data['scenario_overall_stats'].empty:
                scenario_stats = scenario_data['scenario_overall_stats']
                
                if 'A' in scenario_stats.index and 'B' in scenario_stats.index:
                    scenario_a_satisfaction = scenario_stats.loc['A', 'satisfaction_rate']
                    scenario_b_satisfaction = scenario_stats.loc['B', 'satisfaction_rate']
                    scenario_a_interactions = int(scenario_stats.loc['A', 'total_interactions'])
                    scenario_b_interactions = int(scenario_stats.loc['B', 'total_interactions'])
                    
                    if scenario_a_satisfaction > scenario_b_satisfaction:
                        diff = scenario_a_satisfaction - scenario_b_satisfaction
                        recommendations.append(f"🎯 **Scenario Analysis**: Provided questions (Scenario A) perform {diff:.1%} better than open-ended questions (Scenario B). Consider expanding the provided question set.")
                    elif scenario_b_satisfaction > scenario_a_satisfaction:
                        diff = scenario_b_satisfaction - scenario_a_satisfaction
                        recommendations.append(f"💡 **Scenario Analysis**: Open-ended questions (Scenario B) perform {diff:.1%} better than provided questions (Scenario A). Users may prefer more flexibility in question types.")
                    else:
                        recommendations.append(f"⚖️ **Scenario Analysis**: Both provided and open-ended questions show similar satisfaction rates. Current question strategy is balanced.")
                    
                    total_interactions = scenario_a_interactions + scenario_b_interactions
                    scenario_a_percentage = (scenario_a_interactions / total_interactions) * 100
                    scenario_b_percentage = (scenario_b_interactions / total_interactions) * 100
                    
                    if scenario_a_percentage > 70:
                        recommendations.append(f"📋 **Question Distribution**: {scenario_a_percentage:.1f}% of questions are from provided set. Consider monitoring if users need more diverse question options.")
                    elif scenario_b_percentage > 70:
                        recommendations.append(f"🔍 **Question Distribution**: {scenario_b_percentage:.1f}% of questions are open-ended. Consider expanding the provided question set to cover more user needs.")
        
        # Temporal recommendations
        if 'temporal_patterns' in analysis_results:
            peak_date = analysis_results['temporal_patterns']['peak_date']
            if peak_date:
                recommendations.append(f"📅 **Peak Usage**: Highest activity on {peak_date} during the analyzed period.")
        
        #peak_hour = analysis_results['temporal_patterns']['peak_hour']
        #recommendations.append(f"⏰ **Staffing**: Peak usage at {peak_hour}:00 - ensure adequate support coverage.")
        
        # Content quality recommendations
        if 'content_quality' in analysis_results:
            avg_length = analysis_results['content_quality']['length_stats']['avg_response_length']
            if avg_length < 50:
                recommendations.append("📝 **Content**: Responses may be too brief. Consider providing more detailed answers.")
            elif avg_length > 500:
                recommendations.append("📝 **Content**: Responses may be too lengthy. Consider more concise answers.")
        
        # Negative feedback recommendations
        if 'negative_feedback' in analysis_results and analysis_results['negative_feedback']['negative_count'] > 0:
            top_complaint_category = analysis_results['negative_feedback']['negative_by_category'].index[0]
            recommendations.append(f"🔧 **Fix**: Address issues in '{top_complaint_category}' - highest source of complaints.")
        
        return recommendations
    
    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report"""
        if self.df is None:
            self.load_data()
            
        if self.df.empty:
            return "# Error: No data available for analysis"
        
        # Perform all analyses
        basic_metrics = self.calculate_basic_metrics()
        temporal_patterns = self.analyze_temporal_patterns()
        category_performance = self.analyze_category_performance()
        content_quality = self.analyze_content_quality()
        negative_feedback = self.analyze_negative_feedback()
        question_types = self.analyze_question_types()
        
        analysis_results = {
            'basic_metrics': basic_metrics,
            'temporal_patterns': temporal_patterns,
            'category_performance': category_performance,
            'content_quality': content_quality,
            'negative_feedback': negative_feedback,
            'question_types': question_types
        }
        
        recommendations = self.generate_recommendations(analysis_results)
        
        # Generate report
        report = f"""# 🤖 Chatbot Feedback Analysis Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Executive Summary

### Key Metrics
- **Total Interactions:** {basic_metrics['total_interactions']:,}
- **Overall Satisfaction Rate:** {basic_metrics['satisfaction_rate']:.1%}
- **Thumbs Up:** {basic_metrics['thumbs_up_count']:,} ({basic_metrics['thumbs_up_count']/basic_metrics['total_interactions']:.1%})
- **Thumbs Down:** {basic_metrics['thumbs_down_count']:,} ({basic_metrics['thumbs_down_count']/basic_metrics['total_interactions']:.1%})
- **Feedback Completion Rate:** {basic_metrics['feedback_completion_rate']:.1%}
- **Analysis Period:** {basic_metrics['date_range']['start_date']} to {basic_metrics['date_range']['end_date']} ({basic_metrics['date_range']['total_days']} days)

---

## 🎯 Key Recommendations

"""
        
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""

---

## 📈 Temporal Analysis

### Usage Patterns
- **Peak Usage Date:** {temporal_patterns['peak_date'] if temporal_patterns['peak_date'] else 'No data available'}
- **📚 Performance Note:** Starting June 3rd, we began adding examples to responses, which explains the climbing performance trend observed in the data.

### Daily Distribution
| Date | Interactions | Satisfaction Rate |
|------|-------------|------------------|
"""
        
        for date, stats in temporal_patterns['daily_stats'].iterrows():
            interactions = int(stats['total_interactions'])
            satisfaction = stats['satisfaction_rate']
            report += f"| {date} | {interactions} | {satisfaction:.1%} |\n"
        
        report += f"""

### Weekly Distribution
| Day | Interactions | Satisfaction Rate |
|-----|-------------|------------------|
"""
        
        for day, stats in temporal_patterns['weekly_stats'].iterrows():
            interactions = int(stats['total_interactions'])
            satisfaction = stats['satisfaction_rate']
            report += f"| {day} | {interactions} | {satisfaction:.1%} |\n"
        
        # Add scenario temporal analysis if available
        if 'scenario_temporal' in temporal_patterns and temporal_patterns['scenario_temporal']:
            scenario_data = temporal_patterns['scenario_temporal']
            
            report += f"""

### Scenario Analysis (Temporal)

#### Overall Scenario Performance
| Scenario | Total Interactions | Satisfaction Rate | Avg Daily Interactions |
|----------|-------------------|------------------|----------------------|
"""
            
            if 'scenario_overall_stats' in scenario_data:
                for scenario, stats in scenario_data['scenario_overall_stats'].iterrows():
                    interactions = int(stats['total_interactions'])
                    satisfaction = stats['satisfaction_rate']
                    avg_daily = stats['avg_daily_interactions']
                    scenario_label = f"Scenario {scenario}" if scenario in ['A', 'B'] else scenario
                    report += f"| {scenario_label} | {interactions} | {satisfaction:.1%} | {avg_daily:.1f} |\n"
            
            report += f"""
 
 #### Daily Scenario Distribution
 | Date | Scenario A | Scenario B | Total | Scenario A % | Scenario B % | A Satisfaction | B Satisfaction |
 |------|-----------|-----------|-------|-------------|-------------|---------------|---------------|
 """
             
            if 'scenario_distribution' in scenario_data and 'scenario_percentages' in scenario_data and 'scenario_daily_stats' in scenario_data:
                 dist_data = scenario_data['scenario_distribution']
                 perc_data = scenario_data['scenario_percentages']
                 daily_stats = scenario_data['scenario_daily_stats']
                 
                 for date in dist_data.index:
                     scenario_a = dist_data.loc[date, 'A'] if 'A' in dist_data.columns else 0
                     scenario_b = dist_data.loc[date, 'B'] if 'B' in dist_data.columns else 0
                     total = dist_data.loc[date, 'Total']
                     perc_a = perc_data.loc[date, 'A'] if 'A' in perc_data.columns else 0
                     perc_b = perc_data.loc[date, 'B'] if 'B' in perc_data.columns else 0
                     
                     # Get satisfaction rates for each scenario on this date
                     sat_a = "N/A"
                     sat_b = "N/A"
                     
                     if (date, 'A') in daily_stats.index:
                         sat_a = f"{daily_stats.loc[(date, 'A'), 'satisfaction_rate']:.1%}"
                     if (date, 'B') in daily_stats.index:
                         sat_b = f"{daily_stats.loc[(date, 'B'), 'satisfaction_rate']:.1%}"
                     
                     report += f"| {date} | {scenario_a} | {scenario_b} | {total} | {perc_a:.1%} | {perc_b:.1%} | {sat_a} | {sat_b} |\n"
        
        report += f"""

---

## 🏷️ Category Performance

### Overall Category Performance
| Category | Interactions | Satisfaction Rate | Comments Received |
|----------|-------------|------------------|------------------|
"""
        
        for category, stats in category_performance['category_stats'].iterrows():
            interactions = int(stats['total_interactions'])
            satisfaction = stats['satisfaction_rate']
            comments = int(stats['comments_received'])
            report += f"| {category} | {interactions} | {satisfaction:.1%} | {comments} |\n"
        
        report += f"""

### Best vs Worst Performing Categories
- **🏆 Best Performing:** {category_performance['best_category']} ({category_performance['category_stats'].loc[category_performance['best_category'], 'satisfaction_rate']:.1%} satisfaction)
- **⚠️ Needs Improvement:** {category_performance['worst_category']} ({category_performance['category_stats'].loc[category_performance['worst_category'], 'satisfaction_rate']:.1%} satisfaction)

---

## 📝 Content Quality Analysis

### Response Length Statistics
- **Average Response Length:** {content_quality['length_stats']['avg_response_length']:.0f} characters
- **Median Response Length:** {content_quality['length_stats']['median_response_length']:.0f} characters
- **Average Question Length:** {content_quality['length_stats']['avg_question_length']:.0f} characters

### Satisfaction by Response Length
| Length Category | Satisfaction Rate |
|-----------------|------------------|
"""
        
        for length_cat, satisfaction in content_quality['length_satisfaction'].items():
            if pd.notna(satisfaction):
                report += f"| {length_cat} | {satisfaction:.1%} |\n"
        
        report += f"""

---

## ❓ Question Type Analysis

| Question Type | Interactions | Satisfaction Rate |
|---------------|-------------|------------------|
"""
        
        for q_type, stats in question_types['question_type_stats'].iterrows():
            interactions = int(stats['total_interactions'])
            satisfaction = stats['satisfaction_rate']
            report += f"| {q_type} | {interactions} | {satisfaction:.1%} |\n"
        
        if negative_feedback['negative_count'] > 0:
            report += f"""

---

## 👎 Negative Feedback Analysis

### Failed Cases Distribution by Feedback Comment Category
| Feedback Comment Category | Count | Percentage |
|---------------------------|-------|------------|
"""
            
            total_negative = negative_feedback['negative_count']
            for category, count in negative_feedback['feedback_comment_category_distribution'].items():
                percentage = (count / total_negative) * 100
                report += f"| {category} | {count} | {percentage:.1f}% |\n"
            
            report += f"""

### Breakdown by Category and Feedback Comment Category
| Category | """
            
            # Add column headers for feedback comment categories
            feedback_categories = negative_feedback['category_feedback_breakdown'].columns.tolist()
            for fc in feedback_categories:
                report += f"{fc} | "
            report += "Total |\n"
            
            # Add separator row
            report += "|----------|" + "---------|" * len(feedback_categories) + "-------|\n"
            
            # Add data rows
            for category in negative_feedback['category_feedback_breakdown'].index:
                report += f"| {category} | "
                row_total = 0
                for fc in feedback_categories:
                    count = negative_feedback['category_feedback_breakdown'].loc[category, fc]
                    report += f"{count} | "
                    row_total += count
                report += f"{row_total} |\n"


        # In generate_markdown_report method, add after loading other analyses:
        feedback_categories_by_date = self.analyze_feedback_categories_by_date()
        
        # Add to analysis_results dictionary:
        analysis_results['feedback_categories_by_date'] = feedback_categories_by_date
        
        # Add this section to the report (after the Negative Feedback Analysis section):
        if feedback_categories_by_date and 'pivot_table' in feedback_categories_by_date:
            report += f"""

---

## 📅 Feedback Categories by Date

### Distribution of Feedback Categories Across Dates
| Date | {' | '.join(feedback_categories_by_date['categories'])} | Total |
|------|{'----|' * len(feedback_categories_by_date['categories'])}------|
"""
            
            for date in feedback_categories_by_date['dates']:
                report += f"| {date} | "
                for category in feedback_categories_by_date['categories']:
                    count = feedback_categories_by_date['pivot_table'].loc[date, category] if date in feedback_categories_by_date['pivot_table'].index and category in feedback_categories_by_date['pivot_table'].columns else 0
                    report += f"{count} | "
                total = feedback_categories_by_date['pivot_table'].loc[date, 'Total'] if date in feedback_categories_by_date['pivot_table'].index else 0
                report += f"{total} |\n"
            
            report += f"""

### Negative Feedback by Category
| Category | Negative Feedback Count |
|----------|------------------------|
"""
            
            for category, count in negative_feedback['negative_by_category'].head(5).items():
                report += f"| {category} | {count} |\n"
            
            if negative_feedback['common_complaint_words']:
                report += f"""

### Common Complaint Keywords
| Keyword | Frequency |
|---------|----------|
"""
                
                for word, freq in negative_feedback['common_complaint_words']:
                    report += f"| {word} | {freq} |\n"
        
        report += f"""

---

## 📋 Action Items

### Immediate Actions (Next 1-2 weeks)
- [ ] Review and improve responses for '{category_performance['worst_category']}' category
- [ ] Analyze specific complaints in top negative feedback categories
- [ ] Optimize response templates for peak usage hours

### Short-term Improvements (Next 1-3 months)
- [ ] Implement A/B testing for response length optimization
- [ ] Develop category-specific response guidelines
- [ ] Create feedback loop mechanism for continuous improvement

### Long-term Strategy (3+ months)
- [ ] Implement advanced NLP for better question understanding
- [ ] Develop predictive models for user satisfaction
- [ ] Create automated quality assurance system

---

## 📊 Data Quality Notes

- **Total Records Analyzed:** {basic_metrics['total_interactions']:,}
- **Records with Comments:** {int(basic_metrics['feedback_completion_rate'] * basic_metrics['total_interactions']):,}
- **Data Completeness:** {(1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))):.1%}

---

*Report generated by Feedback Analysis Pipeline v1.0*
*For questions or additional analysis, contact the development team.*
"""
        
        return report
    
    def save_report(self, report_content: str) -> None:
        """Save the report to a markdown file"""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Report saved to: {self.output_file}")
    
    def generate_and_save_report(self) -> str:
        """Generate and save the complete analysis report"""
        print("Generating comprehensive feedback analysis report...")
        
        report_content = self.generate_markdown_report()
        self.save_report(report_content)
        
        return report_content

def main():
    """Main function to generate the feedback report"""
    generator = FeedbackReportGenerator()
    
    try:
        report = generator.generate_and_save_report()
        print("\n" + "="*50)
        print("REPORT GENERATION COMPLETE")
        print("="*50)
        print(f"Report saved to: {generator.output_file}")
        print(f"Report length: {len(report):,} characters")
        
        # Display first few lines of the report
        print("\nReport Preview:")
        print("-" * 30)
        lines = report.split('\n')[:10]
        for line in lines:
            print(line)
        print("...")
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return

if __name__ == "__main__":
    main()
