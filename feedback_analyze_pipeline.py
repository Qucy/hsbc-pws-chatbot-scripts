import pandas as pd
import json
import asyncio
import aiohttp
from openai import AsyncOpenAI
import os
import pandas as pd
from typing import List, Dict, Optional, Tuple
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class FeedbackAnalyzer:
    def __init__(self, api_key: str = '', base_url: str = ''):
        # Use environment variables if not provided as parameters
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = base_url or os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        
        if not self.api_key:
            raise ValueError("API key is required. Please set DEEPSEEK_API_KEY in .env file or pass as parameter.")
        
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.data_file = "data/pws_chatbot_qa_feedbacks.csv"
        self.comments_output_file = "data/pws_chatbot_qa_feedbacks_with_comments.csv"
        self.categories_output_file = "data/pws_chatbot_qa_feedbacks_with_categories.csv"
        self.comment_categories_output_file = "data/pws_chatbot_qa_feedbacks_with_comment_categories.csv"
        self.scenarios_output_file = "data/pws_chatbot_qa_feedbacks_with_scenarios.csv"
        self.mapped_questions_file = "data/mapped_questions.csv"
        self.merged_output_file = "data/pws_chatbot_qa_feedbacks_analyzed.csv"
        
        # Get max concurrent requests from environment or use default
        self.max_concurrent = int(os.getenv('MAX_CONCURRENT_REQUESTS', 20))
    
    def load_feedback_data(self) -> pd.DataFrame:
        """Load feedback data from CSV file"""
        try:
            feedbacks = pd.read_csv(self.data_file)
            print(f"Loaded {len(feedbacks)} feedback records")
            return feedbacks
        except FileNotFoundError:
            print(f"Error: {self.data_file} not found")
            return pd.DataFrame()
    
    def create_record_key(self, row) -> str:
        """Create a unique key for each record based on request_time, user_question, and bot_answer"""
        return f"{row['request_time']}|{row['user_question']}|{row['bot_answer']}"
    
    def get_processed_records(self, output_file: str) -> set:
        """Get set of already processed record keys from output file"""
        if not os.path.exists(output_file):
            return set()
        
        try:
            processed_df = pd.read_csv(output_file)
            processed_keys = set()
            for _, row in processed_df.iterrows():
                key = self.create_record_key(row)
                processed_keys.add(key)
            print(f"Found {len(processed_keys)} already processed records in {output_file}")
            return processed_keys
        except Exception as e:
            print(f"Error reading processed records from {output_file}: {str(e)}")
            return set()
    
    def filter_unprocessed_records(self, df: pd.DataFrame, output_file: str) -> pd.DataFrame:
        """Filter dataframe to only include records that haven't been processed yet"""
        processed_keys = self.get_processed_records(output_file)
        
        if not processed_keys:
            print("No previously processed records found, processing all records")
            return df
        
        # Create keys for current data
        current_keys = set()
        unprocessed_indices = []
        
        for idx, row in df.iterrows():
            key = self.create_record_key(row)
            if key not in processed_keys:
                unprocessed_indices.append(idx)
            current_keys.add(key)
        
        unprocessed_df = df.loc[unprocessed_indices].copy()
        print(f"Found {len(unprocessed_df)} new/unprocessed records out of {len(df)} total records")
        
        return unprocessed_df
    
    def merge_with_existing_results(self, new_df: pd.DataFrame, output_file: str) -> pd.DataFrame:
        """Merge new results with existing processed data"""
        if not os.path.exists(output_file):
            return new_df
        
        try:
            existing_df = pd.read_csv(output_file)
            print(f"Merging {len(new_df)} new records with {len(existing_df)} existing records")
            
            # Combine existing and new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates based on record key (keep last occurrence)
            combined_df['_temp_key'] = combined_df.apply(self.create_record_key, axis=1)
            combined_df = combined_df.drop_duplicates(subset=['_temp_key'], keep='last')
            combined_df = combined_df.drop(columns=['_temp_key'])
            
            print(f"Final merged dataset contains {len(combined_df)} records")
            return combined_df
        except Exception as e:
            print(f"Error merging with existing results: {str(e)}")
            return new_df
    
    def identify_thumbs_down_empty_feedback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify THUMBS_DOWN rows with empty feedback comments"""
        thumbs_down_empty = df[
            (df['feedback_rating'] == 'THUMBS_DOWN') & 
            (df['feedback_comment'].isna() | (df['feedback_comment'].str.strip() == ''))
        ]
        print(f"Found {len(thumbs_down_empty)} THUMBS_DOWN records with empty feedback comments")
        return thumbs_down_empty
    
    async def generate_bot_comment_async(self, user_question: str, bot_answer: str) -> Optional[str]:
        """Generate bot comment for THUMBS_DOWN feedback using async LLM API"""
        system_prompt = """
        You are an AI assistant that analyzes chatbot interactions to generate feedback comments.
        
        Given a user question and bot answer that received a THUMBS_DOWN rating, 
        generate a realistic feedback comment that explains why the user might have given a negative rating.
        
        Focus on what might be wrong with the answer:
        - Unhelpful or vague responses
        - Incorrect information
        - Not addressing the user's specific question
        - Directing to customer service instead of providing useful information
        - Missing important details
        
        Keep the comment concise (1-2 sentences) and realistic as if written by a frustrated user.
        
        Return the response in JSON format with a "comment" field.
        """
        
        user_prompt = f"""
        User Question: {user_question}
        Bot Answer: {bot_answer}
        
        Generate an appropriate negative feedback comment explaining why this answer deserves a THUMBS_DOWN.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={'type': 'json_object'}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('comment', '')
        except Exception as e:
            print(f"Error generating bot comment: {str(e)}")
            return None
    
    async def categorize_question_async(self, user_question: str, bot_answer: str) -> Tuple[str, str]:
        """Categorize a question based on user question and bot answer using async LLM API"""
        system_prompt = """
    You are an expert at categorizing banking and financial service questions based on HSBC's website structure.
    
    Based on the user question and bot answer, categorize the question according to this sitemap structure:
    
    Accounts
    ├── HSBC One
    ├── HSBC Premier
    ├── HSBC Advance
    ├── HSBC Jade
    ├── PayMe from HSBC
    ├── Savings accounts
    ├── Current accounts
    ├── Time deposits
    ├── Foreign currency accounts
    ├── Joint accounts
    ├── Children's accounts
    ├── Student accounts
    ├── Fees and charges
    └── Greater Bay Area Services
    └── HSBC GBA Wealth Management Connect
    
    Ways to bank
    ├── Branch Banking
    ├── Online Banking
    ├── Chat with us
    ├── HSBC Mobile Apps
    ├── HSBC HK Mobile Banking app
    ├── Mobile account opening
    ├── HSBC HK Mobile Banking App - Lite Mode
    ├── Mobile Cash Withdrawal
    ├── Mobile cheque deposit
    ├── Phone banking
    └── 30-Day Service Pledge
    
    HSBC credit cards
    ├── Compare credit cards
    ├── eStatements
    ├── HSBC EveryMile Credit Card
    ├── HSBC Premier Mastercard®
    ├── HSBC Red Credit Card
    ├── HSBC Visa Gold Card for Students
    ├── HSBC Pulse UnionPay Dual Currency Diamond Credit Card
    ├── HSBC UnionPay Dual Currency Credit Card
    ├── HSBC Visa Gold Card
    ├── HSBC easy Credit Card / Visa Platinum Credit Card
    ├── HSBC Visa Signature Card
    ├── HSBC Reward+ Mobile App
    ├── Credit Cards Application
    ├── Cash Credit Plan
    ├── Cash Instalment Plan
    ├── Mobile payment and Octopus add value services
    ├── Red Hot Rewards
    ├── RewardCash Certificate Scheme
    ├── Rewards of Your Choice
    ├── Miles and travel privileges
    ├── Instant RewardCash redemption at merchants
    ├── Using your credit card
    ├── Credit card limit transfer
    └── Fees and charges
    
    Loans
    ├── Personal Instalment Loan
    ├── Personal Tax Loan
    ├── Balance Consolidation Program
    ├── Personal Instalment Loan Redraw
    ├── Revolving Credit Facility
    ├── Electric Vehicle Personal Instalment Loan
    └── Fees and charges
    
    Investments
    ├── Open an Investment Account
    ├── Stocks
    ├── Unit Trusts
    ├── Bonds / Certificates of Deposit (CDs)
    ├── Structured products
    ├── Foreign exchange
    ├── Exchange rate calculator
    ├── ESG investing
    ├── HSBC Investment platform guide
    ├── Gold trading
    ├── Wealth Insights
    ├── Wealth financing
    ├── Fees and charges
    ├── Top Trade Club
    └── Trade25
    
    Insurance
    ├── About HSBC Life
    ├── HSBC Life Benefits+
    ├── Life Insurance
    ├── AccidentSurance
    ├── Motor insurance
    ├── TravelSurance
    ├── Home and domestic helper insurance
    ├── Medical and critical illness insurance
    ├── Savings insurance and retirement plans
    ├── Investment Performance
    ├── Well+ (Well Plus)
    └── Making a claim and getting assistance
    
    Help and support
    ├── About our website
    ├── Important notices
    ├── Dealing with Bereavement
    ├── Credit card support
    ├── Contact Us
    ├── Cyber security and fraud hub
    ├── Frequently asked questions
    ├── Forms and Documents Download
    ├── Banking support for customers with health issues
    ├── Money worries
    ├── How a separation could affect your finances
    ├── Accessibility
    ├── Hyperlink policy
    ├── Maintenance schedule
    ├── Privacy and security
    ├── Regulatory disclosures
    └── Terms of use
    
    International services
    ├── How to open an overseas account
    ├── How to open a Hong Kong account
    ├── International mortgages
    ├── Investing in Hong Kong
    └── Living in Hong Kong
            
    Mortgages:
        ├── Home Ownership Scheme
        ├── Tenants Purchase Scheme
        ├── Deposit-linked Mortgage
        ├── Green Mortgage
        ├── HighAdvance Mortgage
        ├── Investor Mortgage
        ├── HIBOR based Mortgage
        ├── Property valuation tool
    
    Payments and transfers:
        ├── Faster Payment System (FPS)
        ├── Local transfers
        ├── Global payments
        ├── Pay abroad with FPS
        ├── Bill Payments
        ├── autoPay
        ├── Daily payment and transfer limits
    
    Community Banking:
        ├── Accessibility for people with disabilities
        ├── Age-friendly Banking
        ├── Banking with mental health struggles and financial stress
        ├── Banking services for minority groups
        ├── Our impact
    
    MPF:
        ├── MPF Academy
        ├── HSBC MPF Awards
        ├── Forms and documents
        ├── Glossary
        ├── MPF management fees
        ├── MPF news
        ├── MPF for members
        ├── MPF for employers
        ├── MPF constituent fund information
        ├── MPF Personal Accounts
        ├── MPF for the self-employed
        ├── Tax Deductible Voluntary Contributions account
        ├── HSBC Retirement Monitor
        ├── Designated branches with MPF services
        ├── Cumulative Performance
        ├── Retirement planner
        ├── Understanding MPF
        ├── Useful links
        ├── Fees and charges
            
        Return the response in JSON format with "category" and "sub_category" fields containing the appropriate level 1 and level 2 category names respectively.
        If no specific sub-category applies, return "General" for the sub_category field.
        """
        
        user_prompt = f"""
        User Question: {user_question}
        Bot Answer: {bot_answer}
        
        Categorize this question.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={'type': 'json_object'}
            )
            
            result = json.loads(response.choices[0].message.content)
            category = result.get('category', 'other')
            sub_category = result.get('sub_category', 'General')
            return category, sub_category
        except Exception as e:
            print(f"Error categorizing question: {str(e)}")
            return 'other', 'General'
    
    async def add_categories_async(self, df: pd.DataFrame, max_concurrent: int = None) -> pd.DataFrame:
        """Add category and sub_category columns to dataframe using async API calls"""
        if max_concurrent is None:
            max_concurrent = self.max_concurrent
            
        print(f"Adding categories to {len(df)} records with async API calls...")
        
        # Add category and sub_category columns if they don't exist
        result_df = df.copy()
        if 'category' not in result_df.columns:
            result_df['category'] = ''
        if 'sub_category' not in result_df.columns:
            result_df['sub_category'] = ''
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_category(idx, row):
            async with semaphore:
                try:
                    category, sub_category = await self.categorize_question_async(
                        row['user_question'], 
                        row['bot_answer']
                    )
                    return idx, category, sub_category
                except Exception as e:
                    print(f"Error processing row {idx}: {str(e)}")
                    return idx, 'other', 'General'
        
        # Create tasks for all rows
        tasks = [
            process_single_category(idx, row) 
            for idx, row in df.iterrows()
        ]
        
        # Process all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        print(f"Completed {len(results)} categorization API calls in {end_time - start_time:.2f} seconds")
        
        # Update the dataframe with results
        success_count = 0
        for result in results:
            if isinstance(result, tuple) and len(result) == 3:
                idx, category, sub_category = result
                if category and sub_category:
                    result_df.at[idx, 'category'] = category
                    result_df.at[idx, 'sub_category'] = sub_category
                    success_count += 1
        
        print(f"Successfully categorized {success_count} records")
        return result_df
    
    async def add_comments_async(self, df: pd.DataFrame, max_concurrent: int = None) -> pd.DataFrame:
        """Add comments to THUMBS_DOWN empty feedback using async API calls"""
        if max_concurrent is None:
            max_concurrent = self.max_concurrent
            
        print(f"Adding comments to THUMBS_DOWN records with empty feedback...")
        
        # Ensure feedback_comment column exists
        result_df = df.copy()
        if 'feedback_comment' not in result_df.columns:
            result_df['feedback_comment'] = ''
        
        # Identify THUMBS_DOWN with empty feedback
        thumbs_down_empty = self.identify_thumbs_down_empty_feedback(result_df)
        
        if thumbs_down_empty.empty:
            print("No THUMBS_DOWN records with empty feedback found")
            return result_df
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_comment(idx, row):
            # Only process THUMBS_DOWN with empty comments
            if (row['feedback_rating'] == 'THUMBS_DOWN' and 
                (pd.isna(row['feedback_comment']) or row['feedback_comment'].strip() == '')):
                
                async with semaphore:
                    try:
                        comment = await self.generate_bot_comment_async(
                            row['user_question'], 
                            row['bot_answer']
                        )
                        return idx, comment
                    except Exception as e:
                        print(f"Error processing row {idx}: {str(e)}")
                        return idx, ''
            else:
                return idx, row['feedback_comment']
        
        # Create tasks for all rows
        tasks = [
            process_single_comment(idx, row) 
            for idx, row in df.iterrows()
        ]
        
        # Process all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        print(f"Completed {len(results)} comment generation API calls in {end_time - start_time:.2f} seconds")
        
        # Update the dataframe with results
        success_count = 0
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                idx, comment = result
                if comment:
                    result_df.at[idx, 'feedback_comment'] = comment
                    success_count += 1
        
        print(f"Successfully generated {success_count} comments")
        
        return result_df
    
    async def process_comments_only(self) -> pd.DataFrame:
        """Process only comment generation for THUMBS_DOWN feedback"""
        print("\n=== PROCESSING COMMENTS ONLY ===")
        
        # Load feedback data
        df = self.load_feedback_data()
        if df.empty:
            return df
        
        # Filter to only unprocessed records
        unprocessed_df = self.filter_unprocessed_records(df, self.comments_output_file)
        
        if unprocessed_df.empty:
            print("No new records to process for comments")
            # Return existing processed data
            if os.path.exists(self.comments_output_file):
                return pd.read_csv(self.comments_output_file)
            else:
                return df
        
        # Generate comments for THUMBS_DOWN feedback with empty comments
        new_results_df = await self.add_comments_async(unprocessed_df)
        
        # Merge with existing results
        final_df = self.merge_with_existing_results(new_results_df, self.comments_output_file)
        
        # Save merged results
        self.save_results(final_df, self.comments_output_file)
        
        return final_df
    
    async def process_categories_only(self) -> pd.DataFrame:
        """Process only categorization"""
        print("\n=== PROCESSING CATEGORIES ONLY ===")
        
        # Load feedback data
        df = self.load_feedback_data()
        if df.empty:
            return df
        
        # Filter to only unprocessed records
        unprocessed_df = self.filter_unprocessed_records(df, self.categories_output_file)
        
        if unprocessed_df.empty:
            print("No new records to process for categories")
            # Return existing processed data
            if os.path.exists(self.categories_output_file):
                return pd.read_csv(self.categories_output_file)
            else:
                return df
        
        # Add categories
        new_results_df = await self.add_categories_async(unprocessed_df)
        
        # Merge with existing results
        final_df = self.merge_with_existing_results(new_results_df, self.categories_output_file)
        
        # Save merged results
        self.save_results(final_df, self.categories_output_file)
        
        return final_df
    
    def save_results(self, df: pd.DataFrame, output_file: str) -> None:
        """Save the processed dataframe to CSV"""
        try:
            df.to_csv(output_file, index=False)
            # save a copy of xlsx
            df.to_excel(output_file.replace('.csv', '.xlsx'), index=False)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    async def categorize_feedback_comment_async(self, feedback_comment: str) -> str:
        """Categorize a feedback comment using async LLM API"""
        if not feedback_comment or pd.isna(feedback_comment) or feedback_comment.strip() == '':
            return ''
            
        system_prompt = """
        You are an expert at categorizing feedback comments for a banking chatbot.
        
        Based on the feedback comment, categorize it into one of the following categories:
        
        Core Problem Categories:
        - Irrelevant Answer: Response doesn't address the query or introduces unrelated topics.
        - Incomplete/Generic Answer: Response lacks specifics, details, or actionable steps.
        - Redirect to Customer Service: Premature deflection to human agents without attempting resolution.
        - Missing Information/Source: Fails to provide requested data, links, or documentation.
        - Broken/Incorrect Links: Hyperlinks lead to errors, irrelevant pages, or outdated content.
        
        Technical & Functional Failures:
        - Conversation Statelessness: Inability to retain context across interactions.
        - Error Messages: Technical failures (e.g., "Something went wrong").
        - Information Retrieval Failure: Cannot fetch data from valid sources (PDFs, websites, PWS).
        - Link Management Issues: Duplicated, missing, or irrelevant URLs.
        
        User Experience & Communication Gaps:
        - Lack of Comparison/Summary: Fails to contrast products/plans as requested.
        - No Step-by-Step Guidance: Omits clear instructions for processes.
        - Ambiguous/Vague Response: Answers lack clarity or specificity.
        - Incorrect/Factual Errors: Provides inaccurate data or outdated info.
        - Poor Tone/Phrasing: Unhelpful language (e.g., "I don't know", negative framing).
        
        Advanced Capability Shortfalls:
        - Contextual Awareness Failure: Misinterpreting follow-ups or user intent.
        - Inability to Handle Complex Queries: Struggles with multi-part or nuanced requests.
        - Lack of Personalization: Ignores user segment (e.g., Premier vs. One).
        
        Specialized Omissions:
        - Product/Service Knowledge Gaps: Unaware of specific offerings or HSBC domain knowledge.
        - Policy/Procedure Ignorance: Misstates requirements (e.g., ID types, fees).
        - Campaign/Promo Support Failure: Cannot explain or apply promo codes/offers.

        If no category is matching return "General".
        
        Return the response in JSON format with a "feedback_comment_category" field containing the most appropriate category.
        """
        
        user_prompt = f"""
        Feedback Comment: {feedback_comment}
        
        Categorize this feedback comment.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={'type': 'json_object'}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('feedback_comment_category', '')
        except Exception as e:
            print(f"Error categorizing feedback comment: {str(e)}")
            return ''
    
    async def add_feedback_comment_categories_async(self, df: pd.DataFrame, max_concurrent: int = 20) -> pd.DataFrame:
        """Add feedback_comment_category column to dataframe using async API calls"""
        if max_concurrent is None:
            max_concurrent = self.max_concurrent
            
        print(f"Adding feedback comment categories to THUMBS_DOWN records with async API calls...")
        
        # Add feedback_comment_category column if it doesn't exist
        result_df = df.copy()
        if 'feedback_comment_category' not in result_df.columns:
            result_df['feedback_comment_category'] = ''
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_comment_category(idx, row):
            # Skip non-THUMBS_DOWN rows
            if row['feedback_rating'] != 'THUMBS_DOWN':
                return idx, ''
                
            async with semaphore:
                try:
                    feedback_comment_category = await self.categorize_feedback_comment_async(row['feedback_comment'])
                    return idx, feedback_comment_category
                except Exception as e:
                    print(f"Error processing row {idx}: {str(e)}")
                    return idx, ''
        
        # Create tasks for all rows
        tasks = [
            process_single_comment_category(idx, row) 
            for idx, row in result_df.iterrows()
        ]
        
        # Process all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        print(f"Completed {len(results)} feedback comment categorization API calls in {end_time - start_time:.2f} seconds")
        
        # Update the dataframe with results
        success_count = 0
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                idx, feedback_comment_category = result
                if feedback_comment_category:
                    result_df.at[idx, 'feedback_comment_category'] = feedback_comment_category
                    success_count += 1
        
        print(f"Successfully categorized {success_count} feedback comments")
        return result_df
    
    async def process_feedback_comment_categories_only(self) -> pd.DataFrame:
        """Process only feedback comment categorization"""
        print("\n=== ADDING FEEDBACK COMMENT CATEGORIES ===")
        
        # Load data with comments
        if os.path.exists(self.comments_output_file):
            df = pd.read_csv(self.comments_output_file)
            print(f"Loaded {len(df)} records with comments")
        else:
            print(f"Error: Comments file {self.comments_output_file} not found. Please run 'Process comments only' first.")
            return pd.DataFrame()
        
        # Filter to only unprocessed records
        unprocessed_df = self.filter_unprocessed_records(df, self.comment_categories_output_file)
        
        if unprocessed_df.empty:
            print("No new records to process for feedback comment categories")
            # Return existing processed data
            if os.path.exists(self.comment_categories_output_file):
                return pd.read_csv(self.comment_categories_output_file)
            else:
                return df
        
        # Add feedback comment categories
        new_results_df = await self.add_feedback_comment_categories_async(unprocessed_df)
        
        # Merge with existing results
        final_df = self.merge_with_existing_results(new_results_df, self.comment_categories_output_file)
        
        # Save results
        self.save_results(final_df, self.comment_categories_output_file)
        
        return final_df
    
    def load_mapped_questions(self) -> set:
        """Load mapped questions from CSV file and return as a set for fast lookup"""
        try:
            mapped_df = pd.read_csv(self.mapped_questions_file)
            # Extract questions and convert to lowercase for case-insensitive matching
            questions_set = set(mapped_df['question'].str.lower().str.strip())
            print(f"Loaded {len(questions_set)} mapped questions")
            return questions_set
        except FileNotFoundError:
            print(f"Error: {self.mapped_questions_file} not found")
            return set()
        except Exception as e:
            print(f"Error loading mapped questions: {str(e)}")
            return set()
    
    def add_scenario_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add scenario column based on whether user_question exists in mapped_questions.csv"""
        print(f"Adding scenario mapping to {len(df)} records...")
        
        # Load mapped questions
        mapped_questions = self.load_mapped_questions()
        if not mapped_questions:
            print("No mapped questions loaded, setting all scenarios to 'B'")
            result_df = df.copy()
            if 'scenario' not in result_df.columns:
                result_df['scenario'] = 'B'
            return result_df
        
        # Add scenario column if it doesn't exist
        result_df = df.copy()
        if 'scenario' not in result_df.columns:
            result_df['scenario'] = ''
        
        # Map scenarios
        scenario_a_count = 0
        scenario_b_count = 0
        
        for idx, row in result_df.iterrows():
            user_question = str(row['user_question']).lower().strip()
            
            if user_question in mapped_questions:
                result_df.at[idx, 'scenario'] = 'A'
                scenario_a_count += 1
            else:
                result_df.at[idx, 'scenario'] = 'B'
                scenario_b_count += 1
        
        print(f"Scenario mapping completed:")
        print(f"  Scenario A (provided questions): {scenario_a_count} ({scenario_a_count/len(result_df)*100:.1f}%)")
        print(f"  Scenario B (open-ended questions): {scenario_b_count} ({scenario_b_count/len(result_df)*100:.1f}%)")
        
        return result_df
    
    def process_scenarios_only(self) -> pd.DataFrame:
        """Process only scenario mapping"""
        print("\n=== PROCESSING SCENARIOS ONLY ===")
        
        # Load feedback data
        df = self.load_feedback_data()
        if df.empty:
            return df
        
        # Filter to only unprocessed records
        unprocessed_df = self.filter_unprocessed_records(df, self.scenarios_output_file)
        
        if unprocessed_df.empty:
            print("No new records to process for scenarios")
            # Return existing processed data
            if os.path.exists(self.scenarios_output_file):
                return pd.read_csv(self.scenarios_output_file)
            else:
                return df
        
        # Add scenario mapping
        new_results_df = self.add_scenario_mapping(unprocessed_df)
        
        # Merge with existing results
        final_df = self.merge_with_existing_results(new_results_df, self.scenarios_output_file)
        
        # Save results
        self.save_results(final_df, self.scenarios_output_file)
        
        return final_df
    
    def merge_files(self) -> pd.DataFrame:
        """Merge comments, categories, and comment categories files into final analyzed file"""
        print("\n=== MERGING FILES ===")
        
        # Load base data
        base_df = self.load_feedback_data()
        if base_df.empty:
            return base_df
        
        result_df = base_df.copy()
        
        # Merge comments if file exists
        if os.path.exists(self.comments_output_file):
            print(f"Loading comments from {self.comments_output_file}")
            comments_df = pd.read_csv(self.comments_output_file)
            # Update feedback_comment column
            result_df['feedback_comment'] = comments_df['feedback_comment']
            print("Comments merged successfully")
        else:
            print(f"Comments file {self.comments_output_file} not found, skipping")
        
        # Merge categories if file exists
        if os.path.exists(self.categories_output_file):
            print(f"Loading categories from {self.categories_output_file}")
            categories_df = pd.read_csv(self.categories_output_file)
            # Add category and sub_category columns
            result_df['category'] = categories_df['category']
            result_df['sub_category'] = categories_df['sub_category']
            print("Categories merged successfully")
        else:
            print(f"Categories file {self.categories_output_file} not found, skipping")
        
        # Merge feedback comment categories if file exists
        if os.path.exists(self.comment_categories_output_file):
            print(f"Loading feedback comment categories from {self.comment_categories_output_file}")
            comment_categories_df = pd.read_csv(self.comment_categories_output_file)
            # Add feedback_comment_category column
            result_df['feedback_comment_category'] = comment_categories_df['feedback_comment_category']
            print("Feedback comment categories merged successfully")
        else:
            print(f"Feedback comment categories file {self.comment_categories_output_file} not found, skipping")
        
        # Merge scenarios if file exists
        if os.path.exists(self.scenarios_output_file):
            print(f"Loading scenarios from {self.scenarios_output_file}")
            scenarios_df = pd.read_csv(self.scenarios_output_file)
            # Add scenario column
            result_df['scenario'] = scenarios_df['scenario']
            print("Scenarios merged successfully")
        else:
            print(f"Scenarios file {self.scenarios_output_file} not found, skipping")
        
        # Save merged file
        self.save_results(result_df, self.merged_output_file)
        return result_df
    
    def generate_summary_report(self, df: pd.DataFrame) -> None:
        """Generate a summary report of the analysis"""
        print("\n=== FEEDBACK ANALYSIS SUMMARY ===")
        print(f"Total records: {len(df)}")
        
        # Feedback rating distribution
        rating_counts = df['feedback_rating'].value_counts()
        print(f"\nFeedback Rating Distribution:")
        for rating, count in rating_counts.items():
            print(f"  {rating}: {count} ({count/len(df)*100:.1f}%)")
        
        # Category distribution
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            print(f"\nCategory Distribution:")
            for category, count in category_counts.items():
                print(f"  {category}: {count} ({count/len(df)*100:.1f}%)")
        
        # Sub-category distribution
        if 'sub_category' in df.columns:
            sub_category_counts = df['sub_category'].value_counts()
            print(f"\nSub-Category Distribution:")
            for sub_category, count in sub_category_counts.items():
                print(f"  {sub_category}: {count} ({count/len(df)*100:.1f}%)")
        
        # Feedback comment category distribution
        if 'feedback_comment_category' in df.columns:
            feedback_comment_category_counts = df['feedback_comment_category'].value_counts()
            print(f"\nFeedback Comment Category Distribution:")
            for category, count in feedback_comment_category_counts.items():
                if category:  # Skip empty categories
                    print(f"  {category}: {count} ({count/len(df)*100:.1f}%)")
        
        # Scenario distribution
        if 'scenario' in df.columns:
            scenario_counts = df['scenario'].value_counts()
            print(f"\nScenario Distribution:")
            for scenario, count in scenario_counts.items():
                scenario_desc = "provided questions" if scenario == 'A' else "open-ended questions"
                print(f"  Scenario {scenario} ({scenario_desc}): {count} ({count/len(df)*100:.1f}%)")
        
        # Category and Sub-category combination
        if 'category' in df.columns and 'sub_category' in df.columns:
            combined_counts = df.groupby(['category', 'sub_category']).size().reset_index(name='count')
            print(f"\nCategory-SubCategory Distribution:")
            for _, row in combined_counts.iterrows():
                print(f"  {row['category']} -> {row['sub_category']}: {row['count']}")
        
        # Empty feedback comments
        empty_comments = df[df['feedback_comment'].isna() | (df['feedback_comment'].str.strip() == '')]
        print(f"\nEmpty feedback comments: {len(empty_comments)} ({len(empty_comments)/len(df)*100:.1f}%)")
        
        # THUMBS_DOWN with empty comments
        thumbs_down_empty = df[
            (df['feedback_rating'] == 'THUMBS_DOWN') & 
            (df['feedback_comment'].isna() | (df['feedback_comment'].str.strip() == ''))
        ]
        print(f"THUMBS_DOWN with empty comments: {len(thumbs_down_empty)} ({len(thumbs_down_empty)/len(df)*100:.1f}%)")
    
    async def run_full_analysis(self) -> pd.DataFrame:
        """Run the complete feedback analysis pipeline with incremental processing"""
        print("\n=== RUNNING FULL ANALYSIS ===")
        
        # Step 1: Process comments
        print("\nStep 1: Processing comments...")
        df_with_comments = await self.process_comments_only()
        if df_with_comments.empty:
            return df_with_comments
        
        # Step 2: Process categories
        print("\nStep 2: Processing categories...")
        df_with_categories = await self.process_categories_only()
        
        # Step 3: Process feedback comment categories
        print("\nStep 3: Processing feedback comment categories...")
        df_with_comment_categories = await self.process_feedback_comment_categories_only()
        
        # Step 4: Process scenarios
        print("\nStep 4: Processing scenarios...")
        df_with_scenarios = self.process_scenarios_only()
        
        # Step 5: Merge all files
        print("\nStep 5: Merging all processed files...")
        final_df = self.merge_files()
        
        # Generate summary report
        self.generate_summary_report(final_df)
        
        return final_df

async def main():
    """Main function to run the feedback analysis pipeline"""
    analyzer = FeedbackAnalyzer()
    
    print("Choose an option:")
    print("1. Process comments only (THUMBS_DOWN empty feedback)")
    print("2. Process categories only")
    print("3. Process feedback comment categories")
    print("4. Process scenarios only (map questions to A/B scenarios)")
    print("5. Merge existing files")
    print("6. Run full analysis (comments + categories + feedback comment categories + scenarios + merge)")
    
    choice = input("Enter your choice (1-6): ").strip()
    
    if choice == "1":
        result_df = await analyzer.process_comments_only()
        print(f"\nComments processing complete! Shape: {result_df.shape}")
        
    elif choice == "2":
        result_df = await analyzer.process_categories_only()
        print(f"\nCategories processing complete! Shape: {result_df.shape}")
        
    elif choice == "3":
        result_df = await analyzer.process_feedback_comment_categories_only()
        print(f"\nFeedback comment categories processing complete! Shape: {result_df.shape}")
        
    elif choice == "4":
        result_df = analyzer.process_scenarios_only()
        print(f"\nScenario processing complete! Shape: {result_df.shape}")
        
    elif choice == "5":
        result_df = analyzer.merge_files()
        analyzer.generate_summary_report(result_df)
        print(f"\nFiles merged successfully! Shape: {result_df.shape}")
        
    elif choice == "6":
        result_df = await analyzer.run_full_analysis()
        print(f"\nFull analysis complete! Shape: {result_df.shape}")
        
    else:
        print("Invalid choice. Please run the script again.")
        return
    
    # Display first few rows if data exists
    if 'result_df' in locals() and not result_df.empty:
        print("\nFirst 3 rows of processed data:")
        print(result_df.head(3))

if __name__ == "__main__":
    asyncio.run(main())
