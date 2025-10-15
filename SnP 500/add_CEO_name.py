from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import time
from logGenerator import log_generator
import logging
import os
#from ETL.load import load_dataset
# ceo column will be added to the s&p500 companies.csv
# run commnad : python -m "SnP 500.add_CEO_name"
class CEODataCollector:
    def __init__(self, list_500, log_file_path, smp_500_path, output_file):
        load_dotenv()
        # Initialize logging
        self.__log_gen = log_generator(log_file_path)
        self.__log_gen.log_config()
        self.__list_500 = os.getenv(list_500)
        self.__output_path = os.getenv(smp_500_path)
        self.__output_file = output_file
        
        # Initialize data
        self.__df = None
        self.__successful_fetches = 0
        self.__failed_fetches = 0

    def load_list_500(self):
        """Load S&P 500 companies list"""
        self.__df = pd.read_csv(self.__list_500)
        print(f"📊 Loaded {len(self.__df)} companies from S&P 500 list")
        logging.info(f"Starting CEO data collection for {len(self.__df)} companies")

    def add_ceo_column(self):
        """Add CEO column to dataframe"""
        self.__df['CEO'] = ""

    def fetch_ceo_name(self, symbol, index):
        """Fetch CEO name for a specific symbol"""
        try:
            # Get company info from yfinance
            ticker = yf.Ticker(symbol)
            officers = None
            if hasattr(ticker, "get_company_officers"):
                officers = ticker.get_company_officers()
                source = "get_company_officers"
            else:
                # ✅ Fallback to legacy get_info()
                info = ticker.get_info()
                officers = info.get("companyOfficers", [])
                source = "get_info"
            if not officers:
                self.__df.at[index, "CEO"] = "N/A"
                self.__failed_fetches += 1
                print(f"⚠️ No officer data found for {symbol}")
                logging.warning(f"No CEO data found for {symbol}")
                return

            ceo_name = "N/A"
            for officer in officers:
                title = officer.get("title", "").lower()
                # ✅ Match flexible CEO titles
                if any(keyword in title for keyword in [
                "chief executive",
                "Chief Executive Officer",
                "Chief Executive", # covers "chief executive officer"
                           # short form
                "president & ceo",
                "ceo",           # e.g., "ceo and chairman"
                "executive director",
                "ceo and president",
                "ceo and chairman",
                'chief executive officer',
                'chairman',
                'CEO',
                'President & CEO',
                'President & Chairman',
                'President & Chairman & CEO',
                'President & Chairman & CEO',
                'Owner',
                'Owner & CEO',
                'Owner & Chairman',
                'Owner & Chairman & CEO',
                'Owner & Chairman & CEO',
                
                 # some global firms use this instead
            ]   ):
                    ceo_name = officer.get("name", "N/A")
                    break


            self.__df.at[index, "CEO"] = ceo_name
            if ceo_name != "N/A":
                self.__successful_fetches += 1
                print(f"✅ {symbol} : {ceo_name}")
                logging.info(f"Fetched CEO for {symbol}: {ceo_name}")
            else:
                self.__failed_fetches += 1
                print(f"⚠️ CEO not listed for {symbol}")
                logging.warning(f"CEO not found for {symbol}")
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            self.__df.at[index, 'CEO'] = "Error"
            self.__failed_fetches += 1
            print(f"  ❌ Error fetching CEO for {symbol}: {e}")
            logging.error(f"Failed to fetch CEO for {symbol}: {e}")
            
            # Add delay for failed requests too
            time.sleep(1)

    def process_all_companies(self):
        """Process all companies to fetch CEO data"""
        for index, row in self.__df.iterrows():
            symbol = row['Symbol']
            company_name = row['Security']
            
            print(f"[{index + 1}/{len(self.__df)}] Fetching CEO for {symbol} - {company_name}")
            logging.info(f"Fetching CEO for {symbol} - {company_name}")
            
            self.fetch_ceo_name(symbol, index)

    def save_to_csv(self):
        """Save data to CSV file"""
        # Save to new file
        output_file = os.path.join(str(self.__output_path), str(self.__output_file))
        self.__df.to_csv(output_file, index=False)
        
        print(f"\n🎉 Process completed!")
        print(f"📁 Saved to: {output_file}")
        print(f"✅ Successful fetches: {self.__successful_fetches}")
        print(f"❌ Failed fetches: {self.__failed_fetches}")
        print(f"📊 Total companies: {len(self.__df)}")
        
        logging.info(f"CEO data collection completed. Success: {self.__successful_fetches}, Failed: {self.__failed_fetches}")
        
        # Display sample results
        print(f"\n📋 Sample Results:")
        sample_df = self.__df[['Symbol', 'Security', 'CEO']].head(10)
        print(sample_df.to_string(index=False))
        
        return self.__df

    def add_ceo_names_to_sp500(self):
        """Main method to add CEO names to S&P 500 companies"""
        self.load_list_500()
        self.add_ceo_column()
        self.process_all_companies()
        return self.save_to_csv()

if __name__ == "__main__":
    try:
        # Add CEO names to S&P 500 data
        collector = CEODataCollector("SMP_500", "LOG_FILE_PATH", "SMP_500_PATH", "full_data.csv")
        df_with_ceos = collector.add_ceo_names_to_sp500()
        
        logging.info(f"✅ Full CEO list saved!")
        
    except Exception as e:
        print(f"❌ Error in main execution : {e}")
        logging.error(f"Main execution error : {e}")