import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import os
from logGenerator import log_generator
import logging
#from ETL.load import load_dataset
#from load_dataset import load_dataset
## run commnad : python -m "SnP 500.snp500"
class snp_500:
    def __init__(self,log_file_path, list_500, base_dir, period):
        load_dotenv()
        self.__logGenerator = log_generator(log_file_path)
        self.__logGenerator.log_config()  # Initialize logging
        self.__list_500 = os.getenv(list_500)
        self.__base_dir = os.getenv(base_dir)
        self.__period = period
        self.__df = None
        self.__successful_fetches = 0
        self.__failed_fetches = 0
    
    def load_SNP500_list(self):
        self.__df = pd.read_csv(self.__list_500)
        print(f"✅ Loaded {len(self.__df)} companies from {self.__list_500}")
        logging.info(f"Loaded S&P 500 list with {len(self.__df)} companies data")

    def _create_folder_for_each_company(self, company_name: str) -> str:
        """Create a clean folder name for each company."""
        folder_name = "".join(c if c.isalnum() or c == " " else "" for c in company_name).strip()
        folder_path = os.path.join(self.__base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def save_to_csv(self, data: pd.DataFrame, csv_path: str, symbol: str):
        """Save clean CSV data for a company."""
        # Flatten columns if MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[1] if col[1] else col[0] for col in data.columns]

        data.reset_index(inplace=True)

        # Add 'Price' column
        if "Adj Close" in data.columns:
            data["Price"] = data["Adj Close"]
        elif "Close" in data.columns:
            data["Price"] = data["Close"]
        else:
            print(f"⚠️ Skipping {symbol}: no 'Adj Close' or 'Close' column found")
            logging.error(f"Skipping {symbol}: no 'Adj Close' or 'Close' column found")
            return

        # Reorder columns
        final_order = ["Price", "Close", "High", "Low", "Open", "Volume", "Date"]
        data = data[final_order]

        data.to_csv(csv_path, index=False)
        print(f"✅ Saved {symbol} data ({len(data)} rows)")

    def download_all(self):
        """Download and save data for all S&P 500 companies."""
        if self.__df is None:
            raise RuntimeError("Call load_sp500_list() before download_all()")

        logging.info("Starting S&P 500 data download process")
        
        for i, row in self.__df.iterrows():
            company_name = row["Security"]
            symbol = row["Symbol"]

            folder_path = self._create_folder_for_each_company(company_name)
            csv_path = os.path.join(folder_path, f"{symbol} {self.__period}.csv")

            print(f"[{i + 1}/{len(self.__df)}] {symbol} → {csv_path}")
            logging.info(f"Processing {i + 1}/{len(self.__df)} : {symbol} - {company_name} - {self.__period}")

            try:
                data = yf.download(symbol, period=self.__period, group_by='ticker')

                if data.empty:
                    self.__failed_fetches += 1
                    print(f"⚠️ No data returned for {symbol}")
                    logging.error(f"No data returned for {symbol}")
                    continue

                self.save_to_csv(data, csv_path, symbol)
                self.__successful_fetches += 1
                logging.info(f"Successfully saved {symbol} data ({len(data)} rows)")

            except Exception as e:
                self.__failed_fetches += 1
                print(f"❌ Failed to fetch {symbol}: {e}")
                logging.error(f"Failed to fetch {symbol}: {e}")

        print("\nAll companies processed!")
        print(f"📊 Total companies : {len(self.__df)}")
        print(f"✅ Successful fetches : {self.__successful_fetches}")
        print(f"❌ Failed fetches : {self.__failed_fetches}")
        logging.info("S&P 500 data download process completed")


    
crawler = snp_500("LOG_FILE_PATH","SMP_500", "RAW_DATA_PATH", "5y")
crawler.load_SNP500_list() #this is not a function, it is a method
crawler.download_all()
