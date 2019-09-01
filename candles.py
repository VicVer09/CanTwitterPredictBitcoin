from binance.client import Client 
import datetime
import pickle
import queue
import sys

# NOTE THIS FILE WILL NOT WORK WITHOUT THESE VARIABLE SET TO A VALID SET OF TOKENS AN KEYS
token = ''
key = ''
client = Client(token , key)
trade_symbol = 'BTCUSDT'

# month, days are integer, day_end is inclusive
# translator for the API call client.get_klines
def get_klines(month, day_start, day_end):

	candles = []

	klimit = 480 # 480 minutes = 8 hours, each 
	thirds_of_day = ['00', '08', '16'] # based on hour of the day ie midnight, etc

	for day in range(day_start, day_end + 1):

		for third in range(0,3):

			dt_obj = datetime.datetime.strptime(str(day) + '.'+ str(month) +'.2018 ' + thirds_of_day[third] + ':00:00,00',
					                           	'%d.%m.%Y %H:%M:%S,%f')
			timestamp = (int)(dt_obj.timestamp() * 1000)

			candles = candles + client.get_klines(symbol=trade_symbol, 
										interval=Client.KLINE_INTERVAL_1MINUTE, 
										limit = klimit, 
										endTime = timestamp)

	return candles

def main():
	print("Acquiring Candles")
	candles = get_klines(12, 1, 14)
	print("Dumping")
	pickle.dump(candles, open('candles/Dec_1_7_1MIN.p', 'wb')) 
	print("Done")

if __name__ == "__main__":
	main()