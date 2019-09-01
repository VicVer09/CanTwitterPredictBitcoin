import sys
from pprint import pprint
import pickle
import datetime
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# KLINE ARRAY INDICES
OPEN_TIME = 0
OPEN = 1
HIGH = 2
LOW = 3
CLOSE = 4
VOLUME = 5
CLOSE_TIME = 6
QUOTE_ASSET_VOLUME = 7
NUMBER_OF_TRADES = 8
TAKER_BUY_BASE_ASSET_VOLUME = 9
TAKER_BUY_QUOTE_ASSET_VOLUME = 10
IGNORE = 11



def get_timestamp(created_at):
	# ignore seconds since we are using 1 minute candlesticks
	created_at -= datetime.timedelta(hours=5) # tweets are in GMT -5
	minute = created_at.minute
	hour = created_at.hour 
	day = created_at.day
	month = created_at.month
	year = created_at.year
	dt_obj = datetime.datetime.strptime(str(day) + '.'+ str(month) +'.'+str(year)+' ' + str(hour) + ':'+str(minute)+':00,00',
			                           	'%d.%m.%Y %H:%M:%S,%f')
	timestamp = (int)(dt_obj.timestamp() * 1000)

	return timestamp

chars = set('#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
def filter_text(text):

	# massage the whitespaces
	text = text.lower() # send to lowercase
	text = text.replace(u'\xa0', u' ') # removes non-breaking space in Latin1
	text = text.replace('\n', ' ')
	text = text.replace(',', ' ')
	text = text.replace('#', ' ') # unhashtag the hashtags
	text = text.replace('\\n', ' ').strip()
	
	# remove urls
	words = text.split(' ')
	for word in words:
		if 'http' in word: # remove links
			words.remove(word)
		elif '@' in word:  # remove tagged users
			words.remove(word)

	words = list(set(words)) # remove duplicates

	#print('begin filtering')
	# filter non-characters
	final_words = []
	for word in words:
		if word == '' or word == 'rt' or word == 'bitcoin': continue
		if word in stop_words: continue
		flag = True
		for letter in word:

			if letter not in chars:
				flag = False
				break

		if (flag): final_words.append(word)
	return ' '.join(final_words)

def load_tweets(chunk_start, chunk_end):

	# initialize 
	candles_id = 0 # corresponding to candles list  
	first_iteration = True # flag for "catching up" the candles timestamp with the tweet timestamp
	prev_timestamp = 0
	X = [] # input data matrix
	Y = [] # possible output vectors
	x = ['',0,0,0] # single vector 
	candles = pickle.load(open('candles/Dec_1_7_1MIN.p', 'rb')) # bitcoin candlestick data
	# assume that the candle period covers the tweet period
	#print(len(candles),'len(candles)')
	# loop through all chunks and create 
	for chunk_id in range(chunk_start, chunk_end + 1):

		tweets = None
		print('Processing '+ str(int((float(chunk_id)/(float(chunk_end) - float(chunk_start)))*100.0))+'% Complete', end="\r")
		if chunk_id > chunk_end: break 

		try:
			tweets = pickle.load(open('unfiltered_data/bitcoin_'+str(chunk_id)+'.pkl',"rb"))
		except:
			print('load fail')
			break 

		for tweet in tweets:
			
			text = tweet[0]
			created_at = tweet[1]
			followers_count = tweet[2]
			verified = tweet[3]
			lang = tweet[4] # new 

			timestamp = get_timestamp(created_at)
			if first_iteration: 

				prev_timestamp = timestamp
				while timestamp > int(candles[candles_id][OPEN_TIME]):
					candles_id += 1

				first_iteration = False

			elif timestamp > prev_timestamp: # change in timestamp -> start new datapoint, dump old
				x[0] = filter_text(x[0])

				X.append(x)  # add to matrix
				x = ['',0,0,0] # reset x vector
				prev_timestamp = timestamp

				delta_1  = float(candles[candles_id + 1 ][CLOSE]) - float(candles[candles_id][OPEN])
				delta_5  = float(candles[candles_id + 5 ][CLOSE]) - float(candles[candles_id][OPEN])
				delta_60 = float(candles[candles_id + 60][CLOSE]) - float(candles[candles_id][OPEN])
				volume = float(candles[candles_id][VOLUME])
				y = [delta_1, delta_5, delta_60, volume] 
				Y.append(y) 

				# let candles catch up in case there were no tweets during a particular timestamp (data flow)
				while timestamp > int(candles[candles_id][OPEN_TIME]):
					candles_id += 1 
				#print(candles[candles_id][OPEN_TIME] == timestamp) # True for discontinuous data too!

			# add tweet text to hive test and check other params
			if lang == 'en': x[0] += ' ' + text
			x[1] += followers_count
			if verified:
				x[2] += 1
			x[3] += 1 # total tweet count 
		#print((candles[candles_id][OPEN_TIME] == timestamp)) # verify that timestamps match 
	print('Processing Complete     \n')
	return X, Y








def main():

	print("***************************")
	print("******* HIVE METHOD *******")
	print("***************************\n")
	
	chunk_start = 0 # 0, or potentially 276 if we wish to only 
					# include full_text from long tweets, and only include continuous data
	chunk_end = 427 # initialize to highest file count in unfiltered_data

	# override with args if necessary
	# command line: python3 hive_method.py <chunk_start> <chunk_end>
	if len(sys.argv) >= 2:
		chunk_start = int(sys.argv[1])
		if (len(sys.argv)) >= 3: 
			chunk_end = int(sys.argv[2])
	
	# call big boy
	X, Y = load_tweets(chunk_start, chunk_end)

	# ill give u a nickel to tickle my pickle
	pickle.dump(X, open('data/hive_X.pkl',"wb"))
	pickle.dump(Y, open('data/hive_Y.pkl',"wb"))



if __name__ == "__main__":
	main()






