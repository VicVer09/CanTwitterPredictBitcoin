import tweepy
import time
import sys
from pprint import pprint
import pickle

# NOTE THIS FILE WILL NOT WORK WITHOUT THESE VARIABLE SET TO A VALID SET OF TOKENS AN KEYS
consumer_token = ''
consumer_secret = ''
access_token = ''
access_token_secret = '' 

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth) 

tweets = []
query = 'bitcoin'
chunk_id = 0
chunk_size = 1000

class MyStreamListener(tweepy.StreamListener): 
	def on_status(self, status):
		text = status.text

		is_retweet = False
		
		try: # get extended tweet if over 140 characters
			if hasattr(status, 'extended_tweet'):
				text = status.extended_tweet['full_text']

			elif hasattr(status,'retweeted_status'): 
				#is_retweet = True
				retweet = status.retweeted_status
				if hasattr(retweet, 'extended_tweet'):
					text = retweet.extended_tweet['full_text']
		except:
			pass
		
		global tweets
		global chunk_id
		tweets.append([text, status.created_at, status.user.followers_count, status.user.verified, status.lang])
		
		print(text) 
		print('Created at:' + str(status.created_at)) 
		print()

		if len(tweets) > chunk_size: 

			print("CHUNK SIZE REACHED, SAVING", chunk_id)
			pickle.dump(tweets, open('unfiltered_data/'+query+'_'+str(chunk_id)+'.pkl',"wb"))
			chunk_id += 1 
			tweets = []

	def on_error(self, status_code):
		if status_code == 420: 
			return False


myStreamListener = MyStreamListener()



def main():
	global tweets
	global chunk_id
 
	if len(sys.argv) >= 2:
		global query
		query = sys.argv[1]
		if (len(sys.argv)) >= 3: 
			chunk_id = int(sys.argv[2])
	
	while True:
		try:
			if len(tweets) > chunk_size: # just in case
				print("UPDATING GLOBALS AFTER CRASH")
				tweets = []
				chunk_id += 1 

			myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener, tweet_mode='extended') 
			myStream.filter(track=[query])
		except Exception as e: 
			print(e)
			print('Stream crash, restarting')
			time.sleep(1)  


if __name__ == "__main__":
	main()