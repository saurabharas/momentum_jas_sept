from flask import Flask, request
import os
import datetime

app = Flask(__name__)

def log_name():
	# logs will be saved in files with current date
	return datetime.datetime.now().strftime("%Y-%m-%d") + '.txt'

@app.route('/post', methods=['POST'])
def post():
	# post back json data will be inside request.get_data()
	# as an example here it is being stored to a file
	f = open(log_name(),'a+')
	f.write(str(request.get_data())+'\n')
	f.close()
	return 'done'

@app.route('/')
def index():
	# show the contents of todays log file
	if not os.path.exists(log_name()):
		open(log_name(), 'a+').close()

	return open(log_name()).read()

app.run(debug=True, host='0.0.0.0', port=80)

# if you have your own ssl certificates place them in this directory
# use this statement to enable https for postbacks
# app.run(debug=True, '0.0.0.0', port=443, ssl_context=('cert.pem', 'key.pem'))