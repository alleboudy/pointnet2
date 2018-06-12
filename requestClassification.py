import argparse
import requests
#get_response = requests.get(url='http://127.0.0.1:5000/')

# POST some form-encoded data:

parser = argparse.ArgumentParser()
parser.add_argument('--ply_path', default='', help='ply file to classify')
FLAGS = parser.parse_args()

testFile = open(FLAGS.ply_path, 'rb')

post_response = requests.post(url='http://192.168.178.58:5070/', data=testFile)
print(post_response.text)