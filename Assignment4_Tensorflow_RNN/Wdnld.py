import urllib
import tarfile
   
url = "https://github.com/kj7kunal/Deep_Learning_Assignments/blob/master/Assignment4_Tensorflow_RNN/weights.tar.gz"
urllib.urlretrieve(url, filename="./weights.tar.gz") 
tar = tarfile.open("./weights.tar.gz", "r:gz")
tar.extractall()
tar.close()
print "download complete!"
