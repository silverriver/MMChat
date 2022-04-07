import hashlib
import sys
from urllib.parse import urlparse


def url2imgid(url):
    url_split = urlparse(url).path.split('.')
    if len(url_split) == 1:
        extend = 'jpg'           # default extension
    else:
        extend = url_split[-1]   # get the original extension
    return hashlib.sha256(url.encode(encoding='UTF-8')).hexdigest() + "." + extend


if __name__ == '__main__':
    url_in = sys.argv
    for u in url_in[1:]:
        print(url2imgid(u))
