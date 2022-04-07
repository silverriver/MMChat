'''
Download images based on the provided urls
Author: Silver
'''
import json
import os
import random
import threading
from tqdm import tqdm, trange
import time
import urllib3
from requests.adapters import HTTPAdapter
import requests
import logging
import argparse
from url2imgid import url2imgid

urllib3.disable_warnings()


# 可以重复，但是不能为空
class DownloadImage:
    def __init__(self, out_path, img_url_file, proxies, header, retries, timeout):
        self.header = header
        self.proxies = proxies
        self.out_path = out_path
        self.retries = retries
        self.timeout = timeout
        self.img_url_file = img_url_file

        self.file_path = os.path.join(out_path, 'image')
        if not os.path.isdir(self.file_path):
            os.makedirs(self.file_path)

        self.url_json = os.path.join(out_path, 'urls.json')   # urls that have been downloaded
        self.failedpath = os.path.join(out_path, 'failedurl.json')

        self.successurl = []
        self.failedUrl = []

        logfilepath = os.path.join(out_path, 'errors.log')
        formater = logging.Formatter(
            'Time:%(asctime)s Level: %(levelname)s URL: %(URL)s STATSUSCODE: %(STATSUSCODE)s MESSAGE: %(message)s')
        self.logger = logging.getLogger()
        fileHandler = logging.FileHandler(logfilepath)
        fileHandler.setFormatter(formater)
        self.logger.addHandler(fileHandler)
        self.logger.setLevel(logging.ERROR)

        self.urllist = self.read_url()

    def read_url(self):
        """
        read weibo_img_url file, remove urls that have been downloaded
        """
        # read input urls
        raw_urls = set()
        with open(self.img_url_file, "r", encoding="utf-8")as f:
            res = [json.loads(i) for i in f.readlines()]
            for weibo in res:
                if not weibo["weibo_img"]:
                    continue
                for url in weibo['weibo_img'].split(';'):
                    raw_urls.add(url)
        if "" in raw_urls:
            raw_urls.remove("")

        if os.path.exists(self.url_json):
            # read existing urls
            with open(self.url_json, "r", encoding="utf-8")as f:
                existing_urls = set([i.strip() for i in f.readlines() if len(i.strip()) != 0])
            print("{} imgs downloaded".format(len(existing_urls)))

            if len(raw_urls) > len(existing_urls):
                return list(raw_urls - existing_urls)
            else:
                return None
        else:
            return list(raw_urls)

    def get_proxies(self):
        if self.proxies is not None:
            return random.choice(self.proxies)
        else:
            return None

    def set_request(self):
        s = requests.Session()
        s.proxies = self.get_proxies()
        s.mount('https://', HTTPAdapter(max_retries=self.retries))
        s.keep_alive = False
        return s

    def getDownload(self, url_list):
        if url_list:
            for i, url in enumerate(url_list):
                try:
                    down_res = self.set_request().get(
                        url, timeout=self.timeout, verify=False, allow_redirects=False)
                    if down_res.status_code == 200:
                        filenamepath = os.path.join(self.file_path, url2imgid(url))
                        # image file
                        with open(filenamepath, "wb")as fb:
                            fb.write(down_res.content)
                        # sucess url  TODO: possible bug, mutli-thread write to the same file
                        with open(self.url_json, "a+", )as fp:
                            fp.write(url + "\n")
                        self.successurl.append(url)
                    else:
                        self.failedUrl.append(url)
                        self.logger.error(down_res.reason, extra={'URL': url, "STATSUSCODE": down_res.status_code})
                except requests.exceptions.RequestException as e:
                    self.failedUrl.append(url)
                    self.logger.error(e, extra={'URL': url, "STATSUSCODE": ""})
        else:
            tqdm.write("下载完成")

    def dump_failed_url(self):
        for url in self.failedUrl:
            with open(self.failedpath, "w", encoding="utf-8")as fp:
                fp.write(url + "\n")
        print("失败url存储在failedurl.json文件中，共{}条".format(len(self.failedUrl)))

    def run(self):
        with trange(len(self.urllist), dynamic_ncols=True) as t:
            for i in t:
                time.sleep(0.3)
                t.set_description("FAILED NUM {}，SUCCESS NUM {}".format(len(self.failedUrl), len(self.successurl)))
                thread = threading.Thread(target=self.getDownload, args=([self.urllist[i]],))
                thread.start()
        self.dump_failed_url()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url_file', help='weibo index file', default='weibo_img_expanded_url_unique_weiboid.json')
    parser.add_argument('--out_dir', help='output dir', default='weibo_img')
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    proxies = None
    image = DownloadImage(out_path=args.out_dir, img_url_file=args.url_file, proxies=proxies, header=None, retries=3, timeout=90)
    image.run()
    print("fin.")
