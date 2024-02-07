# create a data crawler to crawl the data from a static website 
# and save the data to a local file

import requests
from bs4 import BeautifulSoup
import os
import re
import time
import random

# get the html content of the website
def get_html(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return "Error"

# get the title of the website
def get_title(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find('title').text
    return title

# get the content of the website
def get_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', class_ = 'content')
    return content

# get the links of the website
def get_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all('a', href = re.compile(r'^/'))
    return links

# process title to remove space and special characters including return for a legal filename
def process_title(title):
    title = title.replace(' ', '_')
    title = title.replace('\r', '_')
    title = title.replace('\n', '_')
    title = title.replace('\t', '_')
    title = title.replace('/', '_')
    title = title.replace('\\', '_')
    title = title.replace(':', '_')
    title = title.replace('*', '_')
    title = title.replace('?', '_')
    title = title.replace('"', '_')
    title = title.replace('<', '_')
    title = title.replace('>', '_')
    title = title.replace('|', '_')
    return title

# save the data to a local file
def save_data(url, title, content):
    path = 'data/'
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path + title + '.txt', 'w', encoding = 'utf-8') as f:
        f.write(url + '')
        f.write(title + '')
        if content:
            f.write(content.text + '')

# main function
def main():
    #url = 'http://www.jikexueyuan.com/course/494.html'
    url = 'https://browser.geekbench.com/android-benchmarks'
    html = get_html(url)
    title = process_title(get_title(html))
    content = get_content(html)
    save_data(url, title, content)
    links = get_links(html)
    for link in links:
        print(link)
        time.sleep(random.random() * 5)

if __name__ == '__main__':
    main()
