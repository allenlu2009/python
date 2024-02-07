#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
import logging
import requests
from datetime import datetime
import opencc
from googletrans import Translator


def parse_args():
    parser = argparse.ArgumentParser(description='Parse a markdown file, find the figure links, \
    replace the brackets with mathjax and remove the http link.')
    parser.add_argument('file', help='the markdown file to be parsed')
    parser.add_argument('-p', '--path', help='the path of the output file')
    parser.add_argument('-o', '--output', help='the output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    args = parser.parse_args()
    return args


# read a markdown file use utf-8 encoding
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

# write a markdown file use utf-8 encoding
def write_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return


''' replace the empty code block with python using gpt code generation'''
def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    content = read_file(args.file)
    

    # create an instance of the Translator class
    translator = Translator()

    # define the text to be translated
    # text = "Hello, how are you?"
    text = "In the context of the googletrans library, zh-cn is used to represent the language code for Simplified Chinese, which is the standardized writing system used in mainland China. Other variants of written Chinese include Traditional Chinese, which is used in Taiwan, Hong Kong, and other regions, as well as different regional dialects and minority languages."
    #text="測試"
    content = " Recomendation system architecture demonstrating the funnel where candidate video are retrieved and ranked before presneting only a few to the user."

    # translate the text to Chinese
    result = translator.translate(content, src='en', dest='zh-tw')

    # print the translated text
    print(result.text)


    # write file with output path and filename
    if args.output:
        if args.path:
            args.output = os.path.join(args.path, args.output)            
        write_file(args.output, result.text)
    else:
        print(result.text)


# call the main function
if __name__ == '__main__':
    main()
