# parse a markdown file, find the figure links, replace the brackets with mathjax and remove the http link.and

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
import logging
import requests
from datetime import datetime
import shutil
import opencc


def parse_args():
    parser = argparse.ArgumentParser(description='Parse a markdown file, find the figure links, \
    replace the brackets with mathjax and remove the http link.')
    parser.add_argument('file', help='the markdown file to be parsed')
    parser.add_argument('-p', '--path', help='the path of the output file')
    parser.add_argument('-o', '--output', help='the output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    args = parser.parse_args()
    return args


# convert the simplified chinese to traditional chinese using opencc
def convert_simplified_chinese_to_traditional_chinese(content):
    #cc = opencc.OpenCC('s2t.json')
    cc = opencc.OpenCC('s2t')
    content = cc.convert(content)
    return content


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


# find the code block and the programming language in the markdown file
def find_code_block(content):
    # non-greedy match
    pattern = re.compile(r'(```.*?```)', re.DOTALL)
    # greedy match
    # pattern = re.compile(r'(```.*```)', re.DOTALL)
    code_blocks = pattern.findall(content)
    return code_blocks

# find the progamming language in each code block
def find_code_block_language(code_blocks):
    # list of programming language
    languages = []
    pattern = re.compile(r'```(\w+)')
    # check each code block to find the programming language
    for code_block in code_blocks:
        language = pattern.findall(code_block)
        if language:
            languages.append(language[0])
        else:
            languages.append('')
    return languages
            

# remove empty ![img]() in the markdown file
def remove_empty_img(content):
    # regex of ![img]()
    pattern = re.compile(r'\!\[img\]\(\s*\)')
    content = pattern.sub('', content)
    return content


# find the figure links in the markdown file
def find_figure_links(content):
    # non-greedy match
    pattern = re.compile(r'(\!\[.*?\])(\(http.*?\))')
    # greedy match
    # pattern = re.compile(r'\!\[(.*)\]\((http.*)\)')
    #links = pattern.findall(content)
    #return links
    
    # find all the figure key of the 1st group and the http link of the 2nd group
    figure_key, http_links = zip(*pattern.findall(content))
    return figure_key, http_links


# find the img links in the markdown file, pattern <img src="">
def find_img_links(content):
    # non-greedy match
    pattern = re.compile(r'<img src="(.*?)" alt="(.*?)" style="(.*?)" />')
    
    # find all the figure key of the 1st group and the http link of the 2nd group
    src, alt, style_a = zip(*pattern.findall(content))
    return src, alt


# find the img links in the markdown file, ![-w376](media/15690584867087/15696871816594.jpg)
def find_w_img_links(content):
    # non-greedy match
    pattern = re.compile(r'(\!\[.*?\])\((.*)\)')
    #pattern = re.compile(r'(\!\[.*?\])(\(http.*?\))')
    
    # find all the figure key of the 1st group and the http link of the 2nd group
    label, img_path = zip(*pattern.findall(content))
    print(img_path)
    return label, img_path


# grep the http image and save it to the local AND replace the http link with the local link
def grep_http_image(content, links):
    args = parse_args()
    for link in links:       
        # extract the url from the link
        url = link.split('(')[1].split(')')[0]
        # grep the http image and save the image to the local directory
        img_data = requests.get(url).content
        # creat the image filename = "image-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        img_filename = 'image-' + datetime.now().strftime("%Y%m%d%H%M%S")
        # append miniseconds and fill left 0 to the filename to avoid the same filename
        img_filename = img_filename + str(round(datetime.now().microsecond / 1000)).zfill(3)
        img_fullname = img_filename + '.png'
        # check the path
        if args.path:
            img_fullname = os.path.join(args.path, img_fullname)
        with open(img_fullname, 'wb') as handler:   
            handler.write(img_data)
        content = content.replace(link, r'<img src="/media/%s" alt="%s" style="zoom:60%%;" />' % (img_fullname, img_filename))
    return content


# copy files from the source path to the destimation path, 
# skip and warn if the destimation file already exists; skip and warn if the source file does not exist
def copy_src_to_dst(links):
    for link in links:
        # create the complete source file = src_path + link 
        src_path = '/Users/allenlu/Library/Mobile Documents/iCloud~com~coderforart~iOS~MWeb/Documents/mweb_documents_library/docs/'
        src_file_with_path = src_path + link

        # check if the file exists
        if not os.path.exists(src_file_with_path):
            print('file not exists: %s' % src_file_with_path)
            continue
        
        # creat the complete destination file = dst_path + link
        dst_path = '/Users/allenlu/OneDrive/allenlu2009.github.io/'
        dst_file_with_path = dst_path + link
        
        # extract the path of the file exist, if not, create a folder
        dst_file_path = os.path.dirname(dst_file_with_path)
        if not os.path.exists(dst_file_path):
            os.makedirs(dst_file_path)

        # check if the file exists
        if os.path.exists(dst_file_with_path):
            print('destimation file already exists: %s' % dst_file_with_path)
            continue

        # copy the source file to the destination file
        shutil.copy(src_file_with_path, dst_file_with_path)

# replace ![wxxx] format to <img src="/media/xxx" alt="wxxx" style="zoom:60%%;" /> format
def replace_wimg_to_src_img(content, wimg_file):
    for file in wimg_file:       
        # remove the prefix path of the file
        file_p = file.split('/')[-1]
        # remove the ending file extension of file
        alt_name = file_p.split('.')[0]
        # replace the file with (file)
        file_w_paranthesis = "(%s)" % file
        content = content.replace(file_w_paranthesis, r'<img src="/%s" alt="image-%s" style="zoom:60%%;" />' % (file, alt_name))  
    return content

# remove the figure links
def remove_figure_http_links(content, links):
    for link in links:
        # remove the links
        content = content.replace(link, '')
        #content = content.replace(link, r'\(%s\)' % link)
    return content

# replace the figure key with mathjax
def replace_figure_keys(content, keys):
    for key in keys:
        # replace the figure key with (1) remove !, (2) replace [ ] with \$\$
        new_key = key.replace('!', '')
        new_key = new_key.replace('[', '$')
        new_key = new_key.replace(']', '$')
        content = content.replace(key, new_key)
    return content

# remove the figure keys from the content, using the local image link
def remove_figure_keys(content, keys):
    for key in keys:
        # remove the figure key
        content = content.replace(key, '')
    return content

# remove the http link
def remove_http_link(content):
    pattern = re.compile(r'\((http.*)\)')
    content = pattern.sub(r'()', content)
    return content


def replace_http_img_with_local_link():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    content = read_file(args.file)
    content = remove_empty_img(content)
    figure_key, figure_http_links = find_figure_links(content)
    
    GREP_HTTP_IMG = True
    if GREP_HTTP_IMG:  ## http link is image.  grep the http image link with local image link
        content = grep_http_image(content, figure_http_links)
        content = remove_figure_keys(content, figure_key)
    else:   # http link is math. replace the figure key with mathjax
        content = remove_figure_http_links(content, figure_http_links)
        content = replace_figure_keys(content, figure_key)
    
    # write file with output path and filename
    if args.output:
        if args.path:
            args.output = os.path.join(args.path, args.output)            
        write_file(args.output, content)
    else:
        print(content)
    

# write copy_local_img_to_media
def copy_local_img_to_media():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        
    content = read_file(args.file)

    # find the <img src> link in the content
    # src, alt = find_img_links(content)
    # find the ![w376](media/15690584867087/15696871816594.jpg) link in the content
    wlabel, wimg_file = find_w_img_links(content)
    
    # copy the wimg_file from source path to the destimation path
    copy_src_to_dst(wimg_file)
    
    # replace wimg_file to <img src=xxx alt=xxx zoom=60%%;/>
    content = replace_wimg_to_src_img(content, wimg_file)
    
    # remove wlabel ![w]
    content = remove_figure_keys(content, wlabel)

    # write file with output path and filename
    if args.output:
        if args.path:
            args.output = os.path.join(args.path, args.output)            
        write_file(args.output, content)
    else:
        print(content)
    
    

''' my code, don't use it'''
def replace_empty_code_block_with_python():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    content = read_file(args.file)
    
    # find the code block. If there is no associated programming language, use the default language
    code_blocks = find_code_block(content)
    # find the progamming language in the code block
    languages = find_code_block_language(code_blocks)

    # replace the code block without programming language with python and skip the one with programming language
    for i, language in enumerate(languages):
        if not language:
            # insert python to the code_block after the first 3 backticks
            code_block_new = code_blocks[i][:3] + 'python' + code_blocks[i][3:]
            content = content.replace(code_blocks[i], code_block_new)
    
    # write file with output path and filename
    if args.output:
        if args.path:
            args.output = os.path.join(args.path, args.output)            
        write_file(args.output, content)
    else:
        print(content)


''' convert simplifed Chinese to traditional Chinese '''
def markdown_convert_chinese():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    content = read_file(args.file)
    
    # convert the content to traditional Chinese
    content = convert_simplified_chinese_to_traditional_chinese(content)
    
    # write file with output path and filename
    if args.output:
        if args.path:
            args.output = os.path.join(args.path, args.output)            
        write_file(args.output, content)
        print("convert to traditional Chinese successfully!")
    else:
        print(content)


''' replace the empty code block with python using gpt code generation'''
def replace_empty_code_block_with_python_gpt():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    content = read_file(args.file)
    
    # Find all code blocks
    old_code_blocks = re.findall(r'```(.*?)\n(.*?)\n```', content, re.DOTALL)

    # Add 'python' as the default language if none is specified
    new_code_blocks = [(lang.strip() or 'python', code.strip()) for lang, code in old_code_blocks]

    # Update the HTML with the new code blocks
    for (lang, code), (old_lang, old_code) in zip(new_code_blocks, old_code_blocks):
        old_block = f'```{old_lang}\n{old_code}\n```'
        new_block = f'```{lang}\n{code}\n```'
        content = content.replace(old_block, new_block)

    # write file with output path and filename
    if args.output:
        if args.path:
            args.output = os.path.join(args.path, args.output)            
        write_file(args.output, content)
    else:
        print(content)



if __name__ == '__main__':
    # task 1: convert simplied chinese to traditional chinese
    markdown_convert_chinese()

    #task 2: replace_http_img_with_local_link()
    # copy_local_img_to_media()

    #remove_http_link()
    #remove_figure_keys()
    #remove_figure_http_links()
    #replace_empty_code_block_with_python()
    #replace_empty_code_block_with_python_gpt()


