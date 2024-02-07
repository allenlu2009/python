# Parse html use BeautifulSoup

from bs4 import BeautifulSoup
import re

doc_html = '''
    <html>
        <head>
            <title>BeautifulSoup</title>
        </head>
        <body>
            <p class="title"><b>The Dormouse's story</b></p>
            <p class="story">Once upon a time there were three little sisters; and their names were
            <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
            <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
            <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
            and they lived at the bottom of a well.</p>
            <p class="story">...</p>
        </body>
    </html>
'''

geekbench_simple_html = '''
<html>
    <head>
        <title>Geekbench</title>
    </head>
    <body>
        <p class="title"><b>Geekbench</b></p>
        <p class="score">
            <span class="single">Single-Core Score:  1,000</span>
            <span class="multi">Multi-Core Score:  2,000</span>
        </p>
        <p class="story">...</p>
    </body>
</html>
'''

geekbench_html = '''
<html>
    <head>
        <title>Android Benchmarks ... </title>
    </head>
    <body class='corktown'>
        <nav class="navbar ..."> ... </nav>
        <div class='container'> 
            <div class='row'> 
                <div class='primary ...'>   
        	        <div class='page-header'>
                        <h1>Android Benchmarks</h1>
        	        </div>
        	        <div class='tabbable'>
                        <ul class='nav-pills'>
                            <li class='nav-item'>
                            <a class='nav-link active' ... href='#single-core'>Single-Core</a>
                            </li>
                            <li class='nav-item'>
                   	        <a class='nav-link' ... href='#multi-core'>Multi-Core</a>
                            </li>
                            <li class='nav-item'>
                    	        <a class='nav-link' ... href='#opencl'>OpenCL</a>
                            </li>
                            <li class='nav-item'>
                    	        <a class='nav-link' ... href='#vulkan'>Vulkan</a>
                            </li>
                        </ul>
                        <div class='tab-content'>
                            <div class='tab-pane fade show active' id='single-core'>
                                <tbody> .. </tbody>
                            </div>
                            <div class='tab-pane fade' id='multi-core'>
                                <tbody> .. </tbody>
                            </div>
                            <div class='tab-pane fade' id='opencl'>
                                <tbody> .. </tbody>
                            </div>
                            <div class='tab-pane fade' id='vulkan'>
                                <tbody> .. </tbody>
                            </div>
                        </div>   
                    </div>
                </div>
            </div>
        </div>
    </body>
</html>
'''

'''
Use BeautifulSoup to find <head> tag and exatract the title
Use BeautifulSoup to find <tbody> tag and exatract the Single-Core device and score
Use BeautifulSoup to find <tbody> tag and exatract the Multi-Core device and score
Use BeautifulSoup to find <tbody> tag and exatract the OpenCL device and score
Use BeautifulSoup to find <tbody> tag and exatract the Vulkan device and score
'''



def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    print(soup.prettify())
    print(soup.title)
    print(soup.title.name)
    print(soup.title.string)
    print(soup.title.parent.name)
    print(soup.p)
    print(soup.p['class'])
    print(soup.a)
    print(soup.find_all('a'))
    print(soup.find(id="link3"))
    print(soup.get_text())

def parse_geekbench_simple_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    print(soup.prettify())
    print(soup.title)
    print(soup.title.name)
    print(soup.title.string)
    print(soup.title.parent.name)
    print(soup.p)
    print(soup.p['class'])
    print(soup.find_all('span'))
    print(soup.find_all('span')[0].string)
    print(soup.find_all('span')[1].string)

def parse_geekbench_simple2_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    # score
    # device
    '''Find all span'''
    span_list = soup.find_all('span')
    #print(span_list)

    '''Get score'''
    for span in span_list:
        if span['class'][0] == 'single':
            print("single-core score: ", span.string)
        if span['class'][0] == 'multi':
            print("multi-core score: ", span.string)

    '''Get score by find_all'''
    span_list = soup.find_all('span', class_='single')
    print("single-core score: ", span_list[0].string)

    span_list = soup.find_all('span', class_='multi')
    print("multi-core score: ", span_list[0].string)

    '''Get score by find'''
    span = soup.find('span', class_='single')
    print("single-core score: ", span.string)

    span = soup.find('span', class_='multi')
    print("multi-core score: ", span.string)

    '''Get score by select'''
    span = soup.select('span.single')
    print("single-core score: ", span[0].string)

    span = soup.select('span.multi')
    print("multi-core score: ", span[0].string)

    '''Get score by select_one'''
    span = soup.select_one('span.single')
    print("single-core score: ", span.string)

    span = soup.select_one('span.multi')
    print("multi-core score: ", span.string)

    '''Get score by find_all'''
    span_list = soup.find_all('span', class_='single')
    print("single-core score: ", span_list[0].string)

    span_list = soup.find_all('span', class_='multi')
    print("multi-core score: ", span_list[0].string)

    '''Get score by find'''
    span = soup.find('span', class_='single')
    print("single-core score: ", span.string)

    span = soup.find('span', class_='multi')
    print("multi-core score: ", span.string)

    '''Get score by select'''
    span = soup.select('span.single')
    print("single-core score: ", span[0].string)

    span = soup.select('span.multi')
    print("multi-core score: ", span[0].string)

    '''Get score by select_one'''
    span = soup.select_one('span.single')
    print("single-core score: ", span.string)

    span = soup.select_one('span.multi')
    print("multi-core score: ", span.string)


'''extract the device name using regex, <div class="device"></div>..<a href='/android_devices/device'>...</a>'''
def extract_device_name(text):
    text = text.replace('\n', '')
    pattern = re.compile(r'<a href.+>(.+)<\/a>')
    match = pattern.search(text)
    if match:
        return match.group(1)
    else:
        return None


'''remove the leading and trailing \n'''
def remove_newline(text):
    return text.strip('\n')

'''print a list of string'''
def print_list(list):
    for item in list:
        print(item)


def parse_geekbench_html(html):
    soup = BeautifulSoup(html, 'html.parser')

    '''Get title'''
    #print(soup.title)
    title = soup.title.string   # return string
    title2 = list(soup.title.strings)  # return generator
    title3 = list(soup.title.stripped_strings) # return generator
    print(title3)

    '''Get 4 id's in the tab: single-core, multi-core, opencl, vulkan'''
    tab_lst = []
    lst = soup.find('ul', class_='nav nav-pills').find_all('a')
    for item in lst:
        linkname = item['href']
        linkname = linkname[1:]  # remove the leading #
        tab_lst.append(linkname)
    print_list(tab_lst)
    '''Get 4 bullets in the tab: Single core, Multi-Core, OpenCL, Vulkan'''
    tab_str_lst = list(soup.find('ul', class_='nav nav-pills').stripped_strings)
    print_list(tab_str_lst)

    '''Get single-core device, description and score using method 1'''    
    singleCore = []
    tabpane = soup.find('div', id='single-core')
    name_score = tabpane.find_all('tr')
    for i in range(len(name_score)):
        namestr = list(name_score[i].stripped_strings)
        singleCore.append(namestr)
    print_list(singleCore)

    '''Get Multi-core device, description and score using method 1'''    
    multiCore = []
    tabpane = soup.find('div', id='multi-core')
    name_score = tabpane.find_all('tr')
    for i in range(len(name_score)):
        namestr = list(name_score[i].stripped_strings)
        multiCore.append(namestr)
    print_list(multiCore)

    '''Get opencl device, description and score using method 1'''    
    opencl = []
    tabpane = soup.find('div', id='opencl')
    name_score = tabpane.find_all('tr')
    for i in range(len(name_score)):
        namestr = list(name_score[i].stripped_strings)
        opencl.append(namestr)
    print_list(opencl)

    '''Get Vulkan device, description and score using method 1'''    
    vulkan = []
    tabpane = soup.find('div', id='vulkan')
    name_score = tabpane.find_all('tr')
    for i in range(len(name_score)):
        namestr = list(name_score[i].stripped_strings)
        vulkan.append(namestr)
    print_list(vulkan)


    '''
    lst = soup.find('ul', class_='nav nav-pills')
    link = lst.find_all('a', class_='nav-link')
    '''
    '''create a list to store the linkref and linktext'''
    '''linkref: href='#single-core' '''
    '''
    link_list = []
    for eachlink in link:
        linkref = eachlink['href']
        # linkref: href='#single-core' 
        # remove the hashtag from the link
        linkref = linkref[1:]
        linktext = eachlink.string
        link_list.append((linkref, linktext))
    print(link_list)'''

    '''Get single-core device, description and score using method 2'''    
    '''
    singleCore = []
    tabpane = soup.find('div', id='single-core')
    name = tabpane.find_all('td', class_='name')
    dscrp = tabpane.find_all('div', class_='description')
    score = tabpane.find_all('td', class_='score')
    for i in range(len(name)):
        namestr = extract_device_name(str(name[i]))
        namestr1 = extract_device_name(str(name[i]))
        namestr2 = list(name[i].stripped_strings)   # good one
        dscrpstr = remove_newline(dscrp[i].string)
        dscrpstr1 = dscrp[i].string.strip('\n')
        dscrpstr2 = list(dscrp[i].stripped_strings)  
        scorestr = remove_newline(score[i].string)
        scorestr1 = score[i].string.strip('\n')
        scorestr2 = list(score[i].stripped_strings) # good one
        #singleCore.append((namestr, dscrpstr, scorestr))
        singleCore.append((namestr2, scorestr2))
    for i in range(len(singleCore)):
        print(singleCore[i])
    '''
        
    '''Get multi-core device, description and score'''    
    '''multiCore = []
    tabpane = soup.find('div', id='multi-core')
    name = tabpane.find_all('td', class_='name')
    dscrp = tabpane.find_all('div', class_='description')
    score = tabpane.find_all('td', class_='score')
    for i in range(len(name)):
        namestr = extract_device_name(str(name[i]))
        dscrpstr = remove_newline(dscrp[i].string)
        scorestr = remove_newline(score[i].string)
        multiCore.append((namestr, dscrpstr, scorestr))
    for i in range(len(multiCore)):
        print(multiCore[i])
    '''

    '''Get opencl device, description and score'''    
    '''opencl = []
    tabpane = soup.find('div', id='opencl')
    name = tabpane.find_all('td', class_='name')
    dscrp = tabpane.find_all('div', class_='description')
    score = tabpane.find_all('td', class_='score')
    for i in range(len(name)):
        namestr = extract_device_name(str(name[i]))
        dscrpstr = remove_newline(dscrp[i].string)
        scorestr = remove_newline(score[i].string)
        opencl.append((namestr, dscrpstr, scorestr))
    for i in range(len(opencl)):
        print(opencl[i])
    '''

    '''Get vulkan device, description and score'''    
    '''vulkan = []
    tabpane = soup.find('div', id='vulkan')
    name = tabpane.find_all('td', class_='name')
    dscrp = tabpane.find_all('div', class_='description')
    score = tabpane.find_all('td', class_='score')
    for i in range(len(name)):
        namestr = extract_device_name(str(name[i]))
        dscrpstr = remove_newline(dscrp[i].string)
        scorestr = remove_newline(score[i].string)
        vulkan.append((namestr, dscrpstr, scorestr))
    for i in range(len(vulkan)):
        print(vulkan[i])
    '''

    '''for i in range(0, len(link)):
        #print(tbody[i].string)
        name = tbody[i].find_all('td', class_='name')
        dscrp = tbody[i].find_all('div', class_='description')
        score = tbody[i].find_all('td', class_='score')
        print(score)'''



def main():

    with open(r'./geekbench.html','r') as f:
        geekbench_html = f.read()
    f.close()

    parse_html(doc_html)
    parse_geekbench_simple_html(geekbench_simple_html)
    parse_geekbench_html(geekbench_html)

if __name__ == '__main__':
    main()









