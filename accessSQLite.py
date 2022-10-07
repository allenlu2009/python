'''define the location of the sqlite database'''
from bs4 import BeautifulSoup
import sqlite3
import os

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
    #print_list(tab_str_lst)

    '''Get single-core device, description and score using method 1'''    
    singleCore = []
    tabpane = soup.find('div', id='single-core')
    name_score = tabpane.find_all('tr')
    for i in range(len(name_score)):
        namestr = list(name_score[i].stripped_strings)
        singleCore.append(namestr)
    #print_list(singleCore)

    '''Get Multi-core device, description and score using method 1'''    
    multiCore = []
    tabpane = soup.find('div', id='multi-core')
    name_score = tabpane.find_all('tr')
    for i in range(len(name_score)):
        namestr = list(name_score[i].stripped_strings)
        multiCore.append(namestr)
    #print_list(multiCore)

    '''Get opencl device, description and score using method 1'''    
    opencl = []
    tabpane = soup.find('div', id='opencl')
    name_score = tabpane.find_all('tr')
    for i in range(len(name_score)):
        namestr = list(name_score[i].stripped_strings)
        opencl.append(namestr)
    #print_list(opencl)

    '''Get Vulkan device, description and score using method 1'''    
    vulkan = []
    tabpane = soup.find('div', id='vulkan')
    name_score = tabpane.find_all('tr')
    for i in range(len(name_score)):
        namestr = list(name_score[i].stripped_strings)
        vulkan.append(namestr)
    #print_list(vulkan)

    return(singleCore, multiCore, opencl, vulkan)


def get_db_cursor():
    '''get the path to the sqlite database'''
    db_path = os.path.join(os.path.dirname(__file__), 'geekbench.db')
    '''get the database connection'''
    conn = sqlite3.connect(db_path)
    return conn, conn.cursor()


'''create the database'''
def create_db():
    conn, cur = get_db_cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS geekbench
                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Phone TEXT,
                    SoC TEXT,
                    SC INTEGER,
                    MC INTEGER,
                    OpenCL INTEGER,
                    Vulkan INTEGER)''')
    conn.commit()
    conn.close()


def insert_db(singleCore, multiCore, opencl, vulkan):
    conn, cur = get_db_cursor()
    '''for loop skip the first item in the list'''
    for i in range(1, len(singleCore)):
        cur.execute('''INSERT INTO geekbench (Phone, SoC, SC)
                       VALUES (?, ?, ?)''',(singleCore[i][0], singleCore[i][1], singleCore[i][2]),)
    for i in range(1, len(multiCore)):
        cur.execute('''INSERT INTO geekbench (Phone, SoC, MC)
                       VALUES (?, ?, ?)''',(multiCore[i][0], multiCore[i][1], multiCore[i][2]),)
    for i in range(1, len(opencl)):
        cur.execute('''INSERT INTO geekbench (Phone, SoC, OpenCL)
                       VALUES (?, ?, ?)''',(opencl[i][0], opencl[i][1], opencl[i][2]),)
    for i in range(1, len(vulkan)):
        cur.execute('''INSERT INTO geekbench (Phone, SoC, Vulkan)
                       VALUES (?, ?, ?)''',(vulkan[i][0], vulkan[i][1], vulkan[i][2]),)
    conn.commit()
    conn.close()


'''update the database'''
def update_db(singleCore, multiCore, opencl, vulkan):
    conn, cur = get_db_cursor()
    '''for loop skip the first item in the list'''
    for i in range(1, len(singleCore)):
        cur.execute('''UPDATE geekbench SET SC = ? WHERE Phone = ?''',(singleCore[i][2], singleCore[i][0]),)
    for i in range(1, len(multiCore)):
        cur.execute('''UPDATE geekbench SET MC = ? WHERE Phone = ?''',(multiCore[i][2], multiCore[i][0]),)
    for i in range(1, len(opencl)):
        cur.execute('''UPDATE geekbench SET OpenCL = ? WHERE Phone = ?''',(opencl[i][2], opencl[i][0]),)
    for i in range(1, len(vulkan)):
        cur.execute('''UPDATE geekbench SET Vulkan = ? WHERE Phone = ?''',(vulkan[i][2], vulkan[i][0]),)
    conn.commit()
    conn.close()


'''Find repeated phone in the sql database'''
def find_repeated_phone_in_db():
    conn, cur = get_db_cursor()
    cur.execute('''SELECT Phone FROM geekbenchmark''')
    tup = cur.fetchall()
    conn.close()
    (repeatTup, uniqueTup) = find_repeated_tuple(tup)
    '''access the first item, phone, in the tuple'''
    repeatLst = [item[0] for item in repeatTup]
    uniqueLst = [item[0] for item in uniqueTup]
    return (repeatLst, uniqueLst)


'''Find repeated phone and soc in the sql database'''
def find_repeated_phone_soc_in_db():
    conn, cur = get_db_cursor()
    cur.execute('''SELECT Phone, SoC FROM geekbenchmark''')
    tup = cur.fetchall()
    conn.close()
    (repeatTup, uniqueTup) = find_repeated_tuple(tup)
    '''access the first item, phone, in the tuple'''
    repeatLst = [item[0] for item in repeatTup]
    uniqueLst = [item[0] for item in uniqueTup]
    return (repeatLst, uniqueLst)


'''Find repeated items in a list'''
def find_repeated_item(lst):
    '''Find repeated items in a list'''
    uniqueLst = []
    repeatLst = []
    for i in range(len(lst)):
        if lst[i] in uniqueLst:
            print(lst[i])
            repeatLst.append(lst[i])
        else:
            uniqueLst.append(lst[i])    
    return (repeatLst, uniqueLst)


'''Find repeated items in a tuple'''
def find_repeated_tuple(tup):
    uniqueTup = []
    repeatTup = []
    for i in range(len(tup)):
        if tup[i] in uniqueTup:
            print(tup[i])
            repeatTup.append(tup[i])
        else:
            uniqueTup.append(tup[i])    
    return (repeatTup, uniqueTup)


'''query the database by Phone name'''
def query_db_by_phone(phone):
    conn, cur = get_db_cursor()
    cur.execute('''SELECT * FROM geekbench WHERE Phone = ?''',(phone,))
    lst = cur.fetchall()
    conn.close()
    return lst


'''query the database by SoC name'''
def query_db_by_soc(soc):
    conn, cur = get_db_cursor()
    cur.execute('''SELECT * FROM geekbench WHERE SoC = ?''',(soc,))
    lst = cur.fetchall()
    conn.close()
    return lst




def main():

    with open(r'./geekbench.html','r') as f:
        geekbench_html = f.read()
    f.close()

    (conn, loc) = get_db_cursor()

    (SC, MC, OpenCL, Vulkan) = parse_geekbench_html(geekbench_html)
    
    #create_db()
    #insert_db(SC, MC, OpenCL, Vulkan)
    #update_db(SC, MC, OpenCL, Vulkan)

    repeatLst, uniqueLst = find_repeated_phone_in_db()  # find repeated phone in the sql database
    #repeatLst, uniqueLst = find_repeated_phone_soc_in_db()  # both phone and soc are the same
    '''query the database by Phone name'''
    for i in range(len(repeatLst)):
        print_list(query_db_by_phone(repeatLst[i]))
    

    for i in range(1,len(SC)):
        #print(SC[i][0], SC[i][1], SC[i][2])
        '''query if the device is in the database'''
        #loc.execute("UPDATE geekbenchmark SET SC = 0 WHERE Phone = 'Lenovo Legion Y90'")
        #loc.execute("UPDATE geekbenchmark SET SC = 0 WHERE Phone = ?", (str(SC[i][0]),))
        loc.execute("UPDATE geekbenchmark SET SC = ? WHERE Phone = ?", (SC[i][2], SC[i][0],))
        #t = ('LG Phoenix 2',)
        #loc.execute("SELECT * FROM geekbenchmark WHERE Phone = ?", (str(SC[i][0]),))
        #loc.execute("SELECT * FROM geekbenchmark WHERE Phone = ?", t)
        #sql = "SELECT * FROM geekbenchmark WHERE (Phone) = (?)", (SC[i][0])
        #loc.execute(sql)
        #result = loc.fetchall()
        #print(result)
        #conn.commit()
        #loc.execute("INSERT INTO geekbenchmark (Phone, SoC, SC) VALUES (?,?,?)", (SC[i][0], SC[i][1], SC[i][2]))
    for i in range(1,len(MC)):
        #print(MC[i][0], MC[i][1], MC[i][2])
        loc.execute("UPDATE geekbenchmark SET MC = ? WHERE Phone = ?", (MC[i][2], MC[i][0],))
        #conn.commit()
    print(OpenCL)
    for i in range(1,len(OpenCL)):
        #print(OpenCL[i][0], OpenCL[i][1], OpenCL[i][2])
        loc.execute("UPDATE geekbenchmark SET OpenCL = ? WHERE Phone = ?", (OpenCL[i][2], OpenCL[i][0],))
        #conn.commit()
    print(Vulkan)
    for i in range(1,len(Vulkan)):
        #print(Vulkan[i][0], Vulkan[i][1], Vulkan[i][2])
        loc.execute("UPDATE geekbenchmark SET Vulkan = ? WHERE Phone = ?", (Vulkan[i][2], Vulkan[i][0],))
        #conn.commit()
    conn.commit()

if __name__ == '__main__':
    main()
