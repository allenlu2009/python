'''Parse a document and return a list of words'''
import sys
import re
import string

def parseDoc(doc):
    '''Parse a document and return a list of words'''
    # Remove all punctuation
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    # Split into words
    words = doc.split()
    # Make all words lowercase
    words = [word.lower() for word in words]
    # Remove all non-alphabetic words
    words = [word for word in words if word.isalpha()]
    return words

def main():
    '''Main function'''
    # Read the document
    doc = sys.stdin.read()
    # Parse the document
    words = parseDoc(doc)
    # Print the words
    for word in words:
        print(word)

if __name__ == '__main__':
    main()
