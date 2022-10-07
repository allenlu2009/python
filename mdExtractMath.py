'''Extract the math from a markdown file and save it to a separate file.'''
import argparse
import re
import sys
import os
from functools import partial

try:
    from html import escape
    html_escape = partial(escape, quote=False)
except ImportError:
    # Python 2
    from cgi import escape as html_escape

import mistune
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from pygments.util import ClassNotFound

from nbconvert.filters.strings import add_anchor

math_block = []
