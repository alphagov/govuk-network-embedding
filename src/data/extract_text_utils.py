import json
from collections import OrderedDict

from bs4 import BeautifulSoup
from lxml import etree
from lxml import html
from pandas.io.json import json_normalize

look = ['title', 'body']
child_keys = ['title', 'description']
filtered = ['body', 'brand', 'documents', 'final_outcome_detail', 'final_outcome_documents',
            'government', 'headers', 'introduction', 'introductory_paragraph',
            'licence_overview', 'licence_short_description', 'logo', 'metadata', 'more_information', 'need_to_know',
            'other_ways_to_apply', 'summary', 'ways_to_respond', 'what_you_need_to_know', 'will_continue_on', 'parts',
            'collection_groups']


def extract_text(body):
    """
    Extract text from html body
    :param body: <str> containing html.
    """
    # TODO: Tidy this up!
    r = None
    # body != "\n" and
    if body and body != "\n" and not body.isspace():
        try:
            # print("this is", body)
            tree = etree.HTML(body)
            r = tree.xpath('//text()')
            r = ' '.join(r)
            r = r.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            r = r.replace('\n', ' ').replace(',', ' ')
            # r = r.lower()
            r = ' '.join(r.split())
        except ValueError:
            print("exception @ extract:", type(body), body)
    if not r:
        r = ' '
    return r


def extract_html_links(url):
    links = []
    try:
        soup = BeautifulSoup(url, "html5lib")
        links = [link.get('href') for link in soup.findAll('a', href=True)]
    except Exception:
        print("error")
    return [l for l in links if l.startswith("/")]



def iterate_json_keys(for_text, x, aggregator):
    for key, raw_text in sorted(x.items()):
        if key in filtered:
            if isinstance(raw_text, str) and len(raw_text) > 1:
                if for_text:
                    aggregator = flat_extract(0, raw_text)
                else:
                    aggregator.extend(flat_extract(1, raw_text))
            elif isinstance(raw_text, list) and len(raw_text) > 0:
                for sub_text in raw_text:
                    if for_text:
                        aggregator = nested_extract(for_text, sub_text)
                    else:
                        aggregator.extend(nested_extract(1, sub_text))
    return aggregator


# def get_links(x):
#     links = []
#     string_json = json.dumps(OrderedDict(x))
#     order_json = json.loads(string_json, object_pairs_hook=OrderedDict)
#     for key, raw_text in sorted(order_json.items()):
#         if key in filtered:
#             if isinstance(raw_text, str) and len(raw_text) > 1:
#                 links.extend(flat_extract(1, raw_text))
#             elif isinstance(raw_text, list) and len(raw_text) > 0:
#                 for sub_text in raw_text:
#                     if is_json(sub_text):
#                         links.extend(nested_extract_links(sub_text))
#                     elif is_html(sub_text):
#                         links.extend(extract_html_links(sub_text))
#     return list(set(links))


def nested_extract(function_type, sub_text="", total_text=""):
    if function_type == 0:
        if is_json(sub_text):
            total_text += nested_iteration(sub_text, "")
        elif is_html(sub_text):
            str_from_html = extract_text(sub_text)
            total_text += " " + str_from_html
        return total_text
    else:
        if is_json(sub_text):
            return nested_iteration(sub_text, [])
        elif is_html(sub_text):
            return extract_html_links(sub_text)


def flat_extract(function_type, raw_text="", total_text=""):
    if function_type == 0:
        raw_text = raw_text.replace("-", " ")
        raw_token = raw_text.split(" ")
        if len(raw_token) > 0:
            raw_string = extract_text(raw_text)
            total_text += " " + raw_string
        return total_text
    else:
        return extract_html_links(raw_text)


def nested_iteration(x, aggregator):
    """
Iterate over nested json (avoiding recursion), flattening loops.
    :param x: nested `details` cell contents
    :return: plaintext
    """

    string_json2 = json.dumps(OrderedDict(x))
    order_json2 = json.loads(string_json2, object_pairs_hook=OrderedDict)
    if ('body' or 'title') in order_json2.keys():
        for item in look:
            if isinstance(aggregator, str):
                raw_string2 = extract_text(order_json2[item])
                if len(raw_string2.split()) > 1:
                    aggregator += " " + raw_string2
            else:
                aggregator.extend(extract_html_links(order_json2[item]))
    elif 'child_sections' in order_json2.keys():
        for child in order_json2['child_sections']:
            for key in child_keys:
                if isinstance(aggregator, str):
                    aggregator += " " + child[key]
                else:
                    aggregator.extend(extract_html_links(child))
    return aggregator


def get_text_json_file(x, aggregator):
    """
From dict to json and back (to OrderedDict), iterate over json from details column (based on list filtered, should
reconsider) and extract plaintext from included html.
    :param x: details cell from dataset
    :return: plaintext
    """
    total_text = ""
    if x != "":
        total_text = iterate_json_keys(1, x)
    else:
        links = []
        string_json = json.dumps(OrderedDict(x))
        order_json = json.loads(string_json, object_pairs_hook=OrderedDict)
        links = iterate_json_keys(0, x)
    return total_text.strip()

def is_json(raw_text):
    try:
        json_normalize(raw_text).columns.tolist()
    except AttributeError:
        return False
    return True


def is_html(raw_text):
    return html.fromstring(str(raw_text)).find('.//*') is not None


# def nested_extract_text(x):
#     """
# Iterate over nested json (avoiding recursion), flattening loops.
#     :param x: nested `details` cell contents
#     :return: plaintext
#     """
#     ttext = ""
#     string_json2 = json.dumps(OrderedDict(x))
#     order_json2 = json.loads(string_json2, object_pairs_hook=OrderedDict)
#     if ('body' or 'title') in order_json2.keys():
#         for item in look:
#             raw_string2 = extract_text(order_json2[item])
#             if len(raw_string2.split()) > 1:
#                 ttext += " " + raw_string2
#     elif 'child_sections' in order_json2.keys():
#         for child in order_json2['child_sections']:
#             for key in child_keys:
#                 ttext += " " + child[key]
#     return ttext
#
#
# def nested_extract_links(x):
#     links = []
#     string_json2 = json.dumps(OrderedDict(x))
#     order_json2 = json.loads(string_json2, object_pairs_hook=OrderedDict)
#     if ('body' or 'title') in order_json2.keys():
#         for item in look:
#             links.extend(extract_html_links(order_json2[item]))
#     elif 'child_sections' in order_json2.keys():
#         for child in order_json2['child_sections']:
#             if child in child_keys:
#                 links.extend(extract_html_links(child))
#     return links
