# -*- coding: utf-8 -*-
import json
from collections import OrderedDict

from bs4 import BeautifulSoup
from lxml import etree
from lxml import html
from pandas.io.json import json_normalize

LOOK = ['title', 'body']
CHILD_KEYS = ['title', 'description']
DETAILS_SECTIONS = ['body', 'brand', 'documents', 'final_outcome_detail', 'final_outcome_documents',
                    'government', 'headers', 'introduction', 'introductory_paragraph',
                    'licence_overview', 'licence_short_description', 'logo', 'metadata', 'more_information',
                    'need_to_know',
                    'other_ways_to_apply', 'summary', 'ways_to_respond', 'what_you_need_to_know', 'will_continue_on',
                    'parts',
                    'collection_groups']


def is_json(raw_text):
    try:
        json_normalize(raw_text).columns.tolist()
    except AttributeError:
        return False
    return True


def is_html(raw_text):
    return html.fromstring(str(raw_text)).find('.//*') is not None


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


def extract_html_links(text):
    """
    Grab any GOV.UK domain-specific links from page text.
    :param text: Text within a details sub-section, refer to filtered for keys.
    :return: list of links
    """
    links = []
    try:
        soup = BeautifulSoup(text, "html5lib")
        links = [link.get('href') for link in soup.findAll('a', href=True)]
    except Exception:
        print("error")
    return [l.replace("https://www.gov.uk/", "/") for l in links
            if l.startswith("/") or l.startswith("https://www.gov.uk/")]


def nested_json_extract(aggregator, nested_json):
    """
    Iterate over nested json (avoiding recursion), flattening loops.
    :param aggregator: text or links aggregator
    :param nested_json: nested `details` schema contents
    :return: text or links
    """

    sub_json_str = json.dumps(OrderedDict(nested_json))
    order_sub_json = json.loads(sub_json_str, object_pairs_hook=OrderedDict)
    if ('body' or 'title') in order_sub_json.keys():
        for item in LOOK:
            if isinstance(aggregator, str):
                sub_raw_str = extract_text(order_sub_json[item])
                if len(sub_raw_str.split()) > 1:
                    aggregator += " " + sub_raw_str
            else:
                aggregator += extract_html_links(order_sub_json[item])
    elif 'child_sections' in order_sub_json.keys():
        for child in order_sub_json['child_sections']:
            for key in CHILD_KEYS:
                if isinstance(aggregator, str):
                    aggregator += " " + child[key]
                else:
                    aggregator.extend(extract_html_links(child))
    return aggregator


def flat_extract(aggregator, raw_text=""):
    """
    Aggregate text from a details section, then strip it from html or extract links depending on aggregator functionality.
    :param aggregator:
    :param raw_text:
    :return:
    """
    if isinstance(aggregator, str):
        raw_text = raw_text.replace("-", " ")
        raw_token = raw_text.split(" ")
        if len(raw_token) > 0:
            raw_string = extract_text(raw_text)
            aggregator += " " + raw_string
    else:
        aggregator += extract_html_links(raw_text)
    return aggregator


def nested_extract(aggregator, sub_text=""):
    """
    Section of details may be a further nested json dict or html, aggregate text or links from that.
    :param aggregator:
    :param sub_text:
    :return:
    """
    if is_json(sub_text):
        aggregator += nested_json_extract(aggregator, sub_text)
    elif is_html(sub_text):
        if isinstance(aggregator, str):
            str_from_html = extract_text(sub_text)
            aggregator += " " + str_from_html
        else:
            aggregator += extract_html_links(sub_text)
    return aggregator


def extract_from_details(details, function_type):
    """
    Generic implementation to extract text or links from the details entry in a content store item. Details contains
    what is shown in the main body of a page. This is catered to content pages.
    :param details: A nested json dictionary-like structure
    :param function_type: extract texts or links
    :return: the aggregated text or links
    """
    if function_type == "text":
        aggregator = ""
    elif function_type == "links":
        aggregator = []

    for key, raw_text in sorted(details.items()):
        if key in DETAILS_SECTIONS:
            if isinstance(raw_text, str) and len(raw_text) > 1:
                aggregator = flat_extract(aggregator, raw_text)
            elif isinstance(raw_text, list) and len(raw_text) > 0:
                for sub_text in raw_text:
                    aggregator = nested_extract(aggregator, sub_text)

    if isinstance(aggregator, list) and "transaction_start_link" in details.keys():
        aggregator.append(details["transaction_start_link"])
    if isinstance(aggregator, str):
        aggregator = aggregator.strip()

    return aggregator

def cs_extract_text(details):
    """
    Generic implementation to extract text or links from the details entry in a content store item. Details contains
    what is shown in the main body of a page. This is catered to content pages.
    :param details: A nested json dictionary-like structure
    :param function_type: extract texts or links
    :return: the aggregated text or links
    """

    text = ""
    for key, dict_list in sorted(details.items()):
        if key in DETAILS_SECTIONS:
            # print("thru if", key)
            if isinstance(dict_list, list):
                for dict_i in dict_list:
                    # print("dict_i", dict_i)
                    if 'content_type' in dict_i.keys() and dict_i['content_type'] == "text/html":
                        # print("get content")
                        text += " " + extract_text(dict_i['content'])
                    elif any(l in dict_i.keys() for l in LOOK):
                        for l in LOOK:
                            if l in dict_i.keys():
                                # if isinstance(dict_i[l], dict):
                                #     print("this is a dict")
                                #     for dict_j in dict_i[l]:
                                #         print(dict_j)
                                #         if 'content_type' in dict_j.keys() and dict_j['content_type'] == "text/html":
                                #             text += " " + extract_text(dict_j['content'])
                                if isinstance(dict_i[l], list):
                                    for item in dict_i[l]:
                                        text += cs_extract_text(item)
                                elif isinstance(dict_i[l], str):
                                    text += " " + extract_text(dict_i[l])
                                else:
                                    print("not a list, not a str")
                                    print(l, dict_i)
            elif isinstance(dict_list, dict):
                for l in LOOK:
                    if l in dict_list.keys():
                        text += " " + extract_text(dict_list[l])
            else:
                # print("not a dict, not a list")
                # print(key, dict_list)
                text += " " + extract_text(dict_list)

    return text.strip()


def cs_extract_links(details):
    """
    Generic implementation to extract text or links from the details entry in a content store item. Details contains
    what is shown in the main body of a page. This is catered to content pages.
    :param details: A nested json dictionary-like structure
    :param function_type: extract texts or links
    :return: the aggregated text or links
    """

    links = []

    for key, dict_list in sorted(details.items()):
        if key in DETAILS_SECTIONS:
            # print("thru if", key)
            if isinstance(dict_list, list):
                for dict_i in dict_list:
                    # print("dict_i", dict_i)
                    if 'content_type' in dict_i.keys() and dict_i['content_type'] == "text/html":
                        # print("get content")
                        links.extend(extract_html_links(dict_i['content']))
                    elif any(l in dict_i.keys() for l in LOOK):
                        for l in LOOK:
                            print(l)
                            if l in dict_i.keys():
                                for dict_j in dict_i[l]:
                                    print(dict_j)
                                    if 'content_type' in dict_j.keys() and dict_j['content_type'] == "text/html":
                                        links.extend(extract_html_links(dict_j['content']))
    if "transaction_start_link" in details.keys():
        links.append(details["transaction_start_link"])

    return links
