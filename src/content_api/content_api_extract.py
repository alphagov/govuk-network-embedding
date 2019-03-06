# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import logging.config
import os
import re
import urllib.request

import pandas as pd

from src.content_api.details_utils import extract_from_details


def save_all_to_file(json_dict, page_links, related_page_links, collection_links, destination_dir, pre_fix):
    print("Number of pages for links:", len(page_links))
    print("Number of pages for json:", len(json_dict))

    rows_json = [value for key, value in json_dict.items()]
    json_df = pd.DataFrame(rows_json)
    json_df.drop(['analytics_identifier', 'phase', 'public_updated_at',
                  'publishing_request_id', 'publishing_scheduled_at',
                  'scheduled_publishing_delay_seconds', 'schema_name',
                  'updated_at', 'withdrawn_notice'], axis=1, inplace=True)

    json_df.to_csv(os.path.join(destination_dir, pre_fix + "_content_json.csv.gz"), compression="gzip", index=False)

    rows_links = [{"url": key,
                   "embedded_links": value,
                   "related_links": related_page_links[key],
                   "collection_links": collection_links[key]} for key, value in page_links.items()]
    df_rel = pd.DataFrame(rows_links)
    df_rel = df_rel[['url', 'embedded_links', 'related_links', 'collection_links']]

    df_rel['num_rel'] = df_rel['related_links'].map(len)
    df_rel['num_emb'] = df_rel['embedded_links'].map(len)
    df_rel['num_coll'] = df_rel['collection_links'].map(len)

    df_rel.to_csv(os.path.join(destination_dir, pre_fix + "_content_api_links.csv.gz"), index=False, compression="gzip")
    logging.info("Finished writing...")


def chunked_extract(nodes_srs, chunks_list, destination_dir):
    not_found = []

    for j, (start, end) in enumerate(chunks_list):
        page_links = {}
        related_page_links = {}
        collection_links = {}

        json_dict = {}
        logger.info("Working on {}:{}".format(start, end))
        for i, node in enumerate(nodes_srs[start:end].values):
            content_item = None
            try:
                url = "https://www.gov.uk/api/content" + node
                content_item = json.loads(urllib.request.urlopen(url).read())
                content_item['url'] = node
                json_dict[node] = content_item

            except Exception:
                # logger.debug("Url \'{}\' not found".format(url))
                not_found.append(url)

            extract_link_types(collection_links, content_item, related_page_links, page_links)

            if i % 500 == 0:
                print(datetime.datetime.now().strftime("%H:%M:%S"), "run:", j, "row", i)

        save_all_to_file(json_dict, page_links, related_page_links, collection_links, destination_dir,
                         "pt{:02d}of{}_".format(j + 1, len(chunks_list)))
    return not_found


def extract_link_types(collection_links, content_item, related_page_links, page_links):
    if content_item is not None:
        links = extract_from_details(content_item['details'])
        related_links = []
        coll_links = []
        if 'ordered_related_items' in content_item['links'].keys():
            related_links = [related_item['base_path'] for related_item in
                             content_item['links']['ordered_related_items'] if
                             'base_path' in related_item.keys()]

        if 'documents' in content_item['links'].keys():
            coll_links = [document['base_path'] for document in content_item['links']['documents'] if
                          'base_path' in document.keys()]

        related_page_links[content_item['base_path']] = related_links
        collection_links[content_item['base_path']] = coll_links
        page_links[content_item['base_path']] = links


def compute_chunks(start, end, chunk_size):
    return [(i, i + chunk_size) if i + chunk_size < end else (i, end - 1) for i in
            range(start, end, chunk_size)]


def merge_delete(directory, prefix):
    logging.info("Merging and deleting created subfiles...")
    files_to_del = merge_dataframe(directory, prefix)
    for file in files_to_del:
        logging.debug("Deleting {}...".format(file))
        os.remove(file)


def merge_dataframe(directory, prefix):
    all_files = []
    for keyword in ["json", "api_links"]:
        filelist = sorted(
            [os.path.join(directory, file) for file in os.listdir(directory) if keyword in file and "pt" in file])
        print(filelist)
        filename = re.sub("pt\d+of\d+_", prefix, filelist[0])
        logging.info("Saving {} file at {}.".format(keyword, filename))
        pd.concat([pd.read_csv(file, compression="gzip")
                   for file in filelist], ignore_index=True).to_csv(filename,
                                                                    compression="gzip",
                                                                    index=False)
        all_files.extend(filelist)
    return all_files


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Module to extract content item schema (page text and metadata'
                                                 'using the Content API.')
    parser.add_argument('filename', help='Input node dataframe filename. Will look in processed_network dir.')
    parser.add_argument('-start', default=0, type=int, help='Start index.')
    parser.add_argument('-end', default=-1, type=int, help='End index.')
    parser.add_argument('-step', default=10000, type=int, help='Step')

    args = parser.parse_args()
    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('content_api_extract')

    DATA_DIR = os.getenv("DATA_DIR")
    processed_network = os.path.join(DATA_DIR, "processed_network")
    nodefile = os.path.join(processed_network, args.filename + ".csv.gz" if ".csv.gz" else args.filename)

    content_api = os.path.join(DATA_DIR, "content_api")
    dest_dir = os.path.join(content_api, datetime.datetime.now().strftime("%d-%m-%y"))

    nodes = pd.read_csv(nodefile, compression="gzip", sep="\t")
    print(nodes.shape)
    shape = nodes.shape[0]
    if args.end != -1:
        shape = args.end
    chunks = compute_chunks(args.start, shape, args.step)
    print(chunks)

    if not os.path.isdir(dest_dir):
        logging.info("Specified destination directory \"{}\" does not exist, creating...".format(dest_dir))
        os.mkdir(dest_dir)
    else:
        logging.info("Specified destination directory \"{}\" does exist, creating v2...".format(dest_dir))
        dest_dir = dest_dir + "_v2"
        os.mkdir(dest_dir)

    not_found = chunked_extract(nodes.Node, chunks, dest_dir)
    print(len(not_found))

    with open(os.path.join(dest_dir, "not_found_urls.csv"), "w") as writer:
        for f in not_found:
            writer.write("{}\n".format(f))

    merge_delete(dest_dir, args.filename)
