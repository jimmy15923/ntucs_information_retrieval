import os
import time
import json
import math
import jieba
import argparse
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def text_strip(x):
    return x.strip("\n").replace(" ", '').replace("。", '').replace("「", '').replace("」", '').replace("－", '').replace("，",'')

def okapi(tf, doc_len, k=1.4, b=0.75):
    new_tf = (tf * (k+1)) / (tf + k*(1-b + b*(doc_len/1378.85)))
    return new_tf    

def rocchio(query, rankings, alpha=1, beta=0.5, gamma=0.15, rel_count=5, nrel_count=1, iters=5):
    for _ in range(iters):
        # Update query vectors with Rocchio algorithm
        rel_vecs = doc_tfidf[rankings[:, :rel_count]].mean(axis=1)
        nrel_vecs = doc_tfidf[rankings[:, -nrel_count:]].mean(axis=1)
        query = alpha * query + beta * rel_vecs - gamma * nrel_vecs

    return query    

def text_to_gram(texts, n_gram='bi'):
    """
    Split the text into Ngram and return all the unique token in the texts
    """
    token_frequency = Counter()
    for term in texts:
        if n_gram == 'u':
            for voc in term:
                token_frequency[str(voc_to_token[voc])] += 1
        else:
            if (len(term) % 2 == 0):
                n_token = int(len(term) / 2)
                for i in range(n_token):
                    t = term[i*2: (i+1)*2]
                    token = f"{voc_to_token[t[0]]}_{voc_to_token[t[1]]}"
                    token_frequency[token] += 1
            else:
                n_token = int(len(term) / 2) + 1
                if n_token == 1:
                    token_frequency[str(voc_to_token[term])] += 1
                else:
                    for i in range(n_token-1):
                        t = term[i*2: (i+1)*2]
                        token = f"{voc_to_token[t[0]]}_{voc_to_token[t[1]]}"
                        token_frequency[token] += 1
                    token_frequency[str(voc_to_token[term[-1]])] += 1
                    
    return token_frequency

def query_to_vector(documents):
    query_tfidf = np.zeros(shape=(len(documents), len(query_tokens)))
    for i, doc in enumerate(documents):
        term_freq = text_to_gram(doc)
        for token in term_freq:
            try:
                query_tfidf[i, token_to_index[token]] = okapi(int(term_freq[token]), len(doc)) * idf[token]
            except:
                continue
                
    return query_tfidf   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('-r', help='Use rocchio', action="store_true")
    args = parser.parse_args()
    if args.r:
        print("Using rocchio algorithm")

    all_t = time.time()
    with open("model/inverted-file") as f:
        inverted_file = [x.strip('\n\r') for x in f.readlines()]

    with open("model/file-list") as f:
        document_filenames = [os.path.basename(x.strip('\n\r')) for x in f.readlines()]

    with open("model/vocab.all") as f:
        vocab = [x.strip('\n\r') for x in f.readlines()]      

    root = ET.parse("queries/query-test.xml").getroot()   

    N = len(document_filenames)

    # Prepare documents information
    doc_ids = {}
    for i, doc in (enumerate((document_filenames))):
        doc_id = os.path.basename(doc).lower()
        doc_ids[i] = {'unique_id': doc_id, "tokens": []}

    # Find all tokens in inverted file
    token_list = []
    for i, x in enumerate(inverted_file):
        if len(x.split(" ")) == 3:
            token, uni, counts = x.split(" ")
            if int(uni) != -1:
                token = f'{token}_{uni}'    
            token_list.append(token)

    # Create the hash 
    voc_to_token, token_to_voc = {}, {}
    for i, voc in enumerate(vocab):
        voc_to_token[voc] = i
        token_to_voc[i] = voc           

    all_texts = []
    for i in range(len(root)):
        keywords = text_strip(root[i][4].text).split("、")
        title = jieba.lcut(text_strip(root[i][1].text).replace("、", ''))
        abstract = jieba.lcut(text_strip(root[i][2].text).replace("、", ''))
        # texts = jieba.lcut(''.join(text_strip(root[i][3].text)).replace("、", ''))
        all_texts += keywords + title + abstract

    token_frequency = text_to_gram(all_texts)
    query_tokens = list(set(token_frequency.keys()) & set(token_list))

    t = time.time()
    idf = {}
    document_length = Counter()
    idx = 0
    print("Creating doucment tokens..")
    while idx < len(inverted_file):
        token, uni, counts = inverted_file[idx].split(" ")
        if int(uni) != -1:
            token = f'{token}_{uni}' 
            
        for i in range(1, int(counts)+1):
            doc_id, freq = inverted_file[idx+i].split(" ")
            document_length[doc_id] += int(freq)
            
        if token in query_tokens:
            for i in range(1, int(counts)+1):
                doc_id, freq = inverted_file[idx+i].split(" ")
                doc_ids[int(doc_id)]['tokens'].append([token, freq])
                idf[token] = math.log(((N - int(counts) + 0.5) / (int(counts)+  0.5 )) + 1)
        idx += (int(counts)+1)
    print("Finish doucment tokens: ", time.time() - t)   

    # Create index for each token
    token_to_index = {}
    for i, token in enumerate(query_tokens):
        token_to_index[str(token)] = i

    # Create document tf-idf matrix
    print("Creating doucment tfidf...")
    t = time.time()
    doc_tfidf = np.zeros(shape=(N, len(query_tokens)))
    for i, doc in enumerate(doc_ids):
        doc_len = document_length[doc]
        doc_vectors = np.zeros(len(query_tokens))
        for term in doc_ids[doc]['tokens']:
            token, freq = term
            if token in query_tokens:
                tfidf = okapi(int(freq), doc_len) * idf[token]
                doc_vectors[token_to_index[token]] = tfidf
        doc_tfidf[i] = doc_vectors
    print("Finish doucment tfidf: ", time.time() - t)   

    # Parse the query text and create query tf-idf matrix
    query_texts = []
    for i in range(len(root)):
        keywords = text_strip(root[i][4].text).split("、")
        title = jieba.lcut(text_strip(root[i][1].text).replace("、", ''))
        abstract = jieba.lcut(text_strip(root[i][2].text).replace("、", ''))
        texts = jieba.lcut(''.join(text_strip(root[i][3].text)).replace("、", ''))
        query_texts.append(keywords+title+abstract)
    query_tfidf = query_to_vector(query_texts)

    all_results = []
    for i in range(len(root)):
        # Calculate cosine similiarity and get the rankings
        cos_sim  = cosine_similarity(query_tfidf[i][None], doc_tfidf)
        rankings = np.flip(cos_sim.argsort(), axis=1)

        if args.r:
            query_tfidf_ro = rocchio(query_tfidf[i][None], rankings, rel_count=10)
            cos_sim  = cosine_similarity(query_tfidf_ro, doc_tfidf)

        rankings = np.flip(cos_sim.argsort(), axis=1)
        return_ids = rankings[0][:100]

        results = ''
        for x in return_ids:
            results += (doc_ids[x]['unique_id']) + " "
        all_results.append(results)

    qids = [str.zfill(str(x), 3) for x in range(11, 31)]
    df = pd.DataFrame({"query_id":qids, "retrieved_docs":all_results})
    df.to_csv("submission.csv", index=False)       

    print("Running time: ", time.time() - all_t)
