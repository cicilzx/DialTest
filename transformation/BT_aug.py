#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import http.client
import hashlib
import urllib
import random
import json
import time


def baidu_translate(ori_query: str, toLang, fromLang='auto'):
    # appid and key need to be applied on the website: https://api.fanyi.baidu.com/
    appid = ''
    secretKey = ''
    query_arr = []
    time.sleep(1)  # reduce call frequency
    myurl = '/api/trans/vip/translate'

    salt = random.randint(32768, 65536)
    sign = appid + ori_query + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(ori_query) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        response = httpClient.getresponse()  # response is the object of HTTPResponse
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)
        # print(result)
        for each in result['trans_result']:
            query_arr.append(each['dst'])
        return query_arr

    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()
    return query_arr


def back_translate(query):
    cur_arr = []
    lan_list = "zh,fra,spa".split(",")  # Chinese, French, Spanish
    # lan_list = ["zh"]
    for tmp_lan in lan_list:
        for tmp_q in baidu_translate(query, tmp_lan):
            cur_arr.extend(baidu_translate(tmp_q, 'en'))
    out_arr = list(set(cur_arr))
    return out_arr


def augment(infile, outfile):
    with open(infile, encoding='utf-8') as f, open(outfile, 'w', encoding='utf-8') as out:
        lines = f.readlines()
        for q in lines:
            out_arr = back_translate(q)
            print(out_arr)
            out.write("\n".join(out_arr))
            out.write("\n"+"\n")


if __name__ == '__main__':
    infile = "../dataset/test/test_clean"
    outfile = "../dataset/test/test_clean.btaug"
    augment(infile, outfile)
