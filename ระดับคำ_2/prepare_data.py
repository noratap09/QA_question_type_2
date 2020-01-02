from gensim.models import Word2Vec
from scipy.spatial import distance

# load word2vec
model = Word2Vec.load('Word2Vec_1.model') #load model

#print(model.wv.vocab.keys())
#print("Consin:",model.wv.similarity("พรีเดเตอร์","มอนสเตอร์"))
#print("Eucli:",distance.euclidean(model.wv.get_vector("พรีเดเตอร์"), model.wv.get_vector("มอนสเตอร์")))

#import deepcut
import json

my_file = open("data_set_fix.json",'r',encoding = 'utf-8-sig')
txt = my_file.read()

json_obj = json.loads(txt)

#หา max_question_len
"""
max_question_len = 0
for data in json_obj['data']:
    if(data['question_type']==1):
        question = data['question']
        answer_begin_position = data['answer_begin_position']
        answer_end_position = data['answer_end_position']
        max_question_len = max(max_question_len,len(deepcut.tokenize(question)))
        #print(question)
print("max_question_len:",max_question_len)
max_question_len = 54
"""

import numpy as np
import math
from elasticsearch import Elasticsearch
es = Elasticsearch()

question_len = 100
input_text_len = 128
Word2Vec_len = 300
slide_size = 64 #overlap 100
num_class = 3

all_input = list()
all_output = list()

ck_point_time = 1000
last_data_count = 0

from pythainlp.tag import pos_tag, pos_tag_sents
from pythainlp.tag.named_entity import ThaiNameTagger

ner = ThaiNameTagger()
#Load POS
pos_file = open("POS_ALL.txt",'r',encoding = 'utf-8-sig')
txt = pos_file.read()
pos_all = txt.split(",")
#Load Name Enity
name_file = open("NAME_ALL.txt",'r',encoding = 'utf-8-sig')
txt = name_file.read()
name_all = txt.split(",")

pos_len = len(pos_all)
name_len = 14

def get_index_name(in_name):
        if(in_name == "B-URL" or in_name == "B-URL"): return 0
        elif(in_name == "O"): return 1
        elif(in_name == "B-PERSON" or in_name == "I-PERSON"): return 2
        elif(in_name == "B-ORGANIZATION" or in_name == "I-ORGANIZATION"): return 3
        elif(in_name == "B-PERCENT" or in_name == "I-PERCENT"): return 4
        elif(in_name == "B-DATE" or in_name == "I-DATE"): return 5
        elif(in_name == "B-MONEY" or in_name == "I-MONEY"): return 6
        elif(in_name == "B-LOCATION" or in_name == "I-LOCATION"): return 7
        elif(in_name == "B-TIME" or in_name == "I-TIME"): return 8
        elif(in_name == "B-LEN" or in_name == "I-LEN"): return 9
        elif(in_name == "B-LAW" or in_name == "I-LAW"): return 10   
        elif(in_name == "B-PHONE" or in_name == "I-PHONE"): return 11 
        elif(in_name == "B-ZIP"): return 12
        elif(in_name == "B-EMAIL" or in_name == "I-EMAIL"): return 13  
        else: return 1

#res2 = es.get(index="test_search_engine_v1",doc_type='_doc',id=str(69492))
#res2 = res2['_source']['text'][0:10]
#res2 = [x.lower() for x in res2]
#print(res2)

#save file
def ck_point(all_input,all_output,k):
        print("SAVE ",k)
        all_input = np.asarray(all_input)
        all_output = np.asarray(all_output)

        np.save("train_data\input\input_B_"+str(k),all_input)
        np.save("train_data\output\output_B_"+str(k),all_output)

for count_data , data in enumerate(json_obj['data'][0:15010],start=1):
    if(data['question_type']==2):
        question_id = data['question_id']
        print("QUESTION_ID: ",question_id)

        question = data['question'].lower()
        answer_begin_position = data['answer_begin_position']
        answer_end_position = data['answer_end_position']
        article_id = data['article_id']
        txt = es.get(index="test_search_engine_v1",doc_type='_doc',id=str(article_id))
        txt = txt['_source']['text']
        txt = [x.lower() for x in txt]

        print("TXT : ",txt)

        ck_answer = list()
        answer = data['answer']

        print("t ->",math.ceil(len(txt)/slide_size))
        for i in range(0,math.ceil(len(txt)/slide_size)):
                pre_data = np.zeros((input_text_len,Word2Vec_len+pos_len),dtype=np.float32)
                pre_ans = np.zeros((num_class),dtype=np.float32)
                input_text = txt[i*slide_size:i*slide_size+input_text_len-1]

                result_pos = pos_tag(input_text)

                print("QUESTION : ",question)
                print("INPUT TXT : ","".join(input_text))
                #print(len(txt)," > ",i,":",i*slide_size,"-",i*slide_size+input_text_len)
                #get input feature

                for n_j,j in enumerate(input_text,start=0):
                        if(j in model.wv.vocab.keys()):
                                pre_data[n_j,0:Word2Vec_len] = (model.wv.get_vector(j)+2.6491606)/(2.6491606+2.6473184)
                        pre_data[n_j,Word2Vec_len+(pos_all.index(result_pos[n_j][1]))] = 1.0
                #draw_heat_map
                import heat_map
                heat_map.make_heatmap("heatmap/"+str(question_id)+"_"+str(i)+".png",question,input_text,pre_data)

                #get output
                temp = input("INPUT ANS :")
                pre_ans[temp] = 1.0
                #0 = Yes
                #1 = midle
                #2 = No


                #add input and output to list
                all_input.append(pre_data)
                all_output.append(pre_ans)

        print("CK_ANS : ",answer)
        #check_point
        ck_point(all_input,all_output,int(question_id))
        all_input.clear()
        all_output.clear()

#save final
#ck_point(all_input,all_output,math.ceil((last_data_count+1)/ck_point_time))