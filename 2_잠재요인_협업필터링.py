#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 유저 프로세스 부분 코드
# 1. json 형태로 받은 정보를 user_info와 결합하여 user_face 생성
# 2. 사용자 속성을 기반으로 코사인 유사도 
# 3. 사용자 협업 필터링
# 4. 나온 결과를 다시 DB에 저장


# In[2]:


import math
import numpy as np
import pandas as pd


# In[3]:


from scipy.spatial.distance import cosine


# In[4]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds


# In[6]:


import pymysql
from sqlalchemy import create_engine

# MySQL Connector using pymysql
pymysql.install_as_MySQLdb()
#import MySQLdb


# ## 사진을 찍은 사용자의 데이터가 들어오게 됨

# In[7]:


# lambda로 받은 데이터를 df로 만듬
import json
with open("user_data.json", "r") as st_json:
    data = json.load(st_json)
data = pd.DataFrame(data.items(), columns=['data','freq'])

# machine 번호를 받음
for i in range(len(data)) :
    if (data.loc[i][0] == 'serialnum') :
        machine = data.loc[i][1]


# In[10]:


db = pymysql.connect(
    host = '35.180.122.212', 
    port=3306, user = 'root',
    password='team09',
    db = 'mydb',
    charset='utf8'
)

SQL = "SELECT sym_id, sym_name FROM prescription_data"
sym_info = pd.read_sql(SQL,db)
SQL = "SELECT user_id FROM user_info WHERE machine_no = '{}'".format(machine)
user_info = pd.read_sql(SQL,db)
db.close()

# 진단할 수 있는 sym의 종류
sym_list = sym_info['sym_name'][1:]


# In[11]:


cnt = 1
final_df = pd.DataFrame()
final_df2 = pd.DataFrame()
have_sym = []
count_list = []

engine = create_engine("mysql://root:"+"team09"+"@35.180.122.212:3306/mydb?charset=utf8", encoding='utf8')
conn = engine.connect()
for sym in sym_list :
    # 증상 횟수 만큼 반복하게 됨
    # temp 초기화
    temp = {}
    temp2 = {}
    cnt += 1
    
    # count의 경우 여드름만 갯수이고 나머지는 유무이므로
    # 다른 케이스로 취급함
    if (sym == '구진성여드름') :
        count = (int(data.iloc[0]['freq']) + int(data.iloc[2]['freq']) + int(data.iloc[4]['freq']) + int(data.iloc[6]['freq'])
                    + int(data.iloc[8]['freq']) + int(data.iloc[10]['freq']))
        temp2 = {
            'forehead' : [data.iloc[0]['freq']],
            'cheek_R' : [data.iloc[2]['freq']],
            'cheek_L' : [data.iloc[4]['freq']],
            'nose' : [data.iloc[6]['freq']],
            'philtrum' : [data.iloc[8]['freq']],
            'chin' : [data.iloc[10]['freq']]
        }
    elif (sym == '농포성여드름') :
        count = (int(data.iloc[1]['freq']) + int(data.iloc[3]['freq']) + int(data.iloc[5]['freq']) + int(data.iloc[7]['freq'])
                    + int(data.iloc[9]['freq']) + int(data.iloc[11]['freq']))
        temp2 = {
            'forehead' : [data.iloc[1]['freq']],
            'cheek_R' : [data.iloc[3]['freq']],
            'cheek_L' : [data.iloc[5]['freq']],
            'nose' : [data.iloc[7]['freq']],
            'philtrum' : [data.iloc[9]['freq']],
            'chin' : [data.iloc[11]['freq']]
        }
    elif (sym == '기미'):
        count = int(data.iloc[12:17]['freq'].sum())
        temp2 = {
            'forehead' : [data.iloc[12]['freq']],
            'cheek_R' : [data.iloc[13]['freq']],
            'cheek_L' : [data.iloc[14]['freq']],
            'nose' : [data.iloc[15]['freq']],
            'philtrum' : [data.iloc[16]['freq']],
            'chin' : [data.iloc[17]['freq']]
        }
    elif (sym =='다크서클') :
        count = int(data.iloc[-4][1])

    temp = {
        'user_id' : [user_info['user_id'][0]],
        'sym_id' : [cnt],
        'date' : [data.iloc[-1][1]],
        'machine_no' : [data.iloc[-2][1]],
        'img_url1' : [data.iloc[-3][1]],
    }
    temp = pd.DataFrame(temp)
    temp2 = pd.DataFrame(temp2)
    count_list.append(count)
    # 만약 검출이 되지 않았다면 DB에 넣지 않게 하기 위해
    if(count == 0) :
        continue
    # 최종 user 정보를 user_face에 입력
    temp.to_sql(name='user_face',con=engine, if_exists='append', index=False)

    db = pymysql.connect(
        host = '35.180.122.212', 
        port=3306, user = 'root',
        password='team09',
        db = 'mydb',
        charset='utf8'
    )

    SQL = """SELECT MAX(user_face_id) FROM user_face """
    user_face_id = pd.read_sql(SQL,db)
    db.close()
    
    have_sym.append(sym)
    # user_face_id.iloc[0][0] 해당하는 id는 단 하나만 존재함
    if(sym =='다크서클') :
        continue
    id_info = {'user_face_id' : [user_face_id.iloc[0][0]]}
    id_info = pd.DataFrame(id_info)
    temp2 = pd.concat([temp2,id_info], axis=1)
    temp2.to_sql(name='face_detail',con=engine, if_exists='append', index=False)
    
if (have_sym == []) :
    temp = {
        'user_id' : [user_info['user_id'][0]],
        'sym_id' : [1],
        'date' : [data.iloc[-1][1]],
        'machine_no' : [data.iloc[-2][1]],
        'img_url1' : [data.iloc[-3][1]],
    }
    temp = pd.DataFrame(temp)
    temp.to_sql(name='user_face',con=engine, if_exists='append', index=False)


# # 1. 데이터 전처리 - 스킨, 로션, 에센스 분리

# In[14]:


db = pymysql.connect(
    host = '35.180.122.212', 
    port=3306, user = 'root',
    password = 'team09',
    db = 'mydb',
    charset='utf8'
)

# 크롤링 데이터 가져오기
sql = 'SELECT * FROM product_data'
crawling_df = pd.read_sql(sql,db)

db.close()


# In[16]:


#c_df = crawling_df.copy()


# In[17]:


crawling_df['category'].unique()


# In[18]:


crawling_df[crawling_df['gender']=='m']['category'].unique()


# In[19]:


rm_idx1 = crawling_df[(crawling_df['category']=='스킨/토너')&(crawling_df['gender']=='m')].index
crawling_df.drop(rm_idx1, inplace=True)
crawling_df.reset_index(drop=True, inplace=True)

rm_idx2 = crawling_df[(crawling_df['category']=='에센스/세럼')&(crawling_df['gender']=='m')].index
crawling_df.drop(rm_idx2, inplace=True)
crawling_df.reset_index(drop=True, inplace=True)

rm_idx3 = crawling_df[(crawling_df['category']=='로션/에멀젼')&(crawling_df['gender']=='m')].index
crawling_df.drop(rm_idx3, inplace=True)
crawling_df.reset_index(drop=True, inplace=True)

prod_list = crawling_df['prod_name'].unique().tolist()


for i in range(len(prod_list)):
    if crawling_df[crawling_df['prod_name']==prod_list[i]]['price'].nunique()!=1:
        unique_price = crawling_df[crawling_df['prod_name']==prod_list[i]][:1]['price'].values[0]
        crawling_df['price']=crawling_df.apply(lambda x : unique_price if (x.prod_name ==prod_list[i])&(x.price!=unique_price) else x.price ,axis=1)


for i in range(len(prod_list)):
    if crawling_df[crawling_df['prod_name']==prod_list[i]]['img_url'].nunique()!=1:
        unique_url = crawling_df[crawling_df['prod_name']==prod_list[i]][:1]['img_url'].values[0]
        crawling_df['img_url']=crawling_df.apply(lambda x : unique_url if (x.prod_name ==prod_list[i])&(x.price!=unique_url) else x.img_url ,axis=1)

for i in range(len(prod_list)):
    if crawling_df[crawling_df['prod_name']==prod_list[i]]['category'].nunique()!=1:
        unique_category = crawling_df[crawling_df['prod_name']==prod_list[i]][:1]['category'].values[0]
        crawling_df['category']=crawling_df.apply(lambda x : unique_category if (x.prod_name ==prod_list[i])&(x.price!=unique_category) else x.category ,axis=1)



# # 2. 사용자 관련 df 생성

# ### 오늘어때 사용자 데이터 들어옴

# In[21]:


db = pymysql.connect(
    host = '35.180.122.212', 
    port=3306, user = 'root',
    db = 'mydb',
    password = 'team09',
    charset='utf8'
)
user_info=[]
sql = """
    SELECT user_info.user_id, age, skin_type, gender
    FROM user_info JOIN user_face
    WHERE user_info.machine_no = '"""+str(machine)+"""'"""
user_info = pd.read_sql(sql,db)
db.close()

user_info = pd.DataFrame(user_info.loc[0]).T
user_info['acne'] = 0
user_info['dark_circle'] = 0
user_info['freckle'] = 0

for sym in have_sym :
    if(sym == '구진성여드름') :
        user_info['acne'] = 1
    elif(sym == '농포성여드름') :
        user_info['acne'] = 1
    elif(sym == '기미') :
        user_info['freckle'] = 1
    elif(sym =='다크서클') :
        user_info['dark_circle'] = 1


# In[22]:


# user_info가 원하는 사용자의 데이터가 됨 
# one-hot encoding을 통해 matrix화 시키기
user_info


# ### 오늘어때 사용자의 성별이 남자면 남자 데이터만 추출 / 여자면 여자 데이터만 추출

# In[23]:


if user_info['gender'][0]=='m':
    crawling_df = crawling_df[crawling_df['gender']=='m']
    
else :
    crawling_df = crawling_df[crawling_df['gender']=='f']
    
crawling_df.reset_index(inplace=True, drop=True)
del crawling_df['gender']
crawling_df.head()


# In[24]:


def make_user_df(df):
    df =df[['user_id','age','skin_type','acne','dark_circle','freckle']]
    
    #범주형 데이터 더미변수로 변환
    skintype_df = pd.get_dummies(df['skin_type'], prefix = 'skin_type')
    #gender_df = pd.get_dummies(df['gender'], prefix = 'gender')
    age_df = pd.get_dummies(df['age'],prefix='age')
    df = pd.concat([df,skintype_df,age_df],axis=1) 
    df.drop(['skin_type','age'],axis=1,inplace=True) # 더미 변환 이전 데이터 삭제
    
    #사용자 중복 데이터 제거
    df = df.drop_duplicates()
    df.reset_index(inplace=True,drop=True)
    df2 = df.copy()
    #del df2['user_id']
    
    return df


# In[25]:


user_df = make_user_df(crawling_df)
user_df.head()


# # 3.사용자 속성을 기반으로  코사인유사도 구하기

# In[26]:


from sklearn.metrics.pairwise import cosine_similarity


# ### 사용자 데이터와 오늘어때 사용자 데이터 합치기

# In[27]:


user_info['age'] = user_info['age'].astype(int)


# In[28]:


user_info['age_cut']=''


# In[29]:


if user_info['age'][0]<20:
    user_info['age_cut'][0]='10s'
    
elif user_info['age'][0]>=20 and user_info['age'][0]<30:
    user_info['age_cut'][0]='20s'
    
elif user_info['age'][0]>=30 and user_info['age'][0]<40:
    user_info['age_cut'][0]='30s'

elif user_info['age'][0]>=40 and user_info['age'][0]<50:
    user_info['age_cut'][0]='40s'

else:
    user_info['age_cut'][0]='50s'
    


# In[30]:


user_info


# In[31]:


age_X = pd.get_dummies(user_info['age_cut'],prefix='age')
skin_X = pd.get_dummies(user_info['skin_type'],prefix='skin_type')
#gender_X = pd.get_dummies(user_info['gender'],prefix='gender')
new_data = pd.concat([user_info,skin_X,age_X],axis=1)


# In[32]:


new_data.drop(columns=['age','skin_type','age_cut','gender'],inplace=True)


# In[33]:


new_data


# In[34]:


user_with_newdata = user_df.append(new_data).fillna(0)
user_with_newdata 


# In[35]:


del user_with_newdata['user_id']


# ## StandardScaler

# In[36]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# In[37]:


scaled_values = scaler.fit_transform(user_with_newdata)
user_with_newdata.loc[:,:] = scaled_values
user_matrix =user_with_newdata.to_numpy()


# In[38]:


# 사용자 데이터 입력 -예시) 여드름, 다크서클, 건성, 20대초반
new_data=user_matrix[-1]
len(new_data)


# In[39]:


user_matrix = np.delete(user_matrix,-1,0)


# In[40]:


cos_sim = cosine_similarity(user_matrix, new_data.reshape(1,-1))
cos_sim


# In[41]:


cos_sim_list = cos_sim.transpose().tolist()[0]


# In[42]:


cos_sim_list.sort()
cos_sim_list.reverse()


# In[43]:


similar_user_idx = np.argsort(cos_sim.transpose()[0])[::-1][:30]
similar_user_idx = similar_user_idx.tolist()


# In[44]:


similar_user_id_df = user_df.iloc[similar_user_idx,]
similar_user_id_list = similar_user_id_df[['user_id']] 
similar_user_id_list= similar_user_id_list.values.reshape(1,-1).tolist()[0]


# In[45]:


#similar_user_id_list


# In[46]:


import re


# In[47]:


idx_list_1=[]
for u_id in similar_user_id_list:
    idx = re.search('[\d ,]+',str(crawling_df[crawling_df['user_id']==u_id].index).split('[')[1]).group()
    idx_list_1.extend(idx.split(', '))


# # 4. 잠재요인 협업필터링

# In[48]:


crawling_df.info()


# In[49]:


crawling_df['score'] = crawling_df['score'].astype(int)


# In[50]:


user_prod = crawling_df.pivot_table('score', index='user_id', columns='prod_name',aggfunc='mean')
user_prod.fillna(0, inplace=True)
user_prod


# In[51]:


user_id_df = pd.DataFrame(user_prod.index)
user_id_idx_df = pd.DataFrame([i for i in range(len(user_id_df))],columns=['idx'])
user_idx_df = pd.concat([user_id_idx_df, user_id_df],axis=1)
user_idx_df


# In[52]:


# matrix는 pivot_table 값을 numpy matrix로 만든 것 
matrix = user_prod.values


# In[53]:


# user_ratings_mean은 사용자의 평균 평점 
user_ratings_mean = np.mean(matrix, axis = 1)


# In[54]:


# R_user_mean : 사용자-아이템에 대해 사용자 평균 평점을 뺀 것.
matrix_user_mean = matrix - user_ratings_mean.reshape(-1, 1)


# In[55]:


pd.DataFrame(matrix_user_mean, columns = user_prod.columns).head()


# ## 행렬 분해

# In[56]:


# scipy에서 제공해주는 svd.  
# U 행렬, sigma 행렬, V 전치 행렬을 반환.

U, sigma, Vt = svds(matrix_user_mean, k = 24)  # user k-means 군집결과 최적의 k =24


# In[57]:


print(U.shape)
print(sigma.shape)
print(Vt.shape)


# In[58]:


sigma = np.diag(sigma)
sigma.shape


# In[59]:


# U, Sigma, Vt의 내적을 수행하면, 다시 원본 행렬로 복원이 된다. 
# 거기에 + 사용자 평균 rating을 적용한다. 
svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)


# In[60]:


df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns = user_prod.columns)
df_svd_preds.head()


# In[61]:


def recommend_items(df_svd_preds, user_id, ori_df, num_recommendations=10):
    
    #현재는 index로 적용이 되어있으므로 user_id - 1을 해야함.
    user_row_number = user_idx_df[user_idx_df.user_id ==user_id].index[0]
    
    # 최종적으로 만든 pred_df에서 사용자 index에 따라 제품 데이터 정렬 -> 제품 평점이 높은 순으로 정렬 됌
    sorted_user_predictions = df_svd_preds.iloc[user_row_number].sort_values(ascending=False)

    recommendations = ori_df
    
    # 사용자의 제품 평점이 높은 순으로 정렬된 데이터와 위 recommendations을 합친다. 
    recommendations = recommendations.merge(pd.DataFrame(sorted_user_predictions).reset_index(), on = 'prod_name')

    #상품 추천을 위한 중복 제품 삭제
    recommendations.drop(['user_id','score','age','skin_type','acne','freckle','dark_circle','product_data_id'],axis=1,inplace=True)
    recommendations.drop_duplicates(inplace=True)

    # 컬럼 이름 바꾸고 정렬해서 return
    recommendations = recommendations.rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :]
                      

    return recommendations


# In[62]:


#사용자와 유사한 user가 추천받을 제품 목록
result_df = recommend_items(df_svd_preds, similar_user_id_list[0], crawling_df, 30)
for i in range(1,len(similar_user_id_list)):
    predictions = recommend_items(df_svd_preds, similar_user_id_list[i], crawling_df, 30)
    result_df = pd.concat([result_df,predictions])


# In[63]:


def recommend_each_category(df,category,num=5):
    category_df = df[df['category']==category].sort_values(by='Predictions',ascending=False)
    del category_df['Predictions']
    category_df.drop_duplicates(inplace=True)
    #category_df.reset_index(drop=True,inplace=True)
    return category_df.head(num)


# In[64]:


if user_info['gender'][0]=='m':
    cate1, cate2, cate3 = '스킨/로션','에센스/크림','올인원'
else:
    cate1, cate2, cate3 = '스킨/토너','로션/에멀젼','에센스/세럼'


# In[65]:


recommend1 = recommend_each_category(result_df,cate1)
recommend2 = recommend_each_category(result_df,cate2)
recommend3 = recommend_each_category(result_df,cate3)


# In[66]:


final_recommend=pd.concat([recommend1,recommend2,recommend3])


# In[67]:


final_recommend.reset_index(drop=True,inplace=True)
final_recommend


# In[68]:


prod_name_list =final_recommend['prod_name'].tolist()


# In[69]:


prod_id = list()
for name in prod_name_list:
    product_data_id=crawling_df[crawling_df['prod_name']==name].index[0]
    prod_id.append(crawling_df['product_data_id'][product_data_id])
final_recommend['product_data_id']=prod_id


# In[70]:


final_recommend


# In[71]:


final_recommend2 = final_recommend.copy()


# In[72]:


## 나온 결과를 INSERT할 부분임
# 3개를 합치고 datetime을 추가해서 넣어버리기
# 사진정보의 user_id / product_data_id(prod_name) / date

engine = create_engine("mysql://root:"+"team09"+"@35.180.122.212:3306/mydb?charset=utf8", encoding='utf8')
conn = engine.connect()
final_recommend = final_recommend[['product_data_id']]
final_recommend['user_id'] = user_info['user_id'][0]
final_recommend['date'] = data.iloc[-1][1]
final_recommend.to_sql(name='product_result',con=engine, if_exists='append', index=False)


# ## JSON으로 제작하여 보내는 부분

# In[110]:


temp_json = pd.DataFrame()
df = pd.DataFrame(columns = ['항목','갯수/url','주의사항/가격','제품이름'])


# ### 우선 항목을 제작
# > 여자 =  symlist + 에센스 + 로션 + 스킨 //
# > 남자 =  symlist + 로션/스킨 + 올인원

# In[111]:


# 남자 여자에 따라 항목 변경
if (user_info['gender'][0] == 'm') :
    new_list = ['acne1','acne2','frekle','darkcircle','skin/lotion','essence/cream','all_in_one']
elif (user_info['gender'][0] == 'f') :
    new_list = ['acne1','acne2','frekle','darkcircle','skin/toner','lotion/emulsion','essence/serum']


# ### 갯수/url 제작

# In[114]:


new_list2 = []
# 갯수 들어가는 부분
for i in range(0,len(count_list)) :
    new_list2.append(count_list[i])
    
# 이미지 들어가는 부분
new_list2.append(final_recommend2['img_url'][0])
new_list2.append(final_recommend2['img_url'][5])
new_list2.append(final_recommend2['img_url'][9])


# ### soultion / price 

# In[113]:


db = pymysql.connect(
    host = '35.180.122.212', 
    port=3306, user = 'root',
    password = 'team09',
    db = 'mydb',
    charset='utf8'
)

# 크롤링 데이터 가져오기
sql = 'SELECT * FROM prescription_data'
prescription = pd.read_sql(sql,db)

db.close()


# In[115]:


#sol1, sol2, sol3, sol4, price1, price2, price3
new_list3 = []
cnt = 0
temp = ""
# sol 추가

# 만약 리스트에 구진성여드름이 있다면 append하고 없으면 ''을 append
for sym in sym_list :
    cnt = cnt + 1
    if (sym in have_sym) :
        for i in range(0,len(prescription['caution'][cnt])) :
            if (prescription['caution'][cnt][i].isdigit() == True ) :
                temp += str(prescription['caution'][cnt][i]) + ','
        new_list3.append(temp)
        temp = ""
    elif(sym not in have_sym) :
        new_list3.append("")

# 제품 가격 들어가는 부분
new_list3.append(final_recommend2['price'][0])
new_list3.append(final_recommend2['price'][5])
new_list3.append(final_recommend2['price'][9])
    


# In[116]:


new_list4 = []
for i in range(0,4) :
    new_list4.append("")
    
# 제품 이름이 들어가는 부분
new_list4.append(final_recommend2['prod_name'][0])
new_list4.append(final_recommend2['prod_name'][5])
new_list4.append(final_recommend2['prod_name'][9])


# In[117]:


df = pd.DataFrame([new_list,
                   new_list2,
                   new_list3,
                   new_list4])
df = df.T
df.rename(columns= {0:"item", 1:'element1',2:'element2',3:'element3'}, inplace=True)


# In[119]:


js = df.to_json(r'data.json',orient = 'records')

